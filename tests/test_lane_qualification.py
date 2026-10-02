import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from coldledger.transport.carrier_policy import CarrierPolicy, Rule
from coldledger.transport.lane_qualification import (
    BookingState,
    Cargo,
    ConflictAction,
    Evidence,
    LaneBookingService,
    LaneQualificationRegistry,
    LaneRequirements,
    NotQualified,
    QualificationVersion,
    build_snapshot,
    verify,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def at(days=0, hours=0):
    return T0 + timedelta(days=days, hours=hours)


def evidence(valid_to=None, key="lane-audit"):
    return Evidence(key, "qp", T0, valid_to or at(days=60))


def requirements(temp=(-25.0, -15.0), transit=48.0, caps=("gdp", "dry-ice")):
    return LaneRequirements(temp[0], temp[1], transit, frozenset(caps))


def version(n=1, effective_from=None, rate="2.50", req=None, ev=None, lane="L1"):
    return QualificationVersion(
        lane,
        n,
        effective_from or T0,
        req or requirements(),
        (ev or evidence(),),
        Decimal(rate),
    )


def cargo(caps=("gdp",), temp=(-22.0, -18.0), transit=24.0):
    return Cargo("C1", temp[0], temp[1], transit, Decimal("100"), frozenset(caps))


def service(registry, policy=None, ttl=30 * 24 * 3600):
    return LaneBookingService(registry, policy, quote_ttl_seconds=ttl)


class RegistryTests(unittest.TestCase):
    def test_publish_closes_previous_and_resolves_across_time(self):
        reg = LaneQualificationRegistry()
        reg.publish(version(1, T0))
        reg.publish(version(2, at(days=10)))
        self.assertEqual(reg.version_at("L1", at(days=5)).version, 1)
        self.assertEqual(reg.version_at("L1", at(days=10)).version, 2)
        self.assertEqual(reg.history("L1")[0].effective_to, at(days=10))
        self.assertIsNone(reg.version_at("L1", at(days=-1)))
        self.assertIsNone(reg.version_at("UNKNOWN", at(days=1)))

    def test_publish_rejects_overlap_regression_and_gaps_allowed(self):
        reg = LaneQualificationRegistry()
        reg.publish(version(1, T0))
        with self.assertRaises(ValueError):
            reg.publish(version(1, at(days=5)))  # version number must increase
        with self.assertRaises(ValueError):
            reg.publish(version(2, T0))  # effective_from must move forward
        reg.publish(version(2, at(days=10)))
        with self.assertRaises(ValueError):
            reg.publish(version(3, at(days=5)))  # before previous window start
        reg2 = LaneQualificationRegistry()
        reg2.publish(version(1, T0))
        reg2.publish(version(2, at(days=30)))  # abutting supersede: no gap
        self.assertEqual(reg2.version_at("L1", at(days=15)).version, 1)
        reg3 = LaneQualificationRegistry()
        reg3.publish(
            QualificationVersion("L1", 1, T0, requirements(), (evidence(),), Decimal("2.50"), effective_to=at(days=10))
        )
        reg3.publish(version(2, at(days=30)))
        self.assertIsNone(reg3.version_at("L1", at(days=15)))  # explicit gap

    def test_version_validation(self):
        with self.assertRaises(ValueError):
            QualificationVersion(
                "L1", 1, T0, requirements(), (Evidence("a", "qp", T0, T0),), Decimal("1")
            ).validate()  # evidence window must be positive
        with self.assertRaises(ValueError):
            QualificationVersion("L1", 1, T0, requirements(), (), Decimal("1")).validate()
        with self.assertRaises(ValueError):
            QualificationVersion(
                "L1", 1, T0, requirements(), (evidence(),), Decimal("1"), effective_to=T0
            ).validate()
        with self.assertRaises(ValueError):
            Evidence("a", "qp", T0, T0).validate()
        with self.assertRaises(ValueError):
            Evidence("a", "qp", datetime(2026, 1, 1), at(days=1)).validate()  # naive datetime


class QuoteAndBookTests(unittest.TestCase):
    def setUp(self):
        self.reg = LaneQualificationRegistry()
        self.reg.publish(version(1, T0))
        self.svc = service(self.reg)

    def test_quote_pins_content_addressed_snapshot(self):
        q1 = self.svc.quote("L1", cargo(), at(days=2), {}, at(days=1))
        q2 = self.svc.quote("L1", cargo(), at(days=2), {}, at(days=1, hours=1))
        self.assertEqual(q1.snapshot.version, 1)
        self.assertEqual(q1.snapshot.digest, q2.snapshot.digest)  # captured_at excluded
        self.assertEqual(q1.price, Decimal("2.50") * Decimal("100"))
        self.assertTrue(verify(q1.snapshot))

    def test_quote_rejects_expired_evidence_and_unfit_cargo(self):
        reg = LaneQualificationRegistry()
        reg.publish(version(1, T0, ev=evidence(valid_to=at(days=5))))
        svc = service(reg)
        with self.assertRaises(NotQualified) as ctx:
            svc.quote("L1", cargo(), at(days=10), {}, at(days=1))
        self.assertIn("lane-audit", str(ctx.exception))
        with self.assertRaises(NotQualified) as ctx:
            self.svc.quote("L1", cargo(caps=("gdp", "gps")), at(days=2), {}, at(days=1))
        self.assertIn("gps", str(ctx.exception))
        with self.assertRaises(NotQualified):
            self.svc.quote("L1", cargo(temp=(-30.0, -18.0)), at(days=2), {}, at(days=1))

    def test_quote_unknown_lane_not_qualified(self):
        with self.assertRaises(NotQualified):
            self.svc.quote("NOWHERE", cargo(), at(days=2), {}, at(days=1))

    def test_book_confirms_and_pins_snapshot(self):
        q = self.svc.quote("L1", cargo(), at(days=2), {}, at(days=1))
        outcome = self.svc.book("B1", q, at(days=1, hours=2))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.booking.state, BookingState.CONFIRMED)
        self.assertEqual(outcome.booking.snapshot.digest, q.snapshot.digest)
        with self.assertRaises(ValueError):
            self.svc.book("B1", q, at(days=1, hours=3))  # duplicate reference

    def test_confirmed_booking_immune_to_new_version_but_advisory_reported(self):
        q = self.svc.quote("L1", cargo(), at(days=20), {}, at(days=1))
        booked = self.svc.book("B1", q, at(days=1)).booking
        report = self.svc.publish_version(version(2, at(days=10), rate="3.10"), at(days=9))
        after = self.svc.booking("B1")
        self.assertEqual(after.state, BookingState.CONFIRMED)
        self.assertEqual(after.snapshot.digest, booked.snapshot.digest)  # not silently changed
        self.assertEqual(after.price, Decimal("250.00"))
        self.assertEqual(report.advisories[0].reference, "B1")  # drift is explicit, not silent
        self.assertEqual(self.svc.qualification_of("B1").version, 1)

    def test_book_after_version_change_requotes_when_still_qualified(self):
        q = self.svc.quote("L1", cargo(), at(days=20), {}, at(days=1))
        self.svc.publish_version(version(2, at(days=10), rate="3.10"), at(days=9))
        outcome = self.svc.book("B1", q, at(days=11))
        self.assertIsNone(outcome.booking)
        self.assertEqual(outcome.conflict.action, ConflictAction.REQUOTE)
        self.assertTrue(any("rate changed" in r for r in outcome.conflict.reasons))
        replacement = outcome.conflict.replacement
        self.assertEqual(replacement.snapshot.version, 2)
        self.assertEqual(replacement.price, Decimal("3.10") * Decimal("100"))
        redone = self.svc.book("B1", replacement, at(days=11, hours=1))
        self.assertTrue(redone.ok)
        self.assertEqual(redone.booking.snapshot.version, 2)

    def test_book_after_version_change_rejects_when_no_longer_qualified(self):
        q = self.svc.quote("L1", cargo(), at(days=20), {}, at(days=1))
        narrower = requirements(temp=(-21.0, -15.0))
        self.svc.publish_version(version(2, at(days=10), req=narrower), at(days=9))
        outcome = self.svc.book("B1", q, at(days=11))
        self.assertEqual(outcome.conflict.action, ConflictAction.REJECT)
        self.assertTrue(any("outside qualified" in r for r in outcome.conflict.reasons))
        self.assertIsNone(outcome.conflict.replacement)

    def test_book_rejects_when_evidence_lapses_before_departure(self):
        q = self.svc.quote("L1", cargo(), at(days=20), {}, at(days=1))
        self.svc.publish_version(version(2, at(days=10), ev=evidence(valid_to=at(days=15))), at(days=9))
        outcome = self.svc.book("B1", q, at(days=11))
        self.assertEqual(outcome.conflict.action, ConflictAction.REJECT)
        self.assertTrue(any("lane-audit" in r for r in outcome.conflict.reasons))

    def test_expired_quote_is_rejected(self):
        svc = service(self.reg, ttl=60)
        q = svc.quote("L1", cargo(), at(days=2), {}, at(days=1))
        outcome = svc.book("B1", q, at(days=1, hours=2))
        self.assertEqual(outcome.conflict.action, ConflictAction.REJECT)
        self.assertIn("expired", outcome.conflict.reasons[0])


class HoldSweepTests(unittest.TestCase):
    def setUp(self):
        self.reg = LaneQualificationRegistry()
        self.reg.publish(version(1, T0))
        self.svc = service(self.reg)

    def hold(self, reference, departure, caps=("gdp",)):
        q = self.svc.quote("L1", cargo(caps=caps), departure, {}, at(days=1))
        outcome = self.svc.hold(reference, q, at(days=1))
        self.assertTrue(outcome.ok)
        return outcome.booking

    def test_publish_sweeps_holds_explicitly(self):
        self.hold("H1", at(days=20))  # governed by v2 after publish
        self.hold("H2", at(days=5))  # still governed by v1
        report = self.svc.publish_version(version(2, at(days=10), rate="3.10"), at(days=9))
        self.assertEqual(report.requote_required, ("H1",))
        self.assertEqual(report.rejected, ())
        self.assertEqual(self.svc.booking("H1").state, BookingState.REQUOTE_REQUIRED)
        self.assertEqual(self.svc.booking("H2").state, BookingState.HELD)
        self.assertEqual(self.svc.booking("H2").snapshot.version, 1)

    def test_publish_rejects_hold_when_capability_removed(self):
        self.hold("H1", at(days=20), caps=("gdp", "dry-ice"))
        weaker = requirements(caps=("gdp",))
        report = self.svc.publish_version(version(2, at(days=10), req=weaker), at(days=9))
        self.assertEqual(report.rejected, ("H1",))
        booking = self.svc.booking("H1")
        self.assertEqual(booking.state, BookingState.REJECTED)
        self.assertEqual(booking.history[-1].action, "rejected")
        with self.assertRaises(ValueError):
            self.svc.confirm("H1", at(days=9))

    def test_requote_then_confirm_restores_hold(self):
        self.hold("H1", at(days=20))
        self.svc.publish_version(version(2, at(days=10), rate="3.10"), at(days=9))
        quote = self.svc.requote("H1", at(days=9, hours=1))
        self.assertEqual(quote.snapshot.version, 2)
        self.assertEqual(self.svc.booking("H1").state, BookingState.HELD)
        outcome = self.svc.confirm("H1", at(days=9, hours=2))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.booking.state, BookingState.CONFIRMED)
        self.assertEqual(outcome.booking.snapshot.version, 2)
        actions = [e.action for e in outcome.booking.history]
        self.assertEqual(actions, ["held", "requote_required", "requoted", "confirmed"])

    def test_confirm_detects_policy_drift_without_publish(self):
        # Carrier policy is evaluated at confirm time; a rule expiring between
        # hold and confirm turns the confirm into an explicit rejection.
        rule = Rule("carrier", "eq", "ACME", 1, T0, at(days=10), scope="L1")
        svc = service(self.reg, CarrierPolicy([rule]))
        q = svc.quote("L1", cargo(), at(days=20), {"carrier": "ACME"}, at(days=1))
        self.assertTrue(svc.hold("H1", q, at(days=1)).ok)
        outcome = svc.confirm("H1", at(days=11))
        self.assertEqual(outcome.conflict.action, ConflictAction.REJECT)
        self.assertEqual(outcome.booking.state, BookingState.REJECTED)

    def test_cancel_and_terminal_guards(self):
        self.hold("H1", at(days=20))
        cancelled = self.svc.cancel("H1", at(days=2))
        self.assertEqual(cancelled.state, BookingState.CANCELLED)
        with self.assertRaises(ValueError):
            self.svc.confirm("H1", at(days=2))
        with self.assertRaises(ValueError):
            self.svc.requote("H1", at(days=2))
        with self.assertRaises(ValueError):
            self.svc.qualification_of("MISSING")


class PolicyBindingTests(unittest.TestCase):
    def policy(self, expires_at=None):
        rule = Rule("carrier", "eq", "ACME", 1, T0, expires_at, scope="L1")
        return CarrierPolicy([rule])

    def test_policy_decision_bound_into_snapshot(self):
        reg = LaneQualificationRegistry()
        reg.publish(version(1, T0))
        svc = service(reg, self.policy())
        q = svc.quote("L1", cargo(), at(days=2), {"carrier": "ACME"}, at(days=1))
        self.assertEqual([p.key for p in q.snapshot.policy], ["carrier"])
        self.assertTrue(q.snapshot.policy[0].allowed)
        outcome = svc.book("B1", q, at(days=1, hours=1))
        self.assertTrue(outcome.ok)

    def test_policy_denial_at_quote_time(self):
        reg = LaneQualificationRegistry()
        reg.publish(version(1, T0))
        svc = service(reg, self.policy())
        with self.assertRaises(NotQualified) as ctx:
            svc.quote("L1", cargo(), at(days=2), {"carrier": "OTHER"}, at(days=1))
        self.assertIn("carrier", str(ctx.exception))

    def test_policy_rule_expiry_rejects_booking(self):
        reg = LaneQualificationRegistry()
        reg.publish(version(1, T0))
        svc = service(reg, self.policy(expires_at=at(days=10)))
        q = svc.quote("L1", cargo(), at(days=12), {"carrier": "ACME"}, at(days=1))
        outcome = svc.book("B1", q, at(days=11))  # rule expired meanwhile
        self.assertEqual(outcome.conflict.action, ConflictAction.REJECT)
        self.assertTrue(any("carrier policy" in r for r in outcome.conflict.reasons))


class HistoryAndIntegrityTests(unittest.TestCase):
    def test_historical_queries_share_snapshot_type(self):
        reg = LaneQualificationRegistry()
        reg.publish(version(1, T0))
        svc = service(reg)
        q = svc.quote("L1", cargo(), at(days=20), {}, at(days=1))
        svc.book("B1", q, at(days=1))
        svc.publish_version(version(2, at(days=10), rate="3.10"), at(days=9))
        past = svc.qualification_as_of("L1", at(days=5))
        now = svc.qualification_as_of("L1", at(days=15))
        self.assertEqual(past.version, 1)
        self.assertEqual(now.version, 2)
        self.assertEqual(past.digest, svc.qualification_of("B1").digest)
        self.assertNotEqual(now.digest, svc.qualification_of("B1").digest)
        self.assertIsNone(svc.qualification_as_of("L1", at(days=-5)))

    def test_snapshot_digest_is_tamper_evident(self):
        reg = LaneQualificationRegistry()
        v1 = version(1, T0)
        reg.publish(v1)
        snap = build_snapshot(v1, (), at(days=1))
        self.assertTrue(verify(snap))
        forged = replace(snap, rate_per_kg=Decimal("0.01"))
        self.assertFalse(verify(forged))
        rotated = replace(snap, evidence=(evidence(key="other-audit"),))
        self.assertFalse(verify(rotated))


if __name__ == "__main__":
    unittest.main()
