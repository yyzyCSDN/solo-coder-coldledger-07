import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from coldledger.transport.lane_qualification import (
    BookingRejected,
    CarrierStrategy,
    LaneBooking,
    LaneBookingService,
    LaneQualification,
    LaneQualificationRegistry,
    QualificationEvidence,
    Quote,
    QuoteRequest,
    SnapshotConflict,
    VersionConflict,
)


UTC = timezone.utc


class LaneQualificationTests(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime(2026, 10, 1, tzinfo=UTC)
        self.t1 = self.t0 + timedelta(days=10)
        self.t2 = self.t0 + timedelta(days=20)
        self.evidence_end = self.t0 + timedelta(days=40)
        self.registry = LaneQualificationRegistry()
        self.v1 = self.registry.publish(
            LaneQualification("L1", "DE", "US"),
            self.t0,
            [CarrierStrategy("C1", Decimal("10"), Decimal("1"))],
            [QualificationEvidence("E1", "gdp", self.t0, self.evidence_end)],
            created_at=self.t0,
        )
        self.service = LaneBookingService(self.registry, timedelta(days=100))

    def test_new_quote_uses_current_snapshot_and_books_on_it(self):
        quote = self.service.quote(QuoteRequest("R1", "L1", "BIO", 5, self.t1), self.t0)
        booking = self.service.book(quote.quote_id, self.t0)
        self.assertIsInstance(booking, LaneBooking)
        self.assertEqual(booking.snapshot_digest, self.v1.digest)
        self.assertEqual(booking.carrier_id, "C1")
        self.assertEqual(booking.price, Decimal("15"))

    def test_published_version_does_not_change_existing_booking_snapshot(self):
        quote = self.service.quote(QuoteRequest("R1", "L1", "BIO", 5, self.t1), self.t0)
        booking = self.service.book(quote.quote_id, self.t0)

        self.registry.publish(
            LaneQualification("L1", "DE", "US"),
            self.t2,
            [CarrierStrategy("C1", Decimal("20"), Decimal("2"))],
            [QualificationEvidence("E2", "gdp", self.t2, self.t2 + timedelta(days=20))],
            created_at=self.t2,
        )

        stored = self.service.snapshot_for_booking(booking.booking_id)
        self.assertEqual(stored.version, 1)
        self.assertEqual(stored.strategy("C1").base_price, Decimal("10"))
        history = self.service.booking_history(booking.booking_id)
        self.assertEqual(history.snapshot.digest, booking.snapshot_digest)
        self.assertEqual([event.event_type for event in history.events], ["quoted", "booked"])

    def test_new_quote_rejects_expired_evidence_even_if_snapshot_is_effective(self):
        with self.assertRaises(BookingRejected) as caught:
            self.service.quote(
                QuoteRequest("R2", "L1", "BIO", 5, self.evidence_end + timedelta(days=1)),
                self.t0,
            )
        self.assertEqual(caught.exception.reason_codes[0], "expired_evidence")

    def test_stale_unbooked_quote_conflict_is_rejected_by_default(self):
        quote = self.service.quote(QuoteRequest("R2", "L1", "BIO", 5, self.t1), self.t0)
        self.registry.publish(
            LaneQualification("L1", "DE", "US"),
            self.t2,
            [CarrierStrategy("C1", Decimal("20"), Decimal("2"))],
            [QualificationEvidence("E2", "gdp", self.t2, self.t2 + timedelta(days=20))],
            created_at=self.t2,
        )
        with self.assertRaises(SnapshotConflict):
            self.service.book(quote.quote_id, self.t2)

    def test_explicit_requote_returns_a_quote_and_does_not_book_silently(self):
        quote = self.service.quote(QuoteRequest("R2", "L1", "BIO", 5, self.t1), self.t0)
        self.registry.publish(
            LaneQualification("L1", "DE", "US"),
            self.t2,
            [CarrierStrategy("C1", Decimal("20"), Decimal("2"))],
            [QualificationEvidence("E2", "gdp", self.t2, self.t2 + timedelta(days=20))],
            created_at=self.t2,
        )
        revised = self.service.book(quote.quote_id, self.t2, on_conflict="requote")
        self.assertIsInstance(revised, Quote)
        self.assertNotEqual(revised.quote_id, quote.quote_id)
        self.assertEqual(revised.revision_of, quote.quote_id)
        self.assertEqual(revised.price, Decimal("30"))
        self.assertEqual(revised.snapshot_version, 2)
        self.assertEqual(self.service.book(revised.quote_id, self.t2).snapshot_version, 2)

    def test_expected_version_guards_publishing(self):
        with self.assertRaises(VersionConflict):
            self.registry.publish(
                LaneQualification("L1", "DE", "US"),
                self.t2,
                [CarrierStrategy("C1")],
                created_at=self.t2,
                expected_version=99,
            )

    def test_requote_when_new_version_makes_evidence_unusable_is_an_explicit_rejection(self):
        quote = self.service.quote(QuoteRequest("R3", "L1", "BIO", 5, self.t1), self.t0)
        self.registry.publish(
            LaneQualification("L1", "DE", "US", status="suspended"),
            self.t2,
            [],
            created_at=self.t2,
        )
        with self.assertRaises(BookingRejected) as caught:
            self.service.book(quote.quote_id, self.t2, on_conflict="requote")
        self.assertEqual(caught.exception.reason_code, "lane_suspended")

    def test_as_of_history_resolves_the_correct_effective_version(self):
        v2 = self.registry.publish(
            LaneQualification("L1", "DE", "US"),
            self.t2,
            [CarrierStrategy("C1", Decimal("20"))],
            [QualificationEvidence("E2", "gdp", self.t2, self.t2 + timedelta(days=20))],
            created_at=self.t2,
        )
        self.assertEqual(self.registry.snapshot_at("L1", self.t2 - timedelta(seconds=1)).digest, self.v1.digest)
        self.assertEqual(self.registry.snapshot_at("L1", self.t2).digest, v2.digest)
        self.assertEqual(self.registry.version("L1", 1).digest, self.v1.digest)


if __name__ == "__main__":
    unittest.main()
