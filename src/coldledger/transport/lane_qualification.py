from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from threading import RLock
from typing import Iterable, Mapping

from ..core.clock import utc
from .carrier_policy import CarrierPolicy

UTC = timezone.utc
OPEN_ENDED = datetime.max.replace(tzinfo=UTC)


class ConflictAction(str, Enum):
    """Explicit outcome when a pinned snapshot and the current version disagree."""

    REJECT = "reject"
    REQUOTE = "requote"


class NotQualified(ValueError):
    """The lane cannot qualify the shipment at all; carries machine-readable reasons."""

    def __init__(self, reasons: Iterable[str]):
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class Evidence:
    """Time-boxed qualification evidence (audits, certificates). Half-open window."""

    key: str
    issuer: str
    valid_from: datetime
    valid_to: datetime

    def validate(self) -> None:
        if not self.key or not self.issuer:
            raise ValueError("evidence key and issuer required")
        if utc(self.valid_to) <= utc(self.valid_from):
            raise ValueError("evidence validity window must be positive")

    def covers(self, at: datetime) -> bool:
        return utc(self.valid_from) <= utc(at) < utc(self.valid_to)


@dataclass(frozen=True)
class LaneRequirements:
    temp_min_c: float
    temp_max_c: float
    max_transit_hours: float
    capabilities: frozenset[str] = frozenset()

    def validate(self) -> None:
        if not self.temp_min_c < self.temp_max_c:
            raise ValueError("temperature band must be positive")
        if self.max_transit_hours <= 0:
            raise ValueError("max transit hours must be positive")


@dataclass(frozen=True)
class Cargo:
    reference: str
    temp_min_c: float
    temp_max_c: float
    transit_hours: float
    weight_kg: Decimal
    capabilities_needed: frozenset[str] = frozenset()

    def validate(self) -> None:
        if not self.reference:
            raise ValueError("cargo reference required")
        if not self.temp_min_c < self.temp_max_c:
            raise ValueError("cargo temperature band must be positive")
        if self.transit_hours <= 0:
            raise ValueError("transit hours must be positive")
        if self.weight_kg <= 0:
            raise ValueError("weight must be positive")


def fit_violations(requirements: LaneRequirements, cargo: Cargo) -> list[str]:
    out = []
    if cargo.temp_min_c < requirements.temp_min_c or cargo.temp_max_c > requirements.temp_max_c:
        out.append(
            f"cargo band {cargo.temp_min_c}..{cargo.temp_max_c}C outside "
            f"qualified {requirements.temp_min_c}..{requirements.temp_max_c}C"
        )
    if cargo.transit_hours > requirements.max_transit_hours:
        out.append(f"transit {cargo.transit_hours}h exceeds qualified {requirements.max_transit_hours}h")
    missing = sorted(cargo.capabilities_needed - requirements.capabilities)
    if missing:
        out.append(f"missing capabilities: {', '.join(missing)}")
    return out


@dataclass(frozen=True)
class QualificationVersion:
    """One effective-dated qualification window for a lane."""

    lane_id: str
    version: int
    effective_from: datetime
    requirements: LaneRequirements
    evidence: tuple[Evidence, ...]
    rate_per_kg: Decimal
    effective_to: datetime | None = None

    def validate(self) -> None:
        if not self.lane_id:
            raise ValueError("lane id required")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        utc(self.effective_from)
        self.requirements.validate()
        if not self.evidence:
            raise ValueError("at least one evidence item required")
        for e in self.evidence:
            e.validate()
        keys = [e.key for e in self.evidence]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate evidence key")
        if self.rate_per_kg < 0:
            raise ValueError("rate must be non-negative")
        if self.effective_to is not None and utc(self.effective_to) <= utc(self.effective_from):
            raise ValueError("effective window must be positive")

    def governs(self, at: datetime) -> bool:
        end = utc(self.effective_to) if self.effective_to else OPEN_ENDED
        return utc(self.effective_from) <= utc(at) < end


class LaneQualificationRegistry:
    """Append-only, effective-dated qualification history per lane."""

    def __init__(self) -> None:
        self._lanes: dict[str, list[QualificationVersion]] = {}
        self._lock = RLock()

    def publish(self, version: QualificationVersion) -> QualificationVersion:
        version.validate()
        with self._lock:
            rows = self._lanes.setdefault(version.lane_id, [])
            if rows:
                last = rows[-1]
                if version.version <= last.version:
                    raise ValueError("version number must increase per lane")
                if utc(version.effective_from) <= utc(last.effective_from):
                    raise ValueError("effective_from must be after previous window start")
                if last.effective_to is None:
                    rows[-1] = replace(last, effective_to=utc(version.effective_from))
                elif utc(version.effective_from) < utc(last.effective_to):
                    raise ValueError("overlapping qualification windows")
            rows.append(version)
            return version

    def version_at(self, lane_id: str, at: datetime) -> QualificationVersion | None:
        with self._lock:
            for row in self._lanes.get(lane_id, []):
                if row.governs(at):
                    return row
            return None

    def history(self, lane_id: str) -> tuple[QualificationVersion, ...]:
        with self._lock:
            return tuple(self._lanes.get(lane_id, ()))


@dataclass(frozen=True)
class PolicyRecord:
    key: str
    allowed: bool
    explanation: str


def _payload(
    lane_id: str,
    version: int,
    requirements: LaneRequirements,
    evidence: tuple[Evidence, ...],
    rate_per_kg: Decimal,
    policy: tuple[PolicyRecord, ...],
) -> list:
    return [
        lane_id,
        version,
        [
            requirements.temp_min_c,
            requirements.temp_max_c,
            requirements.max_transit_hours,
            sorted(requirements.capabilities),
        ],
        [
            [e.key, e.issuer, utc(e.valid_from).isoformat(), utc(e.valid_to).isoformat()]
            for e in sorted(evidence, key=lambda x: x.key)
        ],
        str(rate_per_kg),
        [[p.key, p.allowed, p.explanation] for p in sorted(policy, key=lambda x: x.key)],
    ]


def _digest(*args) -> str:
    raw = json.dumps(_payload(*args), separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class QualificationSnapshot:
    """Immutable qualification basis shared by bookings, carrier policy and history queries.

    The digest is content-addressed (captured_at excluded) so the same qualification
    content rebuilt at a later time compares equal.
    """

    lane_id: str
    version: int
    requirements: LaneRequirements
    evidence: tuple[Evidence, ...]
    rate_per_kg: Decimal
    policy: tuple[PolicyRecord, ...]
    captured_at: datetime
    digest: str


def build_snapshot(
    version: QualificationVersion,
    policy: Iterable[PolicyRecord],
    captured_at: datetime,
) -> QualificationSnapshot:
    evidence = tuple(sorted(version.evidence, key=lambda e: e.key))
    records = tuple(sorted(policy, key=lambda p: p.key))
    digest = _digest(version.lane_id, version.version, version.requirements, evidence, version.rate_per_kg, records)
    return QualificationSnapshot(
        version.lane_id,
        version.version,
        version.requirements,
        evidence,
        version.rate_per_kg,
        records,
        utc(captured_at),
        digest,
    )


def verify(snapshot: QualificationSnapshot) -> bool:
    expected = _digest(
        snapshot.lane_id,
        snapshot.version,
        snapshot.requirements,
        snapshot.evidence,
        snapshot.rate_per_kg,
        snapshot.policy,
    )
    return expected == snapshot.digest


@dataclass(frozen=True)
class Quote:
    snapshot: QualificationSnapshot
    cargo: Cargo
    departure_at: datetime
    facts: tuple[tuple[str, str], ...]
    price: Decimal
    valid_until: datetime


class BookingState(str, Enum):
    HELD = "held"
    CONFIRMED = "confirmed"
    REQUOTE_REQUIRED = "requote_required"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BookingEvent:
    at: datetime
    action: str
    detail: str


@dataclass(frozen=True)
class Booking:
    reference: str
    lane_id: str
    cargo: Cargo
    departure_at: datetime
    snapshot: QualificationSnapshot
    price: Decimal
    state: BookingState
    facts: tuple[tuple[str, str], ...]
    history: tuple[BookingEvent, ...]


@dataclass(frozen=True)
class Conflict:
    action: ConflictAction
    reasons: tuple[str, ...]
    replacement: Quote | None = None


@dataclass(frozen=True)
class Outcome:
    booking: Booking | None = None
    conflict: Conflict | None = None

    @property
    def ok(self) -> bool:
        return self.booking is not None and self.conflict is None


@dataclass(frozen=True)
class Advisory:
    """Post-publish warning about a confirmed booking; the booking itself is never mutated."""

    reference: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SweepReport:
    lane_id: str
    version: int
    requote_required: tuple[str, ...]
    rejected: tuple[str, ...]
    advisories: tuple[Advisory, ...]


def diff_reasons(old: QualificationSnapshot, new: QualificationSnapshot) -> list[str]:
    reasons = []
    if new.version != old.version:
        reasons.append(f"qualification version changed {old.version} -> {new.version}")
    if new.requirements != old.requirements:
        reasons.append("qualified requirements changed")
    if new.rate_per_kg != old.rate_per_kg:
        reasons.append(f"rate changed {old.rate_per_kg} -> {new.rate_per_kg} per kg")
    if old.evidence != new.evidence:
        reasons.append("evidence set rotated")
    if old.policy != new.policy:
        reasons.append("carrier policy decision changed")
    if not reasons:
        reasons.append("qualification snapshot changed")
    return reasons


def _freeze_facts(facts: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in facts.items()))


class LaneBookingService:
    """Quotes and books cargo against effective-dated lane qualifications.

    Every quote, booking and historical answer is pinned to a QualificationSnapshot.
    Publishing a new version never mutates existing bookings; conflicts surface as
    explicit REJECT (lane can no longer qualify the shipment) or REQUOTE (still
    qualified, but the commercial/contractual basis changed).
    """

    def __init__(
        self,
        registry: LaneQualificationRegistry,
        policy: CarrierPolicy | None = None,
        quote_ttl_seconds: float = 3600,
    ) -> None:
        if quote_ttl_seconds <= 0:
            raise ValueError("quote ttl must be positive")
        self.registry = registry
        self.policy = policy
        self.quote_ttl = timedelta(seconds=quote_ttl_seconds)
        self._bookings: dict[str, Booking] = {}
        self._lock = RLock()

    def _policy_records(self, facts: tuple[tuple[str, str], ...], at: datetime, lane_id: str) -> tuple[PolicyRecord, ...]:
        if self.policy is None:
            return ()
        data = dict(facts)
        keys = sorted({r.key for r in self.policy.rules if r.scope in {"*", lane_id}})
        records = []
        for key in keys:
            decision = self.policy.decide(key, data.get(key), at, lane_id)
            records.append(PolicyRecord(key, decision.allowed, decision.explanation))
        return tuple(records)

    def _snapshot(
        self,
        lane_id: str,
        cargo: Cargo,
        facts: tuple[tuple[str, str], ...],
        departure_at: datetime,
        at: datetime,
    ) -> QualificationSnapshot:
        version = self.registry.version_at(lane_id, departure_at)
        if version is None:
            raise NotQualified([f"lane {lane_id} has no qualification version effective at departure"])
        stale = [e.key for e in version.evidence if not e.covers(departure_at)]
        if stale:
            raise NotQualified([f"evidence {key} not valid at departure" for key in stale])
        violations = fit_violations(version.requirements, cargo)
        if violations:
            raise NotQualified(violations)
        records = self._policy_records(facts, at, lane_id)
        denied = [f"carrier policy denies {p.key}: {p.explanation}" for p in records if not p.allowed]
        if denied:
            raise NotQualified(denied)
        return build_snapshot(version, records, at)

    def quote(
        self,
        lane_id: str,
        cargo: Cargo,
        departure_at: datetime,
        facts: Mapping[str, object],
        at: datetime,
    ) -> Quote:
        cargo.validate()
        at = utc(at)
        frozen = _freeze_facts(facts)
        snapshot = self._snapshot(lane_id, cargo, frozen, utc(departure_at), at)
        return Quote(
            snapshot,
            cargo,
            utc(departure_at),
            frozen,
            snapshot.rate_per_kg * cargo.weight_kg,
            at + self.quote_ttl,
        )

    def _resolve_quote(self, quote: Quote, at: datetime) -> QualificationSnapshot | Conflict:
        if utc(at) >= quote.valid_until:
            return Conflict(ConflictAction.REJECT, ("quote expired; evidence must be re-validated",))
        try:
            snapshot = self._snapshot(quote.snapshot.lane_id, quote.cargo, quote.facts, quote.departure_at, at)
        except NotQualified as exc:
            return Conflict(ConflictAction.REJECT, exc.reasons)
        if snapshot.digest == quote.snapshot.digest:
            return snapshot
        replacement = Quote(
            snapshot,
            quote.cargo,
            quote.departure_at,
            quote.facts,
            snapshot.rate_per_kg * quote.cargo.weight_kg,
            utc(at) + self.quote_ttl,
        )
        return Conflict(ConflictAction.REQUOTE, tuple(diff_reasons(quote.snapshot, snapshot)), replacement)

    def _resolve_booking(self, booking: Booking, at: datetime) -> QualificationSnapshot | Conflict:
        try:
            snapshot = self._snapshot(booking.lane_id, booking.cargo, booking.facts, booking.departure_at, at)
        except NotQualified as exc:
            return Conflict(ConflictAction.REJECT, exc.reasons)
        if snapshot.digest == booking.snapshot.digest:
            return snapshot
        return Conflict(ConflictAction.REQUOTE, tuple(diff_reasons(booking.snapshot, snapshot)))

    def _get(self, reference: str) -> Booking:
        try:
            return self._bookings[reference]
        except KeyError:
            raise ValueError(f"unknown booking {reference}") from None

    def _store(self, booking: Booking) -> Booking:
        self._bookings[booking.reference] = booking
        return booking

    def book(self, reference: str, quote: Quote, at: datetime) -> Outcome:
        """One-shot confirm. On conflict nothing is recorded; the Conflict is the explicit answer."""
        with self._lock:
            if reference in self._bookings:
                raise ValueError("duplicate booking reference")
            resolved = self._resolve_quote(quote, at)
            if isinstance(resolved, Conflict):
                return Outcome(conflict=resolved)
            booking = Booking(
                reference,
                quote.snapshot.lane_id,
                quote.cargo,
                quote.departure_at,
                resolved,
                resolved.rate_per_kg * quote.cargo.weight_kg,
                BookingState.CONFIRMED,
                quote.facts,
                (BookingEvent(utc(at), "confirmed", f"snapshot v{resolved.version} digest {resolved.digest[:12]}"),),
            )
            return Outcome(booking=self._store(booking))

    def hold(self, reference: str, quote: Quote, at: datetime) -> Outcome:
        """Pin a snapshot without confirming; later drift is settled by confirm/publish."""
        with self._lock:
            if reference in self._bookings:
                raise ValueError("duplicate booking reference")
            resolved = self._resolve_quote(quote, at)
            if isinstance(resolved, Conflict):
                return Outcome(conflict=resolved)
            booking = Booking(
                reference,
                quote.snapshot.lane_id,
                quote.cargo,
                quote.departure_at,
                resolved,
                resolved.rate_per_kg * quote.cargo.weight_kg,
                BookingState.HELD,
                quote.facts,
                (BookingEvent(utc(at), "held", f"snapshot v{resolved.version} digest {resolved.digest[:12]}"),),
            )
            return Outcome(booking=self._store(booking))

    def confirm(self, reference: str, at: datetime) -> Outcome:
        with self._lock:
            booking = self._get(reference)
            if booking.state is BookingState.CONFIRMED:
                return Outcome(booking=booking)
            if booking.state in (BookingState.CANCELLED, BookingState.REJECTED):
                raise ValueError(f"booking {reference} is {booking.state.value}")
            resolved = self._resolve_booking(booking, at)
            if isinstance(resolved, Conflict):
                if resolved.action is ConflictAction.REJECT:
                    updated = self._transition(booking, BookingState.REJECTED, at, "; ".join(resolved.reasons))
                else:
                    updated = self._transition(booking, BookingState.REQUOTE_REQUIRED, at, "; ".join(resolved.reasons))
                return Outcome(booking=updated, conflict=resolved)
            updated = replace(
                booking,
                state=BookingState.CONFIRMED,
                history=booking.history + (BookingEvent(utc(at), "confirmed", f"snapshot v{resolved.version}"),),
            )
            return Outcome(booking=self._store(updated))

    def requote(self, reference: str, at: datetime) -> Quote:
        """Issue a fresh quote for a REQUOTE_REQUIRED hold and re-pin it to the current version."""
        with self._lock:
            booking = self._get(reference)
            if booking.state is not BookingState.REQUOTE_REQUIRED:
                raise ValueError(f"booking {reference} is not awaiting requote")
            snapshot = self._snapshot(booking.lane_id, booking.cargo, booking.facts, booking.departure_at, utc(at))
            quote = Quote(
                snapshot,
                booking.cargo,
                booking.departure_at,
                booking.facts,
                snapshot.rate_per_kg * booking.cargo.weight_kg,
                utc(at) + self.quote_ttl,
            )
            updated = replace(
                booking,
                snapshot=snapshot,
                price=quote.price,
                state=BookingState.HELD,
                history=booking.history + (BookingEvent(utc(at), "requoted", f"snapshot v{snapshot.version}"),),
            )
            self._store(updated)
            return quote

    def cancel(self, reference: str, at: datetime) -> Booking:
        with self._lock:
            booking = self._get(reference)
            if booking.state not in (BookingState.HELD, BookingState.REQUOTE_REQUIRED):
                raise ValueError(f"booking {reference} cannot cancel from {booking.state.value}")
            return self._transition(booking, BookingState.CANCELLED, at, "cancelled")

    def _transition(self, booking: Booking, state: BookingState, at: datetime, detail: str) -> Booking:
        updated = replace(
            booking,
            state=state,
            history=booking.history + (BookingEvent(utc(at), state.value, detail),),
        )
        return self._store(updated)

    def publish_version(self, version: QualificationVersion, at: datetime) -> SweepReport:
        """Publish a new qualification version and settle every open hold on the lane.

        Confirmed bookings keep their pinned snapshot untouched; drift is reported
        as advisories instead of being applied silently.
        """
        self.registry.publish(version)
        requote_required: list[str] = []
        rejected: list[str] = []
        advisories: list[Advisory] = []
        with self._lock:
            for booking in sorted(self._bookings.values(), key=lambda b: b.reference):
                if booking.lane_id != version.lane_id:
                    continue
                if booking.state in (BookingState.HELD, BookingState.REQUOTE_REQUIRED):
                    resolved = self._resolve_booking(booking, at)
                    if isinstance(resolved, Conflict):
                        if resolved.action is ConflictAction.REJECT:
                            self._transition(booking, BookingState.REJECTED, at, "; ".join(resolved.reasons))
                            rejected.append(booking.reference)
                        else:
                            self._transition(booking, BookingState.REQUOTE_REQUIRED, at, "; ".join(resolved.reasons))
                            requote_required.append(booking.reference)
                elif booking.state is BookingState.CONFIRMED:
                    resolved = self._resolve_booking(booking, at)
                    if isinstance(resolved, Conflict):
                        advisories.append(Advisory(booking.reference, resolved.reasons))
        return SweepReport(version.lane_id, version.version, tuple(requote_required), tuple(rejected), tuple(advisories))

    def qualification_of(self, reference: str) -> QualificationSnapshot:
        """The snapshot a booking is pinned to, regardless of later versions."""
        with self._lock:
            return self._get(reference).snapshot

    def qualification_as_of(
        self,
        lane_id: str,
        at: datetime,
        facts: Mapping[str, object] | None = None,
    ) -> QualificationSnapshot | None:
        """Time-travel query over the same snapshot type bookings are bound to."""
        version = self.registry.version_at(lane_id, at)
        if version is None:
            return None
        records = self._policy_records(_freeze_facts(facts), utc(at), lane_id) if facts is not None else ()
        return build_snapshot(version, records, utc(at))

    def booking(self, reference: str) -> Booking:
        with self._lock:
            return self._get(reference)
