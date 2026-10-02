from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from threading import RLock
from typing import Any, Iterable


GENESIS_DIGEST = "GENESIS"


def utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__datetime__": utc(value).isoformat()}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if hasattr(value, "__dataclass_fields__"):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (set, frozenset)):
        rows = [_canonical(v) for v in value]
        return sorted(rows, key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")))
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def digest(value: Any) -> str:
    raw = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class VersionConflict(RuntimeError):
    def __init__(self, expected: int | None, actual: int):
        super().__init__(f"expected lane version {expected}, actual {actual}")
        self.expected = expected
        self.actual = actual


class BookingRejected(ValueError):
    def __init__(self, reason_code: str, reasons: Iterable[str] = (), snapshot_digest: str | None = None):
        self.reason_code = reason_code
        self.reason_codes = (reason_code,)
        self.reasons = tuple(reasons)
        self.snapshot_digest = snapshot_digest
        super().__init__(reason_code + (": " + "; ".join(self.reasons) if self.reasons else ""))


class SnapshotConflict(RuntimeError):
    def __init__(self, expected_digest: str, current_digest: str, current_version: int):
        super().__init__(f"snapshot changed from {expected_digest[:12]} to {current_digest[:12]}")
        self.expected_digest = expected_digest
        self.current_digest = current_digest
        self.current_version = current_version


class QuoteError(RuntimeError):
    pass


class QuoteExpired(QuoteError):
    def __init__(self, quote_id: str):
        super().__init__(f"quote {quote_id} has expired")
        self.quote_id = quote_id


class QuoteSuperseded(QuoteError):
    def __init__(self, quote_id: str, successor_id: str | None):
        super().__init__(f"quote {quote_id} was superseded by {successor_id}")
        self.quote_id = quote_id
        self.successor_id = successor_id


@dataclass(frozen=True)
class QualificationEvidence:
    evidence_id: str
    evidence_type: str
    valid_from: datetime
    expires_at: datetime
    carrier_id: str | None = None
    reference: str = ""
    revoked: bool = False

    def covers(self, at: datetime, valid_through: datetime | None = None) -> bool:
        start = utc(self.valid_from)
        end = utc(self.expires_at)
        moment = utc(at)
        through = utc(valid_through or at)
        return not self.revoked and start <= moment and through < end

    def failure_code(self, at: datetime, valid_through: datetime | None = None) -> str | None:
        moment = utc(at)
        through = utc(valid_through or at)
        if self.revoked:
            return "revoked_evidence"
        if moment < utc(self.valid_from):
            return "evidence_not_yet_valid"
        if through >= utc(self.expires_at):
            return "expired_evidence"
        return None


@dataclass(frozen=True)
class LaneQualification:
    lane_id: str
    origin: str
    destination: str
    product_classes: frozenset[str] = frozenset({"*"})
    status: str = "qualified"

    def __post_init__(self) -> None:
        if not self.lane_id or not self.origin or not self.destination:
            raise ValueError("lane id, origin and destination are required")
        if not self.product_classes:
            raise ValueError("at least one product class is required")
        if self.status not in {"qualified", "suspended"}:
            raise ValueError("unsupported lane qualification status")


@dataclass(frozen=True)
class CarrierStrategy:
    carrier_id: str
    base_price: Decimal = Decimal("0")
    price_per_kg: Decimal = Decimal("0")
    currency: str = "USD"
    priority: int = 0
    max_payload_kg: float | None = None
    product_classes: frozenset[str] = frozenset({"*"})
    allowed: bool = True

    def supports(self, product_class: str) -> bool:
        return "*" in self.product_classes or product_class in self.product_classes

    def has_capacity(self, payload_kg: float) -> bool:
        return self.max_payload_kg is None or payload_kg <= self.max_payload_kg

    def price_for(self, payload_kg: float) -> Decimal:
        return self.base_price + self.price_per_kg * Decimal(str(payload_kg))


@dataclass(frozen=True)
class LaneSnapshot:
    lane_id: str
    version: int
    effective_at: datetime
    created_at: datetime
    qualification: LaneQualification
    carrier_strategies: tuple[CarrierStrategy, ...]
    evidence: tuple[QualificationEvidence, ...]
    published_by: str = "system"
    reason: str = ""
    previous_digest: str = GENESIS_DIGEST
    digest: str = ""

    @property
    def snapshot_id(self) -> str:
        return self.digest

    def strategy(self, carrier_id: str) -> CarrierStrategy | None:
        rows = [s for s in self.carrier_strategies if s.carrier_id == carrier_id]
        return rows[0] if rows else None

    def evidence_for(self, carrier_id: str | None) -> tuple[QualificationEvidence, ...]:
        return tuple(e for e in self.evidence if e.carrier_id is None or e.carrier_id == carrier_id)


class LaneQualificationRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._versions: dict[str, list[LaneSnapshot]] = {}
        self._by_digest: dict[str, LaneSnapshot] = {}

    def publish(
        self,
        qualification: LaneQualification,
        effective_at: datetime,
        carrier_strategies: Iterable[CarrierStrategy],
        evidence: Iterable[QualificationEvidence] = (),
        *,
        created_at: datetime | None = None,
        published_by: str = "system",
        reason: str = "",
        expected_version: int | None = None,
    ) -> LaneSnapshot:
        effective = utc(effective_at)
        created = utc(created_at or datetime.now(timezone.utc))
        strategies = tuple(carrier_strategies)
        evidence_rows = tuple(evidence)
        self._validate_publish(qualification, effective, strategies, evidence_rows)

        with self._lock:
            rows = self._versions.setdefault(qualification.lane_id, [])
            actual_version = rows[-1].version if rows else 0
            if expected_version is not None and expected_version != actual_version:
                raise VersionConflict(expected_version, actual_version)
            if any(v.effective_at == effective for v in rows):
                raise ValueError("another lane qualification version has the same effective time")

            version = actual_version + 1
            previous_digest = rows[-1].digest if rows else GENESIS_DIGEST
            content = [
                "lane-qualification-v1",
                qualification,
                version,
                effective,
                created,
                strategies,
                evidence_rows,
                published_by,
                reason,
                previous_digest,
            ]
            snapshot = LaneSnapshot(
                lane_id=qualification.lane_id,
                version=version,
                effective_at=effective,
                created_at=created,
                qualification=qualification,
                carrier_strategies=strategies,
                evidence=evidence_rows,
                published_by=published_by,
                reason=reason,
                previous_digest=previous_digest,
                digest=digest(content),
            )
            rows.append(snapshot)
            self._by_digest[snapshot.digest] = snapshot
            return snapshot

    @staticmethod
    def _validate_publish(
        qualification: LaneQualification,
        effective_at: datetime,
        strategies: tuple[CarrierStrategy, ...],
        evidence_rows: tuple[QualificationEvidence, ...],
    ) -> None:
        if qualification.status == "qualified" and not any(s.allowed for s in strategies):
            raise ValueError("qualified lane requires at least one allowed carrier strategy")
        carrier_ids = [s.carrier_id for s in strategies]
        if len(carrier_ids) != len(set(carrier_ids)):
            raise ValueError("duplicate carrier strategy")
        evidence_ids = [e.evidence_id for e in evidence_rows]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence")
        for item in evidence_rows:
            if utc(item.expires_at) <= utc(item.valid_from):
                raise ValueError(f"evidence {item.evidence_id} expires before it becomes valid")
            if not item.covers(effective_at):
                raise ValueError(f"evidence {item.evidence_id} is not valid at effective time")

    def snapshot_at(self, lane_id: str, at: datetime) -> LaneSnapshot:
        moment = utc(at)
        with self._lock:
            rows = [v for v in self._versions.get(lane_id, ()) if v.effective_at <= moment]
        if not rows:
            raise LookupError(f"no effective lane qualification for {lane_id}")
        return max(rows, key=lambda v: (v.effective_at, v.version))

    def version(self, lane_id: str, version: int) -> LaneSnapshot:
        with self._lock:
            rows = self._versions.get(lane_id, ())
            for snapshot in rows:
                if snapshot.version == version:
                    return snapshot
        raise LookupError(f"unknown lane qualification version {lane_id}:{version}")

    def snapshot(self, snapshot_digest: str) -> LaneSnapshot:
        with self._lock:
            try:
                return self._by_digest[snapshot_digest]
            except KeyError:
                raise LookupError(f"unknown lane snapshot {snapshot_digest}") from None

    def history(self, lane_id: str) -> tuple[LaneSnapshot, ...]:
        with self._lock:
            return tuple(sorted(self._versions.get(lane_id, ()), key=lambda v: (v.effective_at, v.version)))

    def lane_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._versions))


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    snapshot_digest: str
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    carrier_id: str | None = None

    def __bool__(self) -> bool:
        return self.eligible

    def require(self) -> None:
        if not self:
            raise BookingRejected(self.reason_codes[0], self.reasons, self.snapshot_digest)


def _evidence_failures(
    evidence_rows: Iterable[QualificationEvidence], at: datetime, valid_through: datetime | None
) -> list[str]:
    failures: list[str] = []
    for item in evidence_rows:
        code = item.failure_code(at, valid_through)
        if code:
            failures.append(f"{code}:{item.evidence_id}")
    return failures


def evaluate_snapshot(
    snapshot: LaneSnapshot,
    at: datetime,
    product_class: str,
    payload_kg: float | None = None,
    carrier_id: str | None = None,
    valid_through: datetime | None = None,
) -> EligibilityResult:
    moment = utc(at)
    through = utc(valid_through or at)
    if through < moment:
        raise ValueError("valid_through cannot be before evaluation time")

    reasons: list[str] = []
    if moment < utc(snapshot.effective_at):
        reasons.append("snapshot_not_yet_effective")
    if snapshot.qualification.status != "qualified":
        reasons.append(f"lane_{snapshot.qualification.status}")
    if product_class not in snapshot.qualification.product_classes and "*" not in snapshot.qualification.product_classes:
        reasons.append(f"product_not_supported:{product_class}")

    lane_evidence = snapshot.evidence_for(None)
    for failure in _evidence_failures(lane_evidence, moment, through):
        reasons.append(failure)

    if carrier_id:
        strategy = snapshot.strategy(carrier_id)
        carrier_evidence = tuple(e for e in snapshot.evidence if e.carrier_id == carrier_id)
        for failure in _evidence_failures(carrier_evidence, moment, through):
            reasons.append(failure)
        if strategy is None:
            reasons.append(f"carrier_not_found:{carrier_id}")
        else:
            if not strategy.supports(product_class):
                reasons.append(f"carrier_product_not_supported:{carrier_id}:{product_class}")
            if payload_kg is not None and payload_kg <= 0:
                reasons.append("payload_must_be_positive")
            if payload_kg is not None and not strategy.has_capacity(payload_kg):
                reasons.append(f"payload_exceeds_capacity:{carrier_id}")
            if not strategy.allowed:
                reasons.append(f"carrier_not_allowed:{carrier_id}")

    codes = tuple(reason.split(":", 1)[0] for reason in reasons)
    return EligibilityResult(not reasons, snapshot.digest, codes, tuple(reasons), carrier_id)


def carrier_candidates(
    snapshot: LaneSnapshot,
    product_class: str,
    payload_kg: float,
    at: datetime,
    valid_through: datetime | None = None,
) -> tuple[CarrierStrategy, ...]:
    if payload_kg <= 0:
        raise ValueError("payload must be positive")
    lane = evaluate_snapshot(snapshot, at, product_class, payload_kg, None, valid_through)
    if not lane:
        return ()
    candidates = []
    for strategy in snapshot.carrier_strategies:
        if not strategy.allowed or not strategy.supports(product_class) or not strategy.has_capacity(payload_kg):
            continue
        result = evaluate_snapshot(snapshot, at, product_class, payload_kg, strategy.carrier_id, valid_through)
        if result:
            candidates.append(strategy)
    return tuple(sorted(candidates, key=lambda s: (-s.priority, s.price_for(payload_kg), s.carrier_id)))


@dataclass(frozen=True)
class QuoteRequest:
    request_id: str
    lane_id: str
    product_class: str
    payload_kg: float
    required_valid_through: datetime | None = None
    preferred_carrier_id: str | None = None


@dataclass(frozen=True)
class Quote:
    quote_id: str
    request_id: str
    lane_id: str
    snapshot_digest: str
    snapshot_version: int
    carrier_strategy: CarrierStrategy
    price: Decimal
    currency: str
    created_at: datetime
    expires_at: datetime
    required_valid_through: datetime | None = None
    product_class: str = ""
    payload_kg: float = 0.0
    revision_of: str | None = None
    status: str = "quoted"
    superseded_by: str | None = None
    booking_id: str | None = None


@dataclass(frozen=True)
class LaneBooking:
    booking_id: str
    request_id: str
    quote_id: str
    lane_id: str
    snapshot_digest: str
    snapshot_version: int
    carrier_id: str
    carrier_strategy: CarrierStrategy
    price: Decimal
    currency: str
    product_class: str
    payload_kg: float
    required_valid_through: datetime | None
    booked_at: datetime
    status: str = "booked"


@dataclass(frozen=True)
class BookingEvent:
    at: datetime
    event_type: str
    quote_id: str | None
    snapshot_digest: str | None
    details: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BookingRecord:
    booking: LaneBooking
    quote: Quote
    snapshot: LaneSnapshot
    events: tuple[BookingEvent, ...]


@dataclass(frozen=True)
class CurrentBookingStatus:
    booking_id: str
    booked_snapshot_digest: str
    current_snapshot_digest: str
    current_version: int
    changed: bool
    current_eligibility: EligibilityResult
    action: str


class LaneBookingService:
    def __init__(self, registry: LaneQualificationRegistry | None = None, quote_ttl: timedelta = timedelta(hours=24)):
        self.registry = registry or LaneQualificationRegistry()
        self.quote_ttl = quote_ttl
        self._lock = RLock()
        self._quotes: dict[str, Quote] = {}
        self._bookings: dict[str, LaneBooking] = {}
        self._events: dict[str, list[BookingEvent]] = {}
        self._quote_serial = 0
        self._booking_serial = 0

    def quote(self, request: QuoteRequest, at: datetime | None = None) -> Quote:
        moment = utc(at or datetime.now(timezone.utc))
        self._validate_request(request, moment)
        with self._lock:
            try:
                quote = self._quote(request, moment)
            except BookingRejected as exc:
                self._event(request.request_id, moment, "quote_rejected", None, {"reason": exc.reason_code})
                raise
            self._event(request.request_id, moment, "quoted", quote.snapshot_digest, {"quote_id": quote.quote_id})
            return quote

    def _quote(self, request: QuoteRequest, at: datetime, revision_of: str | None = None) -> Quote:
        snapshot = self.registry.snapshot_at(request.lane_id, at)
        if request.preferred_carrier_id is not None:
            result = evaluate_snapshot(
                snapshot,
                at,
                request.product_class,
                request.payload_kg,
                request.preferred_carrier_id,
                request.required_valid_through,
            )
            result.require()
            strategy = snapshot.strategy(request.preferred_carrier_id)
            assert strategy is not None
        else:
            lane = evaluate_snapshot(
                snapshot, at, request.product_class, request.payload_kg, None, request.required_valid_through
            )
            lane.require()
            candidates = carrier_candidates(
                snapshot,
                request.product_class,
                request.payload_kg,
                at,
                request.required_valid_through,
            )
            if not candidates:
                raise BookingRejected("no_eligible_carrier", (), snapshot.digest)
            strategy = candidates[0]

        self._quote_serial += 1
        created = utc(at)
        price = strategy.price_for(request.payload_kg)
        content = [
            "lane-quote-v1",
            request,
            snapshot.digest,
            strategy,
            str(price),
            created.isoformat(),
            self._quote_serial,
            revision_of,
        ]
        quote_id = "Q-" + digest(content)[:20]
        quote = Quote(
            quote_id=quote_id,
            request_id=request.request_id,
            lane_id=request.lane_id,
            snapshot_digest=snapshot.digest,
            snapshot_version=snapshot.version,
            carrier_strategy=strategy,
            price=price,
            currency=strategy.currency,
            created_at=created,
            expires_at=created + self.quote_ttl,
            required_valid_through=request.required_valid_through,
            product_class=request.product_class,
            payload_kg=request.payload_kg,
            revision_of=revision_of,
        )
        self._quotes[quote_id] = quote
        return quote

    def book(self, quote_id: str, at: datetime | None = None, on_conflict: str = "reject") -> LaneBooking | Quote:
        if on_conflict not in {"reject", "requote"}:
            raise ValueError("on_conflict must be 'reject' or 'requote'")
        moment = utc(at or datetime.now(timezone.utc))
        with self._lock:
            quote = self._latest_quote(quote_id)
            if quote.status == "booked" and quote.booking_id:
                return self._bookings[quote.booking_id]

            if quote.status == "superseded":
                if on_conflict == "reject":
                    raise QuoteSuperseded(quote.quote_id, quote.superseded_by)
                successor = self._latest_quote(quote.quote_id)
                if successor.status == "quoted":
                    return successor

            if moment >= utc(quote.expires_at):
                if on_conflict == "reject":
                    raise QuoteExpired(quote.quote_id)
                return self._requote(quote, moment, "quote_expired")

            current = self.registry.snapshot_at(quote.lane_id, moment)
            if current.digest != quote.snapshot_digest:
                if on_conflict == "reject":
                    self._event(quote.request_id, moment, "booking_rejected", current.digest, {"reason": "snapshot_conflict"})
                    raise SnapshotConflict(quote.snapshot_digest, current.digest, current.version)
                return self._requote(quote, moment, "snapshot_changed")

            result = evaluate_snapshot(
                current,
                moment,
                quote.product_class,
                quote.payload_kg,
                quote.carrier_strategy.carrier_id,
                quote.required_valid_through,
            )
            if not result:
                self._event(quote.request_id, moment, "booking_rejected", current.digest, {"reason": result.reason_codes[0]})
                raise BookingRejected(result.reason_codes[0], result.reasons, current.digest)

            self._booking_serial += 1
            booking_id = "B-" + digest(["lane-booking-v1", quote.quote_id, moment.isoformat(), self._booking_serial])[:20]
            booking = LaneBooking(
                booking_id=booking_id,
                request_id=quote.request_id,
                quote_id=quote.quote_id,
                lane_id=quote.lane_id,
                snapshot_digest=quote.snapshot_digest,
                snapshot_version=quote.snapshot_version,
                carrier_id=quote.carrier_strategy.carrier_id,
                carrier_strategy=quote.carrier_strategy,
                price=quote.price,
                currency=quote.currency,
                product_class=quote.product_class,
                payload_kg=quote.payload_kg,
                required_valid_through=quote.required_valid_through,
                booked_at=moment,
            )
            self._bookings[booking_id] = booking
            self._quotes[quote.quote_id] = replace(quote, status="booked", booking_id=booking_id)
            self._event(quote.request_id, moment, "booked", booking.snapshot_digest, {"booking_id": booking_id})
            return booking

    def _requote(self, old: Quote, at: datetime, reason: str) -> Quote:
        required_through = old.required_valid_through
        if required_through is not None and utc(required_through) < at:
            required_through = None
        request = QuoteRequest(
            request_id=old.request_id,
            lane_id=old.lane_id,
            product_class=old.product_class,
            payload_kg=old.payload_kg,
            required_valid_through=required_through,
            preferred_carrier_id=None,
        )
        try:
            new = self._quote(request, at, revision_of=old.quote_id)
        except BookingRejected as exc:
            self._event(old.request_id, at, "booking_rejected", exc.snapshot_digest, {"reason": exc.reason_code})
            raise
        self._quotes[old.quote_id] = replace(old, status="superseded", superseded_by=new.quote_id)
        self._event(old.request_id, at, "requoted", new.snapshot_digest, {
            "old_quote_id": old.quote_id,
            "new_quote_id": new.quote_id,
            "reason": reason,
        })
        return new

    def _latest_quote(self, quote_id: str) -> Quote:
        quote = self._quotes[quote_id]
        seen = {quote_id}
        while quote.superseded_by:
            quote_id = quote.superseded_by
            if quote_id in seen:
                raise RuntimeError("quote revision cycle")
            seen.add(quote_id)
            quote = self._quotes[quote_id]
        return quote

    def snapshot_for_booking(self, booking_id: str) -> LaneSnapshot:
        with self._lock:
            booking = self._bookings[booking_id]
        return self.registry.snapshot(booking.snapshot_digest)

    def booking_history(self, booking_id: str) -> BookingRecord:
        with self._lock:
            booking = self._bookings[booking_id]
            quote = self._quotes[booking.quote_id]
            events = tuple(self._events.get(booking.request_id, ()))
        snapshot = self.registry.snapshot(booking.snapshot_digest)
        return BookingRecord(booking, quote, snapshot, events)

    def current_booking_status(self, booking_id: str, at: datetime | None = None) -> CurrentBookingStatus:
        moment = utc(at or datetime.now(timezone.utc))
        with self._lock:
            booking = self._bookings[booking_id]
        current = self.registry.snapshot_at(booking.lane_id, moment)
        required_through = max(moment, utc(booking.required_valid_through)) if booking.required_valid_through else None
        eligibility = evaluate_snapshot(
            current,
            moment,
            booking.product_class,
            booking.payload_kg,
            booking.carrier_id,
            required_through,
        )
        changed = current.digest != booking.snapshot_digest
        action = "keep"
        return CurrentBookingStatus(
            booking_id,
            booking.snapshot_digest,
            current.digest,
            current.version,
            changed,
            eligibility,
            action,
        )

    @staticmethod
    def _validate_request(request: QuoteRequest, at: datetime) -> None:
        if not request.request_id or not request.lane_id:
            raise ValueError("request id and lane id are required")
        if request.payload_kg <= 0:
            raise ValueError("payload must be positive")
        if request.required_valid_through is not None and utc(request.required_valid_through) < utc(at):
            raise ValueError("required_valid_through cannot be before quote time")

    def _event(
        self,
        request_id: str,
        at: datetime,
        event_type: str,
        snapshot_digest: str | None,
        details: dict[str, str],
    ) -> None:
        event = BookingEvent(utc(at), event_type, details.get("quote_id"), snapshot_digest, dict(details))
        self._events.setdefault(request_id, []).append(event)
