# ColdLedger Pharmaceutical Cold-Chain Operations Lab

Dependency-free Python reference implementation for temperature-sensitive pharmaceutical distribution. It models inventory expiry, chain-of-custody, telemetry correction, packaging qualification, dry-ice depletion, freezer allocation, customs holds, serialization, recalls, release decisions, vehicle compartment compatibility, returns, SLA calendars, cross-dock state, and effective-dated lane qualification snapshots.

Lane qualifications are published as immutable snapshots with evidence validity windows and carrier strategies. New quotes resolve the currently effective snapshot and reject expired evidence; bookings pin the snapshot digest, carrier strategy, price and audit events so later versions cannot alter them. Stale unbooked quotes reject by default; callers may explicitly request a requote, which returns a new quote rather than silently booking it.

Runtime third-party dependencies: none.
