# Seed data

Reference data loaded by `make seed`. JSON rather than Python literals so the
catalogue can be edited during a demo without touching code.

## `inventory.json`

Extends the four starter items with `unit_price` and `category`. Prices are
**strings** — they land in a `TEXT` column and parse to `Decimal`. A SQLite
`REAL` column would silently make money a float.

`FakeItem` has no price and zero stock. It is the fraud probe (INV-1003).

## `vendors.json`

Fifteen vendors. Twelve appear on the sample invoices; three are filler so the
table reads like a real ledger rather than a fixture.

Two vendors are **deliberately absent**, so `UNKNOWN_VENDOR` fires on exactly
two invoices instead of becoming background noise:

| Missing vendor    | Invoice  |
|-------------------|----------|
| Fraudster LLC     | INV-1003 |
| NoProd Industries | INV-1008 |

`QuickShip Distributers` (INV-1012) is seeded `watchlist` with a recent
`first_seen` and an invoice count of 2. The invoice itself says "formerly
FastShip Ltd." — a vendor that recently renamed and has almost no history is a
real accounts-payable fraud signal, and it gives the critic something genuine to
weigh rather than a fixture that always concurs.
