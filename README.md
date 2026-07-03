# fedqrs — the federated-query execution engine (Rust)

`fedqrs` executes a federated query plan natively in Rust on DataFusion, and is
loaded into the Python `federated_query` engine as a PyO3 extension module.

Python parses, binds, optimizes, and physical-plans a query, then serializes the
plan to a small **IR** and hands it here once. `fedqrs` reads every source over a
native driver, runs the joins/aggregates/sorts on DataFusion, computes and
injects the semi-join filters, and streams the final Arrow result back to Python.
No intermediate data is ever revived into Python objects — the whole execution
and all IO stay in Rust.

## Workspace layout

```
fedqrs/                 the PyO3 extension (cdylib). FFI + native IO + interpreter.
  src/lib.rs            pymodule: register_datasource, execute_ir, fetch_*
  src/ffi.rs            Arrow C-stream import/export over PyCapsule (zero-copy)
  src/connectors.rs     native Postgres reads (ADBC), pooled connections,
                        parallel ctid scan, temp-table ingest, stats queries
  src/engine.rs         the IR interpreter: step loop, dynamic-filter strategy
                        selection, DataFusion fragment execution
core/                   fedqrs-core: the pyo3-FREE logic (unit-tested standalone)
  src/ir.rs             the serializable two-layer IR (serde)
  src/expr.rs           expression sub-IR -> DataFusion Expr
  src/sql.rs            source-SQL emission (scan / temp-join / filters)
  src/partition.rs      ctid range splitting + selectivity estimate
  src/types.rs          shared value types (DsKind)
```

The split keeps all pure logic in `fedqrs-core`, which has no Python dependency,
so it builds and unit-tests with a plain `cargo test`.

## The IR

Two layers, serialized as JSON:

- **Orchestration steps** — an ordered list. `source_scan`, `collect_distinct`,
  `injected_scan` (the probe, with the runtime dynamic filter), `merge` (run a
  fragment), `return`. This layer captures the one thing relational algebra
  cannot: the feedback edge where the probe is read only after the build side's
  distinct keys are known.
- **Relational fragments** — the local operators DataFusion runs: `hash_join`,
  `project`, `aggregate`, `sort`. Each fully specifies one operator.

Every step writes a named binding; a binding's Arrow columns are named by the
emitting node's output schema, so a parent operator resolves its expressions
against the child's aliases (the same contract the engine's own operators use).

## Dynamic-filter strategies (chosen in Rust at run time)

When a cross-source join reduces the probe by the build side's keys, the engine
picks the cheapest strategy by key cardinality and estimated selectivity
(`pg_class.reltuples` + `pg_stats.n_distinct`):

| distinct build keys | selectivity | strategy |
|--------------------:|-------------|----------|
| 0                   | —           | probe returns nothing (`WHERE false`) |
| < 2000              | any         | inline `col IN (v1, ...)` |
| >= 2000             | <= 40%      | ingest keys into a `TEMP TABLE` (ADBC bulk binary COPY) + server-side semi-join (uses the probe's index) |
| >= 2000             | > 40%       | parallel ctid full scan (pooled workers), reduce in DataFusion |

Parallel reads split the table's heap into `ctid` page ranges and read them
concurrently over a pooled set of worker connections — the same approach
DuckDB's postgres scanner uses.

## Build

```bash
cd fedqrs
maturin develop --release          # builds the extension into the active venv
```

Postgres `numeric` columns arrive over ADBC as opaque strings; the connector
casts them to `Float64` at the boundary so DataFusion arithmetic works.

## Test

```bash
# Rust unit tests (no database, no Python) — the core logic:
cargo test -p fedqrs-core

# Python integration / parity tests (need a live Postgres):
cd ../federated-query
POSTGRES_DB=<db> python -m pytest tests/test_rust_engine.py -q
```

The integration tests run real queries through the engine and assert the result
equals the existing DuckDB merge-engine path.

## Performance (TPC-H sf0.1, cross-source over Postgres)

Vs the old Python + in-memory-DuckDB merge engine: **1.9-2.7x faster**. Vs DuckDB
reading the same Postgres (postgres_scanner): **0.69-1.55x** — on par, ahead on
selective / decimal joins. A selective 10k-of-3M join on an indexed table is
**1.9x faster than DuckDB** (temp-table pushdown fetches only matching rows,
while DuckDB downloads the whole table). See `benchmarks/tpch/run_engine_perf.py`
in the `federated-query` repo.

## Status

Working: native Postgres, the full IR + interpreter, hash-join / project /
aggregate / sort fragments, the three dynamic-filter strategies, parallel scan,
decimal-on-read. See `PLAN.md` for the remaining roadmap (more operators, native
DuckDB / ClickHouse connectors, Decimal128, the default cutover, DuckDB removal).

The original Postgres->Arrow fetch micro-benchmark lives in `README-bench.md`
(`cargo run --release --features bench --bin fedqrs-bench`).
