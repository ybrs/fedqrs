# fedqrs: Rust/DataFusion execution engine - plan

## Goal

Replace the Python + in-memory-DuckDB "merge engine" with a Rust engine
(DataFusion + native source drivers, exposed to Python via PyO3). Python parses,
binds, optimizes, and physical-plans a query, then serializes the plan to an IR
and hands it to Rust ONCE. Rust executes the whole thing - reads every source
over a native driver, runs joins/aggregates/etc. locally, and returns the final
result to Python ONCE as an Arrow stream. No intermediate data is ever revived
into Python objects; reviving Arrow bytes into Python is the cost we are
removing.

There is no compatibility flag and no parallel path. The DuckDB merge engine is
removed once the Rust engine reaches parity (removal is the final step, flagged
for approval).

## Locked decisions

- Scope: everything (single-source pushdown AND cross-source merge) runs through
  the Rust engine. Single-source is the degenerate case: one source fetch, no
  local operators.
- Source reads: NATIVE Rust drivers (not a Python callback). Postgres + DuckDB
  first; ClickHouse next. Datasources are duplicated in Rust now; long term all
  datasource logic moves to Rust. Python may later expose a few functions to
  Rust for planning-time statistics (out of near-term scope).
- Source SQL: emitted in RUST. Expression rendering reuses DataFusion's unparser
  (`expr_to_sql`) so we get dialect-correct SQL without porting the whole Python
  emitter; Rust builds the SELECT skeleton. Full single-source-subtree SQL
  (arbitrary joins/aggregates in one source) uses DataFusion `plan_to_sql`
  later.
- Local operators: DataFusion as a mechanical operator library (DataFrame API),
  NOT as a re-optimizing planner. The IR fully specifies each operator.
- IR: a serializable ordered step list (orchestration) plus relational
  fragments, carrying a general expression sub-IR. Not Substrait.
- Dynamic filter (semi-join reduction): computed and applied entirely in Rust.
  Rust reads the build side, computes the distinct keys, and injects them into
  the probe source SQL it emits (`col IN (...)` or a native array parameter).
  The keys never cross into Python. This is the round-trip we are deleting.
- Connections: Python calls `register_datasource(name, kind, params)` once at
  session init; Rust pools and reuses connections; the IR references sources by
  name.
- Result crosses to Python once, as an Arrow C-stream.

## Data flow

```
Python                              Rust (fedqrs)
------                              -------------
parse/bind/optimize/plan
  -> PhysicalPlanNode tree
  -> serialize to IR (JSON)  ---->  parse IR
                                    for each step, in order:
                                      source_scan  -> emit dialect SQL,
                                                       fetch over native driver,
                                                       hold Arrow in Rust
                                      collect_keys -> DataFusion distinct (in Rust)
                                      injected_scan-> emit SQL with IN(keys),
                                                       fetch (keys stay in Rust)
                                      merge        -> DataFusion operator
                                      return       -> export result stream
  pa.Table  <----------------------  Arrow C-stream (result only)
```

## What already exists (Phase 0, done and tested)

- `src/ffi.rs` - Arrow C-stream import/export over PyCapsule (hand-rolled, no
  arrow-pyarrow dependency). Result export path is final; the import/callback
  path is superseded by native reads.
- `src/ir.rs` - serde IR types (to be extended for structured scans + expressions).
- `src/engine.rs` - step interpreter + DataFusion inner-join fragment +
  distinct-key computation. The "export keys to Python" path is being replaced
  by native probe emission.
- `src/lib.rs` - `execute_ir(ir_json, reader)` pymodule. `reader` callback to be
  dropped in favor of the datasource registry.
- Proven end to end: single-source passthrough and a cross-source inner join
  with semi-join reduction, on synthetic IR.

## Phases

- Phase 0 - DONE. FFI, IR interpreter, DataFusion inner join, semi-join feedback.
- Phase 1 - DONE. Native Postgres (ADBC) reads + registry, proven vs duckpoc.
- Phase 2 - DONE. Expression sub-IR -> DataFusion Expr; Rust source SQL emission
  (SELECT skeleton + unparser for filters).
- Phase 3 - DONE. Simple cross-source join fully in Rust: native reads, dynamic
  filter computed and injected in Rust (keys never leave Rust), over-cap
  fallback. Validated vs a direct Postgres join.
- Phase 4 - DONE (core). Python serializer (federated_query/executor/rust_ir.py):
  expr_to_ir, raw + structured scan specs, INNER-join walker with
  in_left/in_right column resolution, datasource bridge, execute_via_rust.
  Proven: the REAL planner's cross-source join and single-source query both
  match the DuckDB path. Making Rust the DEFAULT Executor path is gated on
  operator parity (Phase 5).
- Phase 5 - IN PROGRESS.
  - DONE: recursive serializer foundation (_emit(node)->binding; a binding's
    columns are named by the node's output schema; parents resolve via
    child.column_aliases()). Join emits canonical output columns; projection is
    its own fragment.
  - DONE: aggregate fragment (cross-source GROUP BY, built as DataFusion SQL so
    every agg function works; count(*) handled).
  - DONE: sort fragment (ORDER BY with direction + NULL placement).
  - DONE: thread-local Postgres connection pool (reuse driver+connection).
  - DONE: decimal-on-read - PG numeric arrives over ADBC as opaque strings;
    cast to Float64 at the fetch boundary so decimal arithmetic/SUM runs
    (matches DuckDB to float precision).
  - All parity-tested vs the DuckDB path in tests/test_rust_engine.py (7 tests).
  - Adding an operator is now a fixed 4-step template: ir.rs Fragment variant ->
    engine.rs run_* -> rust_ir.py _emit_* -> a parity test.
  - PERF (benchmarks/tpch/run_engine_perf.py, sf0.1): single-source ~1x (both
    push to PG); cross-source join/agg/sort 1.4-2.6x faster in Rust (largest
    when per-operator overhead dominates, shrinking as I/O dominates).
  - TODO: union/set-ops, outer/semi/anti joins, multi-key joins, window,
    cross-source lateral; Decimal128 (exact) instead of Float64; native DuckDB +
    ClickHouse connectors; concurrency benchmark; then make Rust the default
    Executor path and remove the DuckDB merge engine (approval-gated).

### Original phase notes

- Phase 1 - Native source layer. `register_datasource`; Postgres (ADBC) and
  DuckDB native connectors; a pooled connection registry; `fetch(name, sql) ->
  Arrow`. Smoke-test against the real POSTGRES_DB=duckpoc and a DuckDB file.
- Phase 2 - Expression IR + Rust source SQL. Expression sub-IR -> DataFusion
  `Expr`; single-table scan emission (columns, filter, limit, distinct) via the
  SELECT skeleton + `expr_to_sql`. This one translation serves both source SQL
  and local fragments.
- Phase 3 - Simple cross-source join, fully in Rust. Two single-table scans +
  inner equi-join + semi-join reduction (keys computed and injected in Rust) +
  optional aggregation. Nothing but the result crosses to Python. Validate
  against the current DuckDB path for the same SQL.
- Phase 4 - Python serializer (PhysicalPlanNode -> IR) for the simple-join
  class; route those queries through Rust in the Executor.
- Phase 5+ - Operator parity: outer/semi/anti joins, aggregate variants, sort,
  union/set-ops, window, cross-source lateral; full single-source-subtree
  emission via `plan_to_sql`; ClickHouse connector. Then remove the DuckDB merge
  engine (approval-gated).

## Correctness invariants (carried across the FFI boundary)

- Invalid/unsupported IR node -> raise (propagates as a Python exception). Never
  a silent drop or default.
- Dynamic-filter cap (default 2000 distinct keys): over cap -> no pushdown, full
  probe scan. Logged.
- NULL-aware semi/anti semantics preserved; NULL keys excluded from the IN list.
- Scalar-subquery cardinality guard (SingleRowGuard) still raises.
- Result schema/column names match what the Python after-processors expect.
