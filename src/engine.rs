//! The IR interpreter: walks the orchestration steps, reads sources natively,
//! runs relational fragments on DataFusion, and returns the result as an
//! exportable Arrow stream.
//!
//! Everything stays in Rust. Source reads go over native drivers; the semi-join
//! reduction computes the build's distinct keys and injects them into the probe
//! SQL here, so key values are never handed to Python. Only the final result
//! crosses the boundary, once.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};

use arrow::array::RecordBatch;
use arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::{Column, DataFusionError, JoinType, TableReference};
use datafusion::datasource::MemTable;
use datafusion::execution::memory_pool::{FairSpillPool, MemoryConsumer};
use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use futures::StreamExt;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::Expr;
use datafusion::prelude::{col, lit, SessionConfig, SessionContext};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use crate::connectors;
use fedqrs_core::types::DsKind;
use fedqrs_core::expr::{literal_from_array, to_df_expr};
use fedqrs_core::ir::{AggCall, AggSelectItem, Fragment, Ir, JoinKind, Projection, ScanSpec, Step};
use fedqrs_core::sql::{base_filter_sql, render_expr, scan_sql, select_list_sql, temp_join_sql};

/// Dynamic-filter strategy thresholds. Under `IN_CAP` distinct keys we inline an
/// IN list; above it we push a temp table unless the filter would select more
/// than the source's full-scan fraction, in which case a full scan of the
/// (bandwidth-bound) table wins. Partition count is tuned near the core count.
const IN_CAP: usize = 2000;
/// DuckDB's full-scan alternative is one single-stream read.
const FULL_SCAN_FRACTION: f64 = 0.40;
/// Postgres' full-scan alternative is the 8-way ctid-parallel read, whose
/// wire+decode runs ~3.6x a single stream, while the temp-table semi-join is
/// pinned to ONE connection (temp tables are connection-scoped). Measured on
/// the SF1 customer probe: 25% selectivity temp-join 75.6ms vs parallel full
/// 43ms - the break-even sits near 15%.
const PG_FULL_SCAN_FRACTION: f64 = 0.15;
/// DuckDB key-ingest ceiling: the point past which appending the keys into the
/// temp table and probing costs more than reading the probe whole, REGARDLESS
/// of selectivity. It is no longer a selectivity proxy - the near-superset case
/// that a small cap once guarded (150k keys covering most of the column - the
/// q18 regression) is now caught by `fetches_most_of_table`, which prices the
/// real keys/NDV from the planner's statistics. So this only bounds the ingest
/// work; a selective 100k-key set (q09: 108k green-part keys, 5.4% of lineitem)
/// now reduces instead of reading 60M rows whole.
const DUCK_TEMP_CAP: usize = 2_000_000;
const PARALLEL_PARTITIONS: usize = 8;
const DYN_KEYS_TEMP_TABLE: &str = "fedq_dyn_keys";
/// DataFusion memory cap. Every context draws from ONE shared pool, so the
/// engine as a whole is bounded; a fragment that would blow past it (a CROSS
/// join's cartesian product) fails with ResourcesExhausted instead of OOMing
/// the server. Sorts and grouped aggregates spill to disk via the default
/// DiskManager; hash and nested-loop joins do not spill and error instead.
const MEMORY_LIMIT_BYTES: usize = 32 * 1024 * 1024 * 1024;

/// The shared memory-capped RuntimeEnv (FairSpillPool + default DiskManager)
/// behind every DataFusion context the engine creates.
pub(crate) fn runtime_env() -> Arc<RuntimeEnv> {
    static ENV: OnceLock<Arc<RuntimeEnv>> = OnceLock::new();
    ENV.get_or_init(|| {
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(FairSpillPool::new(MEMORY_LIMIT_BYTES)))
            .build_arc()
            .expect("DataFusion runtime environment construction failed")
    })
    .clone()
}

/// A SessionContext wired to the shared memory-capped runtime.
fn memory_capped_context() -> SessionContext {
    SessionContext::new_with_config_rt(SessionConfig::new(), runtime_env())
}

/// Fully-read Arrow data with its schema.
struct Batches {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

/// A named intermediate produced by a step.
enum Binding {
    /// A relation held in memory (source result or fragment output).
    Materialized(Batches),
    /// The build side's distinct, NULL-free join-key values.
    Keys(Batches),
}

fn df_to_py(e: DataFusionError) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

/// Whether per-step timing lines go to stderr (FEDQRS_PROFILE=1). Read once.
fn profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FEDQRS_PROFILE").map_or(false, |v| v != "0" && !v.is_empty()))
}

/// Total rows currently held by a binding (either variant), for profiling.
fn binding_rows(bindings: &HashMap<String, Binding>, name: &str) -> usize {
    match bindings.get(name) {
        Some(Binding::Materialized(b)) | Some(Binding::Keys(b)) => {
            b.batches.iter().map(|x| x.num_rows()).sum()
        }
        None => 0,
    }
}

/// The fragment's operator name, for profiling labels.
fn fragment_kind(fragment: &Fragment) -> &'static str {
    match fragment {
        Fragment::HashJoin { .. } => "hash_join",
        Fragment::NestedLoopJoin { .. } => "nested_loop_join",
        Fragment::Project { .. } => "project",
        Fragment::Aggregate { .. } => "aggregate",
        Fragment::Sort { .. } => "sort",
        Fragment::Filter { .. } => "filter",
        Fragment::Limit { .. } => "limit",
        Fragment::RawSql { .. } => "raw_sql",
    }
}

/// One stderr line describing a finished step: time, output rows, what ran.
fn log_step(step: &Step, ir: &Ir, bindings: &HashMap<String, Binding>, elapsed_ms: f64) {
    let line = match step {
        Step::SourceScan { datasource, scan, binding, .. } => {
            let what = scan.raw_sql.as_deref().unwrap_or(scan.table.as_deref().unwrap_or("?"));
            let short: String = what.chars().take(100).collect();
            format!("source_scan   ds={datasource:<6} rows={:<9} {short}", binding_rows(bindings, binding))
        }
        Step::CollectDistinct { key, binding, .. } => {
            format!("collect_dist  key={key:<20} rows={}", binding_rows(bindings, binding))
        }
        Step::InjectedScan { datasource, scan, inject_column, binding, .. } => {
            let what = scan.raw_sql.as_deref().unwrap_or(scan.table.as_deref().unwrap_or("?"));
            let short: String = what.chars().take(80).collect();
            format!("injected_scan ds={datasource:<6} rows={:<9} col={inject_column} {short}", binding_rows(bindings, binding))
        }
        Step::Merge { fragment, binding, .. } => {
            let kind = ir.fragments.get(fragment).map_or("?", fragment_kind);
            format!("merge         {kind:<16} rows={}", binding_rows(bindings, binding))
        }
        Step::Return { .. } => "return".to_string(),
    };
    eprintln!("[fedqrs] {elapsed_ms:9.2}ms  {line}");
}

/// Interpret `ir` and return the result schema and batches. Pure Rust with no
/// Python state, so the caller can run it with the GIL RELEASED - holding the
/// GIL here would freeze every Python thread (e.g. a watchdog) for the whole
/// query. The pyo3 wrapper in lib.rs turns the batches into an Arrow stream.
pub fn execute(ir: &Ir) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    let runtime = Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut bindings: HashMap<String, Binding> = HashMap::new();

    for step in &ir.steps {
        let started = std::time::Instant::now();
        match step {
            Step::SourceScan { datasource, scan, binding, materialize: _ } => {
                let batches = fetch_source(datasource, scan)?;
                bindings.insert(binding.clone(), Binding::Materialized(batches));
            }

            Step::CollectDistinct { input, key, cap: _, binding } => {
                let src = materialized(&bindings, input)?;
                let keys = runtime
                    .block_on(collect_distinct(src, key))
                    .map_err(df_to_py)?;
                bindings.insert(binding.clone(), Binding::Keys(keys));
            }

            Step::InjectedScan {
                datasource, scan, inject_column, keys_from, binding, inject_column_ndv,
            } => {
                let keys = keys_binding(&bindings, keys_from)?;
                let batches =
                    run_injected_scan(datasource, scan, inject_column, keys, *inject_column_ndv)?;
                bindings.insert(binding.clone(), Binding::Materialized(batches));
            }

            Step::Merge { fragment, inputs, binding } => {
                let frag = ir.fragments.get(fragment).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("unknown fragment '{fragment}'"))
                })?;
                let result = runtime
                    .block_on(run_fragment(&mut bindings, frag, inputs))
                    .map_err(df_to_py)?;
                bindings.insert(binding.clone(), Binding::Materialized(result));
            }

            Step::Return { input } => {
                return export(&mut bindings, input);
            }
        }
        if profile_enabled() {
            log_step(step, ir, &bindings, started.elapsed().as_secs_f64() * 1000.0);
        }
    }

    Err(PyRuntimeError::new_err("IR had no `return` step"))
}

/// A plain source read. A spec the planner marked `parallel` goes through the
/// ctid-partitioned parallel Postgres path (each partition on its own pooled
/// connection; NOTE: separate connections read separate snapshots, the same
/// trade-off the parallel probe scan already makes - a shared exported
/// snapshot is the follow-up for concurrent-write sources). Everything else
/// is a single-stream fetch.
fn fetch_source(datasource: &str, scan: &ScanSpec) -> PyResult<Batches> {
    if !scan.parallel {
        return fetch_scan(datasource, scan, None);
    }
    require_parallelizable(datasource, scan)?;
    parallel_probe_scan(datasource, scan)
}

/// A `parallel` spec must be a plain structured Postgres table read; anything
/// else is a planner contract violation and refuses loudly rather than
/// guessing (a per-partition DISTINCT or LIMIT would return wrong rows).
fn require_parallelizable(datasource: &str, scan: &ScanSpec) -> PyResult<()> {
    let postgres = connectors::kind(datasource)? == DsKind::Postgres;
    if postgres && scan.table.is_some() && !scan.distinct && scan.limit.is_none() {
        return Ok(());
    }
    Err(PyRuntimeError::new_err(
        "parallel scan spec is not a plain Postgres table read",
    ))
}

/// Render a scan to SQL and fetch it natively.
fn fetch_scan(
    datasource: &str,
    scan: &ScanSpec,
    extra_filter: Option<Expr>,
) -> PyResult<Batches> {
    let kind = connectors::kind(datasource)?;
    let sql = scan_sql(kind, scan, extra_filter).map_err(df_to_py)?;
    let (schema, batches) = connectors::fetch(datasource, &sql)?;
    Ok(Batches { schema, batches })
}

/// Borrow a binding as materialized batches.
fn materialized<'a>(
    bindings: &'a HashMap<String, Binding>,
    name: &str,
) -> PyResult<&'a Batches> {
    match bindings.get(name) {
        Some(Binding::Materialized(b)) => Ok(b),
        Some(_) => Err(PyRuntimeError::new_err(format!(
            "binding '{name}' is not a relation"
        ))),
        None => Err(PyRuntimeError::new_err(format!("unknown binding '{name}'"))),
    }
}

/// Borrow the distinct build-key values a keys binding holds.
fn keys_binding<'a>(
    bindings: &'a HashMap<String, Binding>,
    keys_from: &str,
) -> PyResult<&'a Batches> {
    match bindings.get(keys_from) {
        Some(Binding::Keys(b)) => Ok(b),
        _ => Err(PyRuntimeError::new_err(format!(
            "binding '{keys_from}' does not hold distinct keys"
        ))),
    }
}

/// Read the probe, reducing it by the build keys with the cheapest strategy for
/// the key cardinality and estimated selectivity:
///   - no keys  -> the probe matches nothing;
///   - < IN_CAP -> inline `col IN (v1, ..)` and fetch;
///   - selective -> push a TEMP TABLE of the keys and semi-join server-side;
///   - unselective (would fetch > FULL_SCAN_FRACTION) -> parallel full scan.
fn run_injected_scan(
    datasource: &str,
    scan: &ScanSpec,
    inject_column: &str,
    keys: &Batches,
    inject_column_ndv: Option<u64>,
) -> PyResult<Batches> {
    let num_keys: usize = keys.batches.iter().map(|b| b.num_rows()).sum();

    if num_keys == 0 {
        return fetch_scan(datasource, scan, Some(lit(false)));
    }
    // A small key set delivers as an inline `col IN (v1, ..)` - a bounded
    // literal filter the source plans normally. (A raw-SQL / island probe with
    // injected_sql skips this: its filter is already baked into the SQL.)
    if num_keys < IN_CAP && scan.injected_sql.is_none() {
        let filter = in_list_filter(keys, inject_column)?;
        return fetch_scan(datasource, scan, Some(filter));
    }
    let kind = connectors::kind(datasource)?;
    // Parquet reads are in-process (DataFusion); the downstream join reduces
    // without any transfer, so shipping keys anywhere would be pointless.
    if kind == DsKind::Parquet {
        return fetch_scan(datasource, scan, None);
    }
    // DuckDB: beyond the ingest ceiling, or a key type its temp-table ingest
    // cannot represent, the probe reads whole (correct, just unreduced).
    // Postgres ingests via ADBC and is bounded by the selectivity guard below.
    if kind == DsKind::DuckDb
        && (num_keys > DUCK_TEMP_CAP || !connectors::duck_can_ingest(&keys.schema))
    {
        return fetch_scan(datasource, scan, None);
    }
    // Postgres evaluates a key semi-join with an index scan ONLY when the probe
    // column is indexed; without an index it degrades to a full sequential scan
    // even at low cardinality (a worse disaster than shipping the rows), so an
    // unindexed Postgres probe reads whole and the coordinator join reduces.
    // DuckDB is columnar - its scanner ignores indexes, a semi-join is a full
    // vectorized scan regardless - so it is never gated on an index.
    if kind == DsKind::Postgres
        && scan.table.is_some()
        && !connectors::column_has_index(
            datasource, scan.schema.as_deref(), scan.table.as_deref().unwrap(), inject_column)?
    {
        return unselective_probe_scan(kind, datasource, scan);
    }
    // A raw-SQL probe (a pushed remote subtree) has no catalog identity for
    // the selectivity guard; prefer the safe temp-table path directly.
    if scan.table.is_some()
        && fetches_most_of_table(datasource, scan, inject_column, num_keys, inject_column_ndv)?
    {
        return unselective_probe_scan(kind, datasource, scan);
    }
    temp_table_probe(datasource, scan, keys, inject_column)
}

/// The probe read when the dynamic filter would select most of the table:
/// Postgres gets the ctid-partitioned parallel scan; DuckDB (columnar, no
/// secondary indexes - a semi-join costs a full scan anyway) just reads whole.
fn unselective_probe_scan(kind: DsKind, datasource: &str, scan: &ScanSpec) -> PyResult<Batches> {
    if kind == DsKind::Postgres {
        return parallel_probe_scan(datasource, scan);
    }
    fetch_scan(datasource, scan, None)
}

/// `inject_column IN (v1, v2, ...)` from the collected key values.
fn in_list_filter(keys: &Batches, inject_column: &str) -> PyResult<Expr> {
    let mut values = Vec::new();
    for batch in &keys.batches {
        let column = batch.column(0);
        for i in 0..batch.num_rows() {
            values.push(literal_from_array(column.as_ref(), i).map_err(df_to_py)?);
        }
    }
    let column = Expr::Column(Column::new(None::<TableReference>, inject_column.to_string()));
    Ok(Expr::InList(InList::new(Box::new(column), values, false)))
}

/// True when the dynamic filter is estimated to select more of the probe
/// than the source's full-scan fraction (the point where the full read -
/// ctid-parallel on Postgres, single-stream on DuckDB - beats a semi-join).
/// Unknown statistics => false (prefer the safe temp-table path).
fn fetches_most_of_table(
    datasource: &str,
    scan: &ScanSpec,
    inject_column: &str,
    num_keys: usize,
    inject_column_ndv: Option<u64>,
) -> PyResult<bool> {
    let table = scan
        .table
        .as_deref()
        .ok_or_else(|| PyRuntimeError::new_err("injected scan needs a structured table"))?;
    let threshold = match connectors::kind(datasource)? {
        DsKind::Postgres => PG_FULL_SCAN_FRACTION,
        _ => FULL_SCAN_FRACTION,
    };
    // The planner's probe-column NDV gives the real fraction (keys/NDV);
    // the source-side estimate is the fallback for unstatted columns.
    if let Some(ndv) = inject_column_ndv {
        if ndv > 0 {
            return Ok(num_keys as f64 / ndv as f64 > threshold);
        }
    }
    let fraction =
        connectors::estimate_selectivity(datasource, scan.schema.as_deref(), table, inject_column, num_keys)?;
    Ok(fraction.map(|f| f > threshold).unwrap_or(false))
}

/// Parallel ctid-partitioned full scan of the probe (base predicate only); the
/// downstream DataFusion join does the reduction.
fn parallel_probe_scan(datasource: &str, scan: &ScanSpec) -> PyResult<Batches> {
    let kind = connectors::kind(datasource)?;
    let table = scan
        .table
        .as_deref()
        .ok_or_else(|| PyRuntimeError::new_err("parallel scan needs a structured table"))?;
    let select = select_list_sql(scan);
    let where_clause = base_filter_sql(kind, scan).map_err(df_to_py)?;
    let (schema, batches) = connectors::fetch_parallel(
        datasource,
        scan.schema.as_deref(),
        table,
        scan.alias.as_deref(),
        &select,
        PARALLEL_PARTITIONS,
        where_clause.as_deref(),
    )?;
    Ok(Batches { schema, batches })
}

/// Temp-table pushdown: ingest the keys and semi-join them server-side.
fn temp_table_probe(
    datasource: &str,
    scan: &ScanSpec,
    keys: &Batches,
    inject_column: &str,
) -> PyResult<Batches> {
    let kind = connectors::kind(datasource)?;
    let key_col = keys.schema.field(0).name();
    // A planner-prerendered island with the key filter already placed on its
    // owning base relation beats the generic derived-table wrapper: sources
    // do not push a semi-join through the wrapper (q03 measured 3.1x). The
    // SQL references the same temp table this path is about to fill.
    let sql = match &scan.injected_sql {
        Some(prerendered) => prerendered.clone(),
        None => temp_join_sql(kind, scan, DYN_KEYS_TEMP_TABLE, key_col, inject_column)
            .map_err(df_to_py)?,
    };
    let (schema, batches) = connectors::fetch_temp_join(
        datasource,
        DYN_KEYS_TEMP_TABLE,
        keys.schema.clone(),
        keys.batches.clone(),
        &sql,
    )?;
    Ok(Batches { schema, batches })
}

/// DISTINCT of one key column, NULL-free (the full set; the strategy chosen in
/// `run_injected_scan` decides how to apply it).
async fn collect_distinct(src: &Batches, key: &str) -> Result<Batches, DataFusionError> {
    let ctx = memory_capped_context();
    let table = MemTable::try_new(src.schema.clone(), vec![src.batches.clone()])?;
    ctx.register_table("build", Arc::new(table))?;

    let df = ctx
        .table("build")
        .await?
        .filter(col(key).is_not_null())?
        .select(vec![col(key)])?
        .distinct()?;

    let schema = Arc::new(df.schema().as_arrow().clone());
    let batches = collect_tracked(df).await?;
    Ok(Batches { schema, batches })
}

/// Register the fragment's inputs and run it on DataFusion.
async fn run_fragment(
    bindings: &mut HashMap<String, Binding>,
    fragment: &Fragment,
    inputs: &BTreeMap<String, String>,
) -> Result<Batches, DataFusionError> {
    let ctx = memory_capped_context();
    for (table_name, binding_name) in inputs {
        let b = take_materialized(bindings, binding_name)?;
        let table = MemTable::try_new(b.schema, vec![b.batches])?;
        ctx.register_table(table_name.as_str(), Arc::new(table))?;
    }

    match fragment {
        Fragment::HashJoin { join_type, left_keys, right_keys, project } => {
            run_hash_join(&ctx, *join_type, left_keys, right_keys, project).await
        }
        Fragment::NestedLoopJoin { join_type, condition, project } => {
            run_nested_loop_join(&ctx, *join_type, condition, project).await
        }
        Fragment::Project { project } => run_project(&ctx, project).await,
        Fragment::Aggregate { select, group_by, grouping_sets } => {
            run_aggregate(&ctx, select, group_by, grouping_sets).await
        }
        Fragment::Sort { keys } => run_sort(&ctx, keys).await,
        Fragment::Filter { predicate } => run_filter(&ctx, predicate).await,
        Fragment::Limit { limit, offset } => run_limit(&ctx, *limit, *offset).await,
        Fragment::RawSql { sql } => run_raw_sql(&ctx, sql).await,
    }
}

/// Run pre-rendered SQL over the registered merge inputs (e.g. a whole CTE).
async fn run_raw_sql(ctx: &SessionContext, sql: &str) -> Result<Batches, DataFusionError> {
    let df = ctx.sql(sql).await?;
    collect_batches(df).await
}

/// Apply LIMIT/OFFSET over the single input `in_0`.
async fn run_limit(
    ctx: &SessionContext,
    limit: Option<usize>,
    offset: usize,
) -> Result<Batches, DataFusionError> {
    let limited = ctx.table("in_0").await?.limit(offset, limit)?;
    collect_batches(limited).await
}

/// Filter the single input `in_0` by a boolean predicate.
async fn run_filter(
    ctx: &SessionContext,
    predicate: &fedqrs_core::ir::IrExpr,
) -> Result<Batches, DataFusionError> {
    let filtered = ctx.table("in_0").await?.filter(to_df_expr(predicate)?)?;
    collect_batches(filtered).await
}

/// Order the single input `in_0` by the given keys.
async fn run_sort(
    ctx: &SessionContext,
    keys: &[fedqrs_core::ir::SortKey],
) -> Result<Batches, DataFusionError> {
    let mut sort_exprs = Vec::with_capacity(keys.len());
    for k in keys {
        sort_exprs.push(to_df_expr(&k.expr)?.sort(k.ascending, k.nulls_first));
    }
    let sorted = ctx.table("in_0").await?.sort(sort_exprs)?;
    collect_batches(sorted).await
}

/// Run a GROUP BY over the single input `in_0` by building DataFusion SQL, so
/// every aggregate function DataFusion knows works without per-function wiring.
async fn run_aggregate(
    ctx: &SessionContext,
    select: &[AggSelectItem],
    group_by: &[fedqrs_core::ir::IrExpr],
    grouping_sets: &[Vec<fedqrs_core::ir::IrExpr>],
) -> Result<Batches, DataFusionError> {
    let mut items = Vec::with_capacity(select.len());
    for item in select {
        let rendered = match (&item.agg, &item.expr) {
            (Some(agg), _) => render_agg(agg)?,
            (None, Some(expr)) => render_expr(&to_df_expr(expr)?)?,
            (None, None) => {
                return Err(DataFusionError::Plan(
                    "aggregate select item has neither expr nor agg".into(),
                ))
            }
        };
        items.push(format!("{rendered} AS {}", quote_ident(&item.alias)));
    }

    let mut sql = format!("SELECT {} FROM \"in_0\"", items.join(", "));
    sql.push_str(&group_by_clause(group_by, grouping_sets)?);

    let df = ctx.sql(&sql).await?;
    collect_batches(df).await
}

/// Collect a DataFrame into Batches under ONE schema all batches conform to.
///
/// The Binding's schema must match its batches or registering it as a MemTable
/// rejects it ("Mismatch between schema and batches"). Two ways the logical
/// `df.schema()` and the executed batches disagree:
/// - TYPE: SUM widens decimal precision in execution (Decimal(17,2) ->
///   Decimal(27,2)) but not in the logical schema (q16/q94/q95).
/// - NULLABILITY: a UNION's branches can disagree on a column's nullability, so
///   the collected batches disagree with EACH OTHER (q77).
/// So take the executed types from the first batch, widen every field to
/// nullable (the safe superset), and re-schema every batch to it. Falls back to
/// the logical schema when empty.
async fn collect_batches(
    df: datafusion::dataframe::DataFrame,
) -> Result<Batches, DataFusionError> {
    let logical = Arc::new(df.schema().as_arrow().clone());
    let batches = collect_tracked(df).await?;
    let schema = match batches.first() {
        Some(batch) => nullable_schema(&batch.schema()),
        None => logical,
    };
    let batches = reschema(batches, &schema)?;
    Ok(Batches { schema, batches })
}

/// Materialize a DataFrame while charging every accumulated batch to the
/// shared memory pool. Operators only account their WORKING memory, so a
/// fragment whose OUTPUT explodes (a cross join's cartesian product) would
/// otherwise grow unseen past the cap and OOM the server; charging the
/// accumulation makes it fail with ResourcesExhausted instead. The
/// reservation is released once the batches are handed on as a binding.
async fn collect_tracked(
    df: datafusion::dataframe::DataFrame,
) -> Result<Vec<RecordBatch>, DataFusionError> {
    let reservation =
        MemoryConsumer::new("fedq_collect").register(&runtime_env().memory_pool);
    let mut stream = df.execute_stream().await?;
    let mut batches = Vec::new();
    while let Some(item) = stream.next().await {
        let batch = item?;
        reservation.try_grow(batch.get_array_memory_size())?;
        batches.push(batch);
    }
    Ok(batches)
}

/// A copy of `schema` with every field marked nullable, keeping each field's
/// executed data type. Lets batches that differ only on per-column nullability
/// (a UNION branch whose column is non-null vs one where it is) share one schema.
fn nullable_schema(schema: &Schema) -> SchemaRef {
    let mut fields = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        fields.push(Field::new(field.name(), field.data_type().clone(), true));
    }
    Arc::new(Schema::new(fields))
}

/// The GROUP BY clause: `GROUPING SETS (...)` when sets are given, else a plain
/// grouping key list, else empty.
fn group_by_clause(
    group_by: &[fedqrs_core::ir::IrExpr],
    grouping_sets: &[Vec<fedqrs_core::ir::IrExpr>],
) -> Result<String, DataFusionError> {
    if !grouping_sets.is_empty() {
        return grouping_sets_clause(grouping_sets);
    }
    if group_by.is_empty() {
        return Ok(String::new());
    }
    let mut groups = Vec::with_capacity(group_by.len());
    for g in group_by {
        groups.push(render_expr(&to_df_expr(g)?)?);
    }
    Ok(format!(" GROUP BY {}", groups.join(", ")))
}

/// Render `GROUP BY GROUPING SETS ((a, b), (a), ())`.
fn grouping_sets_clause(
    grouping_sets: &[Vec<fedqrs_core::ir::IrExpr>],
) -> Result<String, DataFusionError> {
    let mut rendered = Vec::with_capacity(grouping_sets.len());
    for set in grouping_sets {
        let mut cols = Vec::with_capacity(set.len());
        for expr in set {
            cols.push(render_expr(&to_df_expr(expr)?)?);
        }
        rendered.push(format!("({})", cols.join(", ")));
    }
    Ok(format!(" GROUP BY GROUPING SETS ({})", rendered.join(", ")))
}

fn render_agg(agg: &AggCall) -> Result<String, DataFusionError> {
    let inner = if agg.star {
        "*".to_string()
    } else {
        let mut parts = Vec::with_capacity(agg.args.len());
        for a in &agg.args {
            parts.push(render_expr(&to_df_expr(a)?)?);
        }
        parts.join(", ")
    };
    let distinct = if agg.distinct { "DISTINCT " } else { "" };
    let mut sql = format!("{}({}{})", agg.func, distinct, inner);
    if let Some(wg) = &agg.within_group {
        let key = render_expr(&to_df_expr(&wg.key)?)?;
        let direction = if wg.desc { " DESC" } else { "" };
        sql.push_str(&format!(" WITHIN GROUP (ORDER BY {key}{direction})"));
    }
    Ok(sql)
}

/// Double-quote an identifier for a DataFusion SQL alias.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Evaluate a projection over the single input `in_0`.
async fn run_project(
    ctx: &SessionContext,
    project: &[Projection],
) -> Result<Batches, DataFusionError> {
    project_dataframe(ctx.table("in_0").await?, project).await
}

/// Project a DataFrame to its output columns. Each expression is aliased to a
/// unique internal name (DataFusion requires unique projection names), then the
/// result schema is renamed to the intended output names - which MAY repeat, as
/// SQL allows (`SELECT a, a`, self-joins); Arrow permits duplicate field names.
async fn project_dataframe(
    df: datafusion::dataframe::DataFrame,
    project: &[Projection],
) -> Result<Batches, DataFusionError> {
    let mut exprs = Vec::with_capacity(project.len());
    for (i, p) in project.iter().enumerate() {
        exprs.push(to_df_expr(&p.expr)?.alias(format!("__c{i}")));
    }
    let projected = df.select(exprs)?;
    let internal = projected.schema().as_arrow().clone();
    let batches = projected.collect().await?;
    let schema = output_schema(&internal, project);
    let batches = reschema(batches, &schema)?;
    Ok(Batches { schema, batches })
}

/// The output schema: internal column types under the intended output names.
fn output_schema(internal: &Schema, project: &[Projection]) -> SchemaRef {
    let mut fields = Vec::with_capacity(project.len());
    for (field, p) in internal.fields().iter().zip(project) {
        fields.push(Field::new(&p.alias, field.data_type().clone(), field.is_nullable()));
    }
    Arc::new(Schema::new(fields))
}

/// Rebuild each batch under `schema` (same columns, renamed fields).
fn reschema(
    batches: Vec<RecordBatch>,
    schema: &SchemaRef,
) -> Result<Vec<RecordBatch>, DataFusionError> {
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        out.push(RecordBatch::try_new(schema.clone(), batch.columns().to_vec())?);
    }
    Ok(out)
}

async fn run_hash_join(
    ctx: &SessionContext,
    join_type: JoinKind,
    left_keys: &[String],
    right_keys: &[String],
    project: &[Projection],
) -> Result<Batches, DataFusionError> {
    let left = ctx.table("in_left").await?;
    let right = ctx.table("in_right").await?;
    let lk: Vec<&str> = left_keys.iter().map(|s| s.as_str()).collect();
    let rk: Vec<&str> = right_keys.iter().map(|s| s.as_str()).collect();

    let joined = left.join(right, datafusion_join_type(join_type), &lk, &rk, None)?;
    project_dataframe(joined, project).await
}

/// A non-equi (nested-loop) join on an arbitrary condition (`None` = cross join).
async fn run_nested_loop_join(
    ctx: &SessionContext,
    join_type: JoinKind,
    condition: &Option<fedqrs_core::ir::IrExpr>,
    project: &[Projection],
) -> Result<Batches, DataFusionError> {
    let left = ctx.table("in_left").await?;
    let right = ctx.table("in_right").await?;
    let on_exprs = match condition {
        Some(expr) => vec![to_df_expr(expr)?],
        None => Vec::new(),
    };
    let joined = left.join_on(right, datafusion_join_type(join_type), on_exprs)?;
    project_dataframe(joined, project).await
}

/// Map the IR join kind to DataFusion's. A free function, not a `From` impl:
/// both types are external to this crate (JoinKind in fedqrs-core, JoinType in
/// datafusion), so the orphan rule forbids the trait impl here.
fn datafusion_join_type(k: JoinKind) -> JoinType {
    match k {
        JoinKind::Inner => JoinType::Inner,
        JoinKind::Left => JoinType::Left,
        JoinKind::Right => JoinType::Right,
        JoinKind::Full => JoinType::Full,
        JoinKind::Semi => JoinType::LeftSemi,
        JoinKind::Anti => JoinType::LeftAnti,
    }
}

/// Consume a binding as materialized batches.
fn take_materialized(
    bindings: &mut HashMap<String, Binding>,
    name: &str,
) -> Result<Batches, DataFusionError> {
    match bindings.remove(name) {
        Some(Binding::Materialized(b)) => Ok(b),
        Some(_) => Err(DataFusionError::Plan(format!(
            "binding '{name}' is not a relation"
        ))),
        None => Err(DataFusionError::Plan(format!("unknown binding '{name}'"))),
    }
}

/// Take the named binding as the result schema and batches.
fn export(
    bindings: &mut HashMap<String, Binding>,
    input: &str,
) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    match bindings.remove(input) {
        Some(Binding::Materialized(b)) => Ok((b.schema, b.batches)),
        Some(_) => Err(PyRuntimeError::new_err(format!(
            "cannot return non-relation binding '{input}'"
        ))),
        None => Err(PyRuntimeError::new_err(format!(
            "return of unknown binding '{input}'"
        ))),
    }
}
