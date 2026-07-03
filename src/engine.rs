//! The IR interpreter: walks the orchestration steps, reads sources natively,
//! runs relational fragments on DataFusion, and returns the result as an
//! exportable Arrow stream.
//!
//! Everything stays in Rust. Source reads go over native drivers; the semi-join
//! reduction computes the build's distinct keys and injects them into the probe
//! SQL here, so key values are never handed to Python. Only the final result
//! crosses the boundary, once.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::{Column, DataFusionError, JoinType, TableReference};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::Expr;
use datafusion::prelude::{col, lit, SessionContext};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use crate::connectors;
use crate::ffi::{stream_from_batches, ArrowStreamExport};
use fedqrs_core::types::DsKind;
use fedqrs_core::expr::{literal_from_array, to_df_expr};
use fedqrs_core::ir::{AggCall, AggSelectItem, Fragment, Ir, JoinKind, Projection, ScanSpec, Step};
use fedqrs_core::sql::{base_filter_sql, render_expr, scan_sql, select_list_sql, temp_join_sql};

/// Dynamic-filter strategy thresholds. Under `IN_CAP` distinct keys we inline an
/// IN list; above it we push a temp table unless the filter would select more
/// than `FULL_SCAN_FRACTION` of the probe, in which case a parallel full scan of
/// the (bandwidth-bound) table wins. Partition count is tuned near the core count.
const IN_CAP: usize = 2000;
const FULL_SCAN_FRACTION: f64 = 0.40;
const PARALLEL_PARTITIONS: usize = 8;
const DYN_KEYS_TEMP_TABLE: &str = "fedq_dyn_keys";

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

/// Interpret `ir` and return the result stream ready for export to Python.
pub fn execute(ir: &Ir) -> PyResult<ArrowStreamExport> {
    let runtime = Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut bindings: HashMap<String, Binding> = HashMap::new();

    for step in &ir.steps {
        match step {
            Step::SourceScan { datasource, scan, binding, materialize: _ } => {
                let batches = fetch_scan(datasource, scan, None)?;
                bindings.insert(binding.clone(), Binding::Materialized(batches));
            }

            Step::CollectDistinct { input, key, cap: _, binding } => {
                let src = materialized(&bindings, input)?;
                let keys = runtime
                    .block_on(collect_distinct(src, key))
                    .map_err(df_to_py)?;
                bindings.insert(binding.clone(), Binding::Keys(keys));
            }

            Step::InjectedScan { datasource, scan, inject_column, keys_from, binding } => {
                let keys = keys_binding(&bindings, keys_from)?;
                let batches = run_injected_scan(datasource, scan, inject_column, keys)?;
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
    }

    Err(PyRuntimeError::new_err("IR had no `return` step"))
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
) -> PyResult<Batches> {
    let num_keys: usize = keys.batches.iter().map(|b| b.num_rows()).sum();

    if num_keys == 0 {
        return fetch_scan(datasource, scan, Some(lit(false)));
    }
    if num_keys < IN_CAP {
        let filter = in_list_filter(keys, inject_column)?;
        return fetch_scan(datasource, scan, Some(filter));
    }
    // The temp-table ingest and parallel ctid scan are Postgres-specific. For
    // other sources, fetch the probe in full and let the DataFusion join reduce.
    if connectors::kind(datasource)? != DsKind::Postgres {
        return fetch_scan(datasource, scan, None);
    }
    if fetches_most_of_table(datasource, scan, inject_column, num_keys)? {
        parallel_probe_scan(datasource, scan)
    } else {
        temp_table_probe(datasource, scan, keys, inject_column)
    }
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

/// True when the dynamic filter is estimated to select more than
/// `FULL_SCAN_FRACTION` of the probe (so a parallel full scan beats a semi-join).
/// Unknown statistics => false (prefer the safe temp-table path).
fn fetches_most_of_table(
    datasource: &str,
    scan: &ScanSpec,
    inject_column: &str,
    num_keys: usize,
) -> PyResult<bool> {
    let table = scan
        .table
        .as_deref()
        .ok_or_else(|| PyRuntimeError::new_err("injected scan needs a structured table"))?;
    let fraction =
        connectors::estimate_selectivity(datasource, scan.schema.as_deref(), table, inject_column, num_keys)?;
    Ok(fraction.map(|f| f > FULL_SCAN_FRACTION).unwrap_or(false))
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
    let sql = temp_join_sql(kind, scan, DYN_KEYS_TEMP_TABLE, key_col, inject_column).map_err(df_to_py)?;
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
    let ctx = SessionContext::new();
    let table = MemTable::try_new(src.schema.clone(), vec![src.batches.clone()])?;
    ctx.register_table("build", Arc::new(table))?;

    let df = ctx
        .table("build")
        .await?
        .filter(col(key).is_not_null())?
        .select(vec![col(key)])?
        .distinct()?;

    let schema = Arc::new(df.schema().as_arrow().clone());
    let batches = df.collect().await?;
    Ok(Batches { schema, batches })
}

/// Register the fragment's inputs and run it on DataFusion.
async fn run_fragment(
    bindings: &mut HashMap<String, Binding>,
    fragment: &Fragment,
    inputs: &BTreeMap<String, String>,
) -> Result<Batches, DataFusionError> {
    let ctx = SessionContext::new();
    for (table_name, binding_name) in inputs {
        let b = take_materialized(bindings, binding_name)?;
        let table = MemTable::try_new(b.schema, vec![b.batches])?;
        ctx.register_table(table_name.as_str(), Arc::new(table))?;
    }

    match fragment {
        Fragment::HashJoin { join_type, left_keys, right_keys, project } => {
            run_hash_join(&ctx, *join_type, left_keys, right_keys, project).await
        }
        Fragment::Project { project } => run_project(&ctx, project).await,
        Fragment::Aggregate { select, group_by } => {
            run_aggregate(&ctx, select, group_by).await
        }
        Fragment::Sort { keys } => run_sort(&ctx, keys).await,
        Fragment::Filter { predicate } => run_filter(&ctx, predicate).await,
        Fragment::Limit { limit, offset } => run_limit(&ctx, *limit, *offset).await,
    }
}

/// Apply LIMIT/OFFSET over the single input `in_0`.
async fn run_limit(
    ctx: &SessionContext,
    limit: Option<usize>,
    offset: usize,
) -> Result<Batches, DataFusionError> {
    let limited = ctx.table("in_0").await?.limit(offset, limit)?;
    let schema = Arc::new(limited.schema().as_arrow().clone());
    let batches = limited.collect().await?;
    Ok(Batches { schema, batches })
}

/// Filter the single input `in_0` by a boolean predicate.
async fn run_filter(
    ctx: &SessionContext,
    predicate: &fedqrs_core::ir::IrExpr,
) -> Result<Batches, DataFusionError> {
    let filtered = ctx.table("in_0").await?.filter(to_df_expr(predicate)?)?;
    let schema = Arc::new(filtered.schema().as_arrow().clone());
    let batches = filtered.collect().await?;
    Ok(Batches { schema, batches })
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
    let schema = Arc::new(sorted.schema().as_arrow().clone());
    let batches = sorted.collect().await?;
    Ok(Batches { schema, batches })
}

/// Run a GROUP BY over the single input `in_0` by building DataFusion SQL, so
/// every aggregate function DataFusion knows works without per-function wiring.
async fn run_aggregate(
    ctx: &SessionContext,
    select: &[AggSelectItem],
    group_by: &[fedqrs_core::ir::IrExpr],
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
    if !group_by.is_empty() {
        let mut groups = Vec::with_capacity(group_by.len());
        for g in group_by {
            groups.push(render_expr(&to_df_expr(g)?)?);
        }
        sql.push_str(&format!(" GROUP BY {}", groups.join(", ")));
    }

    let df = ctx.sql(&sql).await?;
    let schema = Arc::new(df.schema().as_arrow().clone());
    let batches = df.collect().await?;
    Ok(Batches { schema, batches })
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
    Ok(format!("{}({}{})", agg.func, distinct, inner))
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

/// Export the named binding as a Python-consumable Arrow stream.
fn export(bindings: &mut HashMap<String, Binding>, input: &str) -> PyResult<ArrowStreamExport> {
    match bindings.remove(input) {
        Some(Binding::Materialized(b)) => Ok(stream_from_batches(b.schema, b.batches)),
        Some(_) => Err(PyRuntimeError::new_err(format!(
            "cannot return non-relation binding '{input}'"
        ))),
        None => Err(PyRuntimeError::new_err(format!(
            "return of unknown binding '{input}'"
        ))),
    }
}
