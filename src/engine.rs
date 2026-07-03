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
use arrow::datatypes::SchemaRef;
use datafusion::common::{Column, DataFusionError, JoinType, TableReference};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::Expr;
use datafusion::prelude::{col, lit, SessionContext};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use crate::connectors;
use crate::expr::{literal_from_array, to_df_expr};
use crate::ffi::{stream_from_batches, ArrowStreamExport};
use crate::ir::{Fragment, Ir, JoinKind, Projection, ScanSpec, Step};
use crate::sql::scan_sql;

/// Fully-read Arrow data with its schema.
struct Batches {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

/// A named intermediate produced by a step.
enum Binding {
    /// A relation held in memory (source result or fragment output).
    Materialized(Batches),
    /// Distinct build-side keys, or None when the count exceeded the cap.
    Keys(Option<Batches>),
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

            Step::CollectDistinct { input, key, cap, binding } => {
                let src = materialized(&bindings, input)?;
                let keys = runtime
                    .block_on(collect_distinct(src, key, *cap))
                    .map_err(df_to_py)?;
                bindings.insert(binding.clone(), Binding::Keys(keys));
            }

            Step::InjectedScan { datasource, scan, inject_column, keys_from, binding } => {
                let extra = dynamic_filter(&bindings, keys_from, inject_column)?;
                let batches = fetch_scan(datasource, scan, extra)?;
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

/// Build the probe's dynamic `inject_column IN (...)` filter from a keys
/// binding. None => over cap (full scan). Empty keys => `false` (no build rows
/// can match, so the probe returns nothing).
fn dynamic_filter(
    bindings: &HashMap<String, Binding>,
    keys_from: &str,
    inject_column: &str,
) -> PyResult<Option<Expr>> {
    let keys = match bindings.get(keys_from) {
        Some(Binding::Keys(k)) => k,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "binding '{keys_from}' does not hold distinct keys"
            )))
        }
    };
    let batches = match keys {
        None => return Ok(None),
        Some(b) => b,
    };

    let mut values = Vec::new();
    for batch in &batches.batches {
        let column = batch.column(0);
        for i in 0..batch.num_rows() {
            values.push(literal_from_array(column.as_ref(), i).map_err(df_to_py)?);
        }
    }
    let column = Expr::Column(Column::new(None::<TableReference>, inject_column.to_string()));
    if values.is_empty() {
        Ok(Some(lit(false)))
    } else {
        Ok(Some(Expr::InList(InList::new(Box::new(column), values, false))))
    }
}

/// DISTINCT of one key column, NULL-free, capped. None if the distinct count
/// exceeds `cap` (over cap => no dynamic filter is pushed).
async fn collect_distinct(
    src: &Batches,
    key: &str,
    cap: usize,
) -> Result<Option<Batches>, DataFusionError> {
    let ctx = SessionContext::new();
    let table = MemTable::try_new(src.schema.clone(), vec![src.batches.clone()])?;
    ctx.register_table("build", Arc::new(table))?;

    // cap + 1 so we can distinguish "at cap" from "over cap".
    let df = ctx
        .table("build")
        .await?
        .filter(col(key).is_not_null())?
        .select(vec![col(key)])?
        .distinct()?
        .limit(0, Some(cap + 1))?;

    let schema = Arc::new(df.schema().as_arrow().clone());
    let batches = df.collect().await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    if rows > cap {
        Ok(None)
    } else {
        Ok(Some(Batches { schema, batches }))
    }
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
    }
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

    let joined = left.join(right, join_type.into(), &lk, &rk, None)?;

    let mut exprs = Vec::with_capacity(project.len());
    for p in project {
        exprs.push(to_df_expr(&p.expr)?.alias(p.alias.as_str()));
    }
    let projected = joined.select(exprs)?;

    let schema = Arc::new(projected.schema().as_arrow().clone());
    let batches = projected.collect().await?;
    Ok(Batches { schema, batches })
}

impl From<JoinKind> for JoinType {
    fn from(k: JoinKind) -> Self {
        match k {
            JoinKind::Inner => JoinType::Inner,
            JoinKind::Left => JoinType::Left,
            JoinKind::Right => JoinType::Right,
            JoinKind::Full => JoinType::Full,
            JoinKind::Semi => JoinType::LeftSemi,
            JoinKind::Anti => JoinType::LeftAnti,
        }
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
