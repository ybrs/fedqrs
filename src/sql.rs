//! Render a `ScanSpec` to dialect-correct source SQL, in Rust.
//!
//! The SELECT skeleton (columns, FROM, LIMIT) is hand-built; filter expressions
//! are rendered by DataFusion's unparser so dialect-divergent syntax (quoting,
//! literal formatting) is handled without a bespoke per-dialect emitter. The
//! dynamic semi-join filter arrives as an already-built `Expr` and is ANDed in,
//! so the build-side key values are formatted here in Rust and never seen by
//! Python.

use datafusion::common::DataFusionError;
use datafusion::logical_expr::Expr;
use datafusion::sql::unparser::dialect::{DefaultDialect, Dialect, PostgreSqlDialect};
use datafusion::sql::unparser::Unparser;

use crate::connectors::DsKind;
use crate::expr::to_df_expr;
use crate::ir::ScanSpec;

/// Quote an identifier for Postgres/DuckDB (double quotes, doubled internally).
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// `"schema"."table" [AS "alias"]`
fn table_ref(scan: &ScanSpec, table: &str) -> String {
    let mut s = String::new();
    if let Some(schema) = &scan.schema {
        s.push_str(&quote_ident(schema));
        s.push('.');
    }
    s.push_str(&quote_ident(table));
    if let Some(alias) = &scan.alias {
        s.push_str(" AS ");
        s.push_str(&quote_ident(alias));
    }
    s
}

fn dialect_for(kind: DsKind) -> Box<dyn Dialect> {
    match kind {
        DsKind::Postgres => Box::new(PostgreSqlDialect {}),
        // DuckDB SQL is standard enough for the default dialect for now.
        DsKind::DuckDb => Box::new(DefaultDialect {}),
    }
}

/// Render `scan` to a SQL string. `extra_filter`, if present, is ANDed into the
/// WHERE clause (this is the runtime `col IN (...)` semi-join reduction).
pub fn scan_sql(
    kind: DsKind,
    scan: &ScanSpec,
    extra_filter: Option<Expr>,
) -> Result<String, DataFusionError> {
    // Pre-rendered SQL (a complex single-source subtree). No dynamic filter can
    // be spliced into opaque SQL, so an injected scan must be structured.
    if let Some(raw) = &scan.raw_sql {
        if extra_filter.is_some() {
            return Err(DataFusionError::Plan(
                "cannot inject a dynamic filter into a raw_sql scan".into(),
            ));
        }
        return Ok(raw.clone());
    }

    let table = scan.table.as_ref().ok_or_else(|| {
        DataFusionError::Plan("structured scan needs a 'table'".into())
    })?;
    if scan.columns.is_empty() {
        return Err(DataFusionError::Plan(format!(
            "scan of '{table}' has no output columns"
        )));
    }

    let mut sql = String::from("SELECT ");
    if scan.distinct {
        sql.push_str("DISTINCT ");
    }
    let cols: Vec<String> = scan.columns.iter().map(|c| quote_ident(c)).collect();
    sql.push_str(&cols.join(", "));
    sql.push_str(" FROM ");
    sql.push_str(&table_ref(scan, table));

    // Combine the static filter with the dynamic one (either may be absent).
    let filter: Option<Expr> = match (&scan.filter, extra_filter) {
        (Some(f), Some(x)) => Some(to_df_expr(f)?.and(x)),
        (Some(f), None) => Some(to_df_expr(f)?),
        (None, Some(x)) => Some(x),
        (None, None) => None,
    };
    if let Some(f) = filter {
        let dialect = dialect_for(kind);
        let unparser = Unparser::new(dialect.as_ref());
        let rendered = unparser.expr_to_sql(&f)?;
        sql.push_str(" WHERE ");
        sql.push_str(&rendered.to_string());
    }

    if let Some(n) = scan.limit {
        sql.push_str(&format!(" LIMIT {n}"));
    }
    Ok(sql)
}
