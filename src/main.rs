//! Minimal Postgres -> Arrow fetch benchmark, mirroring the Python connector
//! path (fetch all rows, then build one Arrow array per column). Prints the
//! fetch time and the Arrow-build time separately so it can be compared to the
//! psycopg2 -> Python -> pyarrow numbers.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use postgres::types::Type;
use postgres::{Client, NoTls, Row};

mod bench_adbc_datafusion;
mod bench_adbc_duckdb;
mod bench_datafusion;
mod bench_federated;
mod bench_duckdb;
mod bench_ext;

const DEFAULT_ADBC_DRIVER: &str =
    "/workspace/venv-fedq/lib/python3.13/site-packages/adbc_driver_postgresql/libadbc_driver_postgresql.so";

fn uri(args: &HashMap<String, String>) -> String {
    let get = |k: &str, d: &str| args.get(k).cloned().unwrap_or_else(|| d.to_string());
    let host = get("host", "localhost");
    let port = get("port", "5432");
    let user = get("user", "postgres");
    let db = args
        .get("database")
        .or_else(|| args.get("dbname"))
        .cloned()
        .unwrap_or_else(|| "test_db".to_string());
    let auth = match args.get("password") {
        Some(pw) => format!("{user}:{pw}"),
        None => user,
    };
    format!("postgresql://{auth}@{host}:{port}/{db}")
}

fn parse_args() -> HashMap<String, String> {
    // Accepts --key=value and --key value.
    let mut map = HashMap::new();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        if let Some(rest) = arg.strip_prefix("--") {
            if let Some(eq) = rest.find('=') {
                map.insert(rest[..eq].to_string(), rest[eq + 1..].to_string());
            } else if i + 1 < argv.len() {
                map.insert(rest.to_string(), argv[i + 1].clone());
                i += 1;
            } else {
                map.insert(rest.to_string(), String::new());
            }
        }
        i += 1;
    }
    map
}

fn conn_string(args: &HashMap<String, String>) -> String {
    let get = |k: &str, d: &str| args.get(k).cloned().unwrap_or_else(|| d.to_string());
    let host = get("host", "localhost");
    let port = get("port", "5432");
    let user = get("user", "postgres");
    let db = args
        .get("database")
        .or_else(|| args.get("dbname"))
        .cloned()
        .unwrap_or_else(|| "test_db".to_string());
    let mut s = format!("host={host} port={port} user={user} dbname={db}");
    if let Some(pw) = args.get("password") {
        s.push_str(&format!(" password={pw}"));
    }
    s
}

/// Build one Arrow array (and its field type) for column `i` of `rows`.
fn build_column(rows: &[Row], i: usize, ty: &Type) -> (ArrayRef, DataType) {
    match ty.name() {
        "int8" => int_array(rows.iter().map(|r| r.get::<_, Option<i64>>(i)).collect()),
        "int4" => int_array(
            rows.iter()
                .map(|r| r.get::<_, Option<i32>>(i).map(|v| v as i64))
                .collect(),
        ),
        "int2" => int_array(
            rows.iter()
                .map(|r| r.get::<_, Option<i16>>(i).map(|v| v as i64))
                .collect(),
        ),
        "float8" => float_array(rows.iter().map(|r| r.get::<_, Option<f64>>(i)).collect()),
        "float4" => float_array(
            rows.iter()
                .map(|r| r.get::<_, Option<f32>>(i).map(|v| v as f64))
                .collect(),
        ),
        "bool" => {
            let v: Vec<Option<bool>> = rows.iter().map(|r| r.get::<_, Option<bool>>(i)).collect();
            (Arc::new(BooleanArray::from(v)), DataType::Boolean)
        }
        "uuid" => str_array(
            rows.iter()
                .map(|r| r.get::<_, Option<uuid::Uuid>>(i).map(|u| u.to_string()))
                .collect(),
        ),
        "timestamptz" => str_array(
            rows.iter()
                .map(|r| {
                    r.get::<_, Option<chrono::DateTime<chrono::Utc>>>(i)
                        .map(|t| t.to_rfc3339())
                })
                .collect(),
        ),
        "timestamp" => str_array(
            rows.iter()
                .map(|r| {
                    r.get::<_, Option<chrono::NaiveDateTime>>(i)
                        .map(|t| t.to_string())
                })
                .collect(),
        ),
        "date" => str_array(
            rows.iter()
                .map(|r| {
                    r.get::<_, Option<chrono::NaiveDate>>(i)
                        .map(|t| t.to_string())
                })
                .collect(),
        ),
        "text" | "varchar" | "bpchar" | "name" => str_array(
            rows.iter().map(|r| r.get::<_, Option<String>>(i)).collect(),
        ),
        other => panic!("unsupported column type '{other}' (cast it to text in --sql)"),
    }
}

fn int_array(v: Vec<Option<i64>>) -> (ArrayRef, DataType) {
    (Arc::new(Int64Array::from(v)), DataType::Int64)
}

fn float_array(v: Vec<Option<f64>>) -> (ArrayRef, DataType) {
    (Arc::new(Float64Array::from(v)), DataType::Float64)
}

fn str_array(v: Vec<Option<String>>) -> (ArrayRef, DataType) {
    (Arc::new(StringArray::from_iter(v.into_iter())), DataType::Utf8)
}

fn build_batch(rows: &[Row]) -> RecordBatch {
    let columns: &[postgres::Column] = rows.first().map(|r| r.columns()).unwrap_or(&[]);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    let mut fields: Vec<Field> = Vec::with_capacity(columns.len());
    for (i, col) in columns.iter().enumerate() {
        let (array, dt) = build_column(rows, i, col.type_());
        fields.push(Field::new(col.name(), dt, true));
        arrays.push(array);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).expect("record batch")
}

fn usage() {
    eprintln!(
        "fedqrs — Postgres -> Arrow fetch benchmark\n\n\
         USAGE:\n  fedqrs [--host H] [--port P] [--user U] [--password PW] \
         [--database DB] --sql \"SELECT ...\"\n\n\
         OPTIONS (--key=value or --key value):\n\
         \x20 --host       default localhost\n\
         \x20 --port       default 5432\n\
         \x20 --user       default postgres\n\
         \x20 --password   default: none\n\
         \x20 --database   (alias --dbname) default test_db\n\
         \x20 --sql        query to run (default: the perf_files wide query)\n\
         \x20 -h, --help   show this help\n\n\
         Prints fetch vs Arrow-build time for a warmup + 3 timed runs."
    );
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        usage();
        return;
    }

    let args = parse_args();
    // Self-contained DuckDB uuid experiment; needs no Postgres connection.
    if args.contains_key("duckdb") {
        bench_duckdb::run();
        return;
    }
    if args.contains_key("datafusion") {
        bench_datafusion::run();
        return;
    }
    let sql = args
        .get("sql")
        .cloned()
        .unwrap_or_else(|| "SELECT id, table_id, path, file_type, file_size_bytes, file_hash, created_at FROM perf_files".to_string());
    let conn = conn_string(&args);
    let uri = uri(&args);
    let driver = args
        .get("adbc-driver")
        .cloned()
        .unwrap_or_else(|| DEFAULT_ADBC_DRIVER.to_string());

    if args.contains_key("adbc-df") {
        bench_adbc_datafusion::run(&driver, &uri, &sql);
        return;
    }
    if args.contains_key("adbc-duckdb") {
        bench_adbc_duckdb::run(&driver, &uri, &sql);
        return;
    }
    if args.contains_key("federated") {
        let duck_path = args
            .get("duckdb-path")
            .cloned()
            .unwrap_or_else(|| "/tmp/fedq_source.duckdb".to_string());
        bench_federated::run(&driver, &uri, &duck_path);
        return;
    }

    println!("sql: {sql}\n(best of 3 timed runs, after a warmup)\n");

    // 1) rust-postgres + hand-built Arrow (mirrors the Python psycopg2 path)
    let mut client = match Client::connect(&conn, NoTls) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not connect ({conn}): {e}");
            std::process::exit(1);
        }
    };
    bench_ext::time_best("rust-postgres+arrow", || {
        let rows = client.query(sql.as_str(), &[]).expect("query");
        build_batch(&rows).num_rows()
    });

    // 2) ADBC (C driver, postgres wire -> Arrow directly)
    bench_ext::bench_adbc(&driver, &uri, &sql);

    // 3) connectorx (native Rust, postgres -> Arrow)
    bench_ext::bench_connectorx(&uri, &sql);
}
