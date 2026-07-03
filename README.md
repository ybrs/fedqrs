# fedqrs — Postgres → Arrow fetch benchmark

Compares three ways of pulling a SQL result into Arrow, to put numbers on the
psycopg2 → Python → pyarrow path used by the engine:

1. **rust-postgres + hand-built Arrow** — fetch all rows, build one Arrow array
   per column (the native mirror of our psycopg2 path).
2. **ADBC** — the `adbc_driver_postgresql` C driver via `adbc_driver_manager`;
   reads the postgres wire straight into Arrow buffers (no per-cell objects).
3. **connectorx** — the native-Rust connectorx source → Arrow.

It prints the best of 3 timed runs (after a warmup) for each.

## Build & run
```bash
cargo run --release -- \
  --host=localhost --port=5432 --user=postgres --password=secret \
  --database=mydb \
  --sql="SELECT id, path FROM catalog_files WHERE table_id = '...'"
```

Args (`--key=value` or `--key value`):
- `--host` (default `localhost`), `--port` (5432), `--user` (postgres),
  `--password` (none), `--database` / `--dbname` (test_db)
- `--sql` the query (default: the `perf_files` wide query)
- `--adbc-driver` path to `libadbc_driver_postgresql.so`
  (default: the one bundled in `/workspace/venv-fedq`)
- `-h` / `--help`

## Example (11,041-row test table `perf_files`)
```
WIDE  (7 cols): rust-postgres+arrow 14.7 ms | adbc 6.6 ms | connectorx 14.6 ms
NARROW (id):    rust-postgres+arrow  3.7 ms | adbc 1.8 ms | connectorx  8.9 ms
```
ADBC is fastest in both (and matches Python ADBC ~7.4/2.0 ms — same C driver).
connectorx has a high fixed per-query cost, so it loses on small results.

## Notes
- All paths unify on Arrow 54.
- Native path supports int2/4/8, float4/8, bool, uuid, timestamp[tz], date,
  text/varchar/bpchar/name. Cast anything else to text in the SQL.
