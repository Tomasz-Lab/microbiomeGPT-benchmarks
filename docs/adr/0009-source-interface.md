# 0009. Source interface

Status: Proposed
Date: 2026-11-16

## Context
Our readers currently call pandas read_csv directly, with no way to handle other formats. This became a problem when we e.g want to migrated our files from CSV to Parquet: every reader depends on one specific file format. To remove this dependency, we need a `Source` interface. The same interface will let us load datasets from other backends, such as a database, without rewriting the loading code later


## Decision

1. Interface. A Source exposes a single method, read().
2. One implementation per backend. Each source (e.g. CsvSource, ParquetSource, SqlSource) implements how to read its own format or backend.
3. No domain knowledge. A source only reads data into memory. It knows nothing about modalities, schemas or sample IDs. It doesn't validate, rename or transform data; that is the job of the modality (or metadata) parsing step.
4. Configuration. All configuration needed to read the data (file path, connection string, query, read options) is passed to the source, and only to the source. Secrets are not stored in config files. They are injected from the environment (e.g. ${oc.env:DB_PASSWORD}).

## Consequences

Positive

- Format independence. Moving from CSV to Parquet, or to a database, only means changing the source in the dataset recipe. Modalities, load_dataset and consumers stay unchanged.
- Easy to test. Any code that consumes a source can be tested with an in-memory fake source, without files or a database.
