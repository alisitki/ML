# Architecture

## Core Flow

1. `data` loads raw market events from a source adapter.
2. `trajectories` aligns events onto a canonical time grid and emits
   single-asset trajectories with cross-exchange context.
3. `training` consumes the train split and produces a `PolicyArtifact`.
4. `evaluation` replays the eval split with a shared reward boundary.
5. `scoring` turns the evaluation report into a comparable `PolicyScore`.
6. `registry` stores lineage, coverage, score history, and champion state.
7. `selection` only promotes scored candidates; unscored records stay as plain
   candidates.

The data contract separates the target universe from smoke fixtures. See
`docs/data-contract.md` for exchange-specific stream availability and profile naming.

## Why `trajectories` is its own layer

The repository is not just a model harness. The critical contract is the learning
surface between raw compacted streams and policy discovery. That surface owns:

- Time alignment
- Cross-symbol and cross-exchange context assembly
- Action masks
- Reward timestamping
- Episode segmentation
- Stable tensor-like schema and masks

## V1 Implementation Notes

- The local fixture adapter is the default local verification source.
- The parquet adapter is implemented behind an optional dependency boundary.
- The S3 compact adapter is metadata-first: it reads `compacted/_state.json`, filters
  logical partitions by the dataset contract, then resolves root storage prefixes such
  as `exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet`.
- The trainer is intentionally simple so the repo stabilizes the interfaces first.
- The replay engine validates its supported evaluation boundary instead of treating
  boundary fields as decorative metadata.
