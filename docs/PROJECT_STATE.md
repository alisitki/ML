# Project State

## Purpose

This is the short source of truth for what current HEAD actually is.

Use it to answer:

- what QuantLab ultimately aims to become
- what the repo materially implements today
- whether the offline side is professionally closed
- what is blocking the repo before live/runtime work becomes the main focus

Detailed historical narrative belongs in `docs/DECISIONS.md` and `docs/history/`.

---

## Ultimate goal

QuantLab aims to become an end-to-end multi-exchange futures ML trading system:

- websocket ingestion
- canonical exchange-aware state
- offline training and evaluation
- runtime inference
- thin-executor handoff
- commercialization gates toward live capital deployment

---

## Current phase

QuantLab is in late Phase 1 hardening.

The repo materially implements the canonical and offline foundation. It does not yet materially implement the live-operating half. Phase 2 remains planned next-phase work, not current implemented reality and not the main focus while broader offline-closure evidence remains partial.

---

## Current verdict

QuantLab is `offline operational but not professionally closed`.

Why:

- current HEAD contains a real offline engine with trajectory build, train, evaluate, score, export, and registry flows
- current HEAD contains authority-aware continuity audit semantics plus repo-tracked closeout records for the temporary continuity windows
- current HEAD now also carries a repo-tracked minimum offline-closure evidence pack that indexes one inspected-scope continuity audit, a same-surface current-head retained-run comparison, and one distinct external retained surface on `2026-01-26`
- current HEAD now also carries a fresh authoritative controlled rerun at `/workspace/runs/ql016-ql004-authoritative-20260418` with a confirmed external active registry root
- current HEAD now carries decided continuity closeout records that retire the NumPy and legacy compat windows on that authoritative scope
- current HEAD does not include repo-tracked QL-021 retained bundles under `outputs/`; any such bundle is external retained evidence until its provenance and authority are confirmed
- current HEAD now also carries a retained same-root `QL-031` proof run at `outputs/ql031-same-root-proof-20260419`; that run produced four scored same-surface finalists, one linked paper/sim record, and a rejected promotion decision on `economics.post_cost_positive`
- current HEAD can now persist registry-backed comparison reports and reusable offline evidence-pack summaries, but those surfaces do not by themselves prove broader offline closure
- offline closure is still incomplete because the retained same-root `2026-01-25` surface still has no promotable champion and therefore no valid same-root comparison-report chain, even though the retained proof set is now broader than the earlier single-surface baseline

---

## Current implemented strengths

- canonical exchange-aware market-data and observation semantics
- offline trajectory building
- walk-forward training and evaluation discipline
- artifact export and registry discipline
- registry-backed comparison-report persistence and reusable offline evidence-pack summaries
- runtime-facing contracts and thin-executor boundary definitions
- governance, runbook, and repo-tracked closeout-record discipline around retained proof surfaces

---

## Current missing layers

- production websocket ingestion across the active venue scope
- online state / feature service
- replay-vs-live parity tooling over live-style inputs
- selector runtime daemon
- thin executor integration and live control loop
- shadow/paper operating loop
- system-generated commercialization evidence above the offline gate

These missing layers are planned later-phase work, not current defects by default.

---

## Current focus

- keep repo-truth docs aligned with current HEAD rather than planned target-state architecture
- keep `quantlab-ml audit-continuity` authority-aware so inspected-scope truth does not read as authoritative closeout truth
- keep continuity-authority discovery conditional so repo-local retained bundles never read like eligible external authority candidates
- keep repo-tracked continuity closeout decisions explicit and keep retained bundles classified as non-authoritative control surfaces only
- treat historical local authority discovery as closed in this workspace; future continuity closure should prefer fresh external controlled reruns or already-present concrete external roots
- run `QL-031` broader offline evidence expansion as the single active next batch in this workspace
- define explicit offline-closure criteria and continuity-audit procedure
- leave evidence-dependent items visible instead of writing optimistic closure language

---

## Blocked before live-path focus

- broader multi-window evidence remains partial; the retained proof set now includes the same-root `2026-01-25` run plus the distinct `2026-01-26` retained surface, but it is still not enough to move offline closure to `PASS`
- the retained same-root `2026-01-25` run still has no current champion because the promotion gate rejected the selected finalist on `economics.post_cost_positive`; as a result, same-root comparison-report linkage is still blocked by `no_current_champion`
- commercialization-grade evidence packaging still depends on broader offline proof than the single authoritative continuity rerun

Until those are explicit, Phase 2 is still planned next work but not the main execution focus.

---

## Not started / not main focus yet

The following remain visible but are not the current main focus:

- websocket ingestion services
- online feature/state service
- replay-vs-live parity harnesses over live-style inputs
- selector runtime
- thin executor operating loop
- shadow/paper operation
- live-path observability and recovery evidence

---

## Current interpretation notes

- The default configured registry root lives under ignored `outputs/registry` and is not repo-tracked on current HEAD.
- Current HEAD does not include repo-tracked QL-021 bundles under `outputs/`. If external retained QL-021 bundles exist, they are external retained evidence rather than current-head repo-tracked proof until provenance and authority are attached explicitly.
- A relocation-safe external retained bundle may prove `clear_in_inspected_scope` via registry-local fallback, but that still does not make it authoritative evidence.
- If an external active registry root cannot be confirmed, retained bundles remain external retained evidence only and must not be promoted to authoritative evidence.
- Repo-local `outputs/.../registry` candidates may appear in discovery summaries as `retained_bundle_only`, but they are never eligible external authority candidates.
- Historical authoritative continuity roots are unavailable in this workspace baseline as a source of historical authority; local discovery remains closed as a main task in this workspace.
- Fresh authoritative evidence has now been produced by a controlled rerun at `/workspace/runs/ql016-ql004-authoritative-20260418`.
- The repo-tracked continuity closeout records are now decision records for retired windows. They still do not, by themselves, prove live readiness.
- The repo-tracked minimum offline-closure evidence pack now includes one distinct external retained surface on `2026-01-26`; it remains evidence of progress rather than evidence that multi-window closure or champion/challenger proof is complete.
- The retained same-root `QL-031` bundle at `outputs/ql031-same-root-proof-20260419` remains `external_retained_evidence`; it sharpens the blocker from generic missing linkage to an exact failure mode: every evaluated finalist on the retained `2026-01-25` surface remained post-cost negative, so promotion was rejected and no current champion was created.
- Comparison-report and offline evidence-pack tooling exist on current HEAD; neither one, by itself, upgrades offline closure to `PASS`.
- Runtime and executor contracts exist as governance and artifact surfaces. They are not the same thing as a live selector daemon plus executor loop.
- Commercialization gates are defined, but no gate above the offline side is currently operationally evidenced in this repository.
