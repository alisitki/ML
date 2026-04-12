# Runtime Boundary

## Purpose

This document defines the hard boundary between:
- offline training
- runtime selector / inference
- executor

This boundary is mandatory.

Allowed implementation families inside this boundary do not define the current default or active development priority by themselves.

## 1. Fixed runtime architecture

QuantLab runtime path is:

offline training
-> registry
-> runtime selector / inference
-> thin executor

## 2. Offline training responsibilities

Offline training is responsible for:
- learning policies from historical data
- producing training artifacts
- producing deployment / inference artifacts
- producing evaluation evidence
- registering policy candidates

Offline training is not allowed to become the live executor.

## 3. Runtime selector / inference responsibilities

Runtime selector / inference is responsible for:
- loading inference artifacts
- consuming live or near-live observation context
- scoring or selecting policies
- resolving policy conflicts
- choosing actionable decision candidates
- narrowing the candidate set before execution

Runtime selector may perform:
- policy ranking
- arbitration between policies
- venue arbitration
- candidate filtering
- no-trade decision

Runtime selector must not:
- silently retrain online
- mutate model weights as live learning
- delegate hidden strategy logic to executor

## 4. Executor responsibilities

Executor is responsible for:
- feasibility checks
- balance and constraint checks
- capital allocation
- order submission
- order lifecycle handling

Executor must not:
- interpret arbitrary training payloads
- discover new policy logic
- rank raw policy libraries
- become the hidden selector
- silently learn online

## 5. Capital allocation boundary

Capital allocation belongs to the executor layer or an execution-side allocator.

But capital allocation must consume already-selected decision candidates.
It must not become a hidden policy-discovery engine.

## 6. Policy artifact boundary

Policy artifacts are consumed by runtime selector / inference, not by executor directly.

Executor should consume only execution intent.

## 7. Runtime modes

Two runtime implementation families are acceptable:

### A. Direct model inference
The runtime loads a trained inference model and runs it directly.

### B. Artifact-driven runtime
The runtime loads exported policy artifacts or compiled policy payloads.

Both are acceptable if:
- they use inference artifacts only
- they preserve selector/executor separation

Acceptable implementation families define an allowed boundary, not the current default or active priority.
`docs/PROJECT_STATE.md` and `docs/DECISIONS.md` still determine what should be built now.

## 8. TensorRT / ONNX rule

PyTorch is the training stack.

ONNX and TensorRT are allowed only for runtime inference acceleration.
Allowed does not mean default or current priority.
Future acceleration options do not override the active next task.

They are not the training system.
They are not executor logic.

## 9. Traceability

Every runtime decision should be traceable back to:
- policy id
- artifact id
- snapshot/config version
- selector decision path where possible

## 10. Prohibited runtime drift

Forbidden:
- live retraining in the initial operating model
- executor-side hidden policy ranking
- runtime inference that cannot be traced to registered artifacts
- deployment that bypasses registry and champion/challenger discipline
