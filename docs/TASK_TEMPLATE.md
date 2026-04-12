# Task Template

Use this template when starting a meaningful Codex task.

## 1. Path classification and execution target

State explicitly:
- task class: `core direction`, `optional experiment`, `temporary compatibility maintenance`, or `forbidden-as-default area`
- intended execution mode: `real training`, `continuity baseline`, or `smoke/debug`
- does this task accidentally grow a temporary path
- does this task make an optional path behave like core
- which temporary compromise narrows, and which one widens
- does this task turn an allowed area into a default
- does this task accidentally assume local CPU / laptop is the long-term execution target
- if the task touches training scale, search budget, data volume, or production observation profiles, should the intended execution target be remote GPU
- explicit non-goal: do not optimize the core path around laptop constraints

## 2. Task restatement

Restate the task in one paragraph.

## 3. Relevant rules

List the relevant:
- constitutional rules
- canonical contract docs
- operational docs

## 4. Main risks

Call out the main risks explicitly:
- leakage
- overfitting
- hidden reduction
- reward drift
- registry drift
- selector/executor boundary drift

## 5. Affected areas

State whether the task affects:
- learning surface
- reward
- split logic
- policy artifact
- registry
- runtime selector
- executor boundary
- docs only

## 6. Minimal plan

Give the smallest plan that solves the task.

## 7. Files likely touched

List the likely files.

## 8. Definition of done

State `done_when` explicitly.

## 9. Unverified assumptions

List assumptions that still need proof.
