# Task Template

Use this template when starting a meaningful Codex task.

## 1. Path classification

State explicitly:
- task class: `core direction`, `optional experiment`, `temporary compatibility maintenance`, or `forbidden-as-default area`
- does this task accidentally grow a temporary path
- does this task make an optional path behave like core
- which temporary compromise narrows, and which one widens
- does this task turn an allowed area into a default
- explicit non-goal

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
