# Evaluation Runbook

## Purpose

This document explains how official evaluation is supposed to run operationally.

It answers:
- how an official evaluation starts
- what inputs are required
- what outputs must exist
- what must be checked before promotion

## 1. Preconditions

Before running official evaluation:

- relevant canonical docs are frozen for the run
- data snapshot id is known
- reward version is known
- split version is known
- candidate artifacts are registered
- champion artifact is known if a comparison is expected

## 2. Required inputs

Every official evaluation run must know:

- data snapshot id
- split policy version
- reward version
- code commit/hash
- config hash
- candidate artifact id(s)
- current champion artifact id if applicable

## 3. Official flow

### Step 1 — Build evaluation surface
- use time-ordered split
- apply purge/embargo if overlap exists
- persist fold boundaries
- persist split artifact metadata
- consume persisted walk-forward folds for candidate selection rather than leaving them metadata-only
- keep final untouched test outside candidate selection and epoch selection

### Step 2 — Evaluate candidate policies
- if candidate search is active, rank candidate specs on walk-forward fold evidence first
- after a candidate spec is chosen, allow canonical train refit plus canonical validation-only epoch selection
- evaluate candidate artifacts on the canonical evaluation surface
- ensure reward logic matches declared reward version
- ensure post-cost objective is computed

### Step 3 — Compare against champion
- compare challenger and champion on the same surface
- do not compare across incomparable surfaces
- persist the comparison result as a first-class registry-backed comparison report
- `quantlab-ml compare-policies` is the official CLI surface for this linkage

### Step 4 — Produce reports
Required outputs:
- evaluation report
- comparison report if champion exists
- artifact linkage
- search-budget summary
- failure notes if evaluation fails

### Step 5 — Paper/sim gate
If candidate passes evaluation:
- produce paper/sim evidence and record it as a first-class registry-linked evidence record
- link the paper/sim evidence to the evaluation report and comparison report if one exists
- if a current champion exists, `quantlab-ml record-paper-sim` requires a valid `comparison_report_id`
- do not promote directly from offline evaluation alone

## 4. Failure handling

If official evaluation fails:
- do not promote
- record failure reason
- record whether failure is:
  - split issue
  - leakage issue
  - reward issue
  - artifact issue
  - runtime compatibility issue
  - economic issue

## 5. Required outputs

A completed official evaluation should leave behind:
- evaluation report
- comparison report
- paper/sim evidence record once paper/sim is completed
- registry updates
- linked artifact ids
- search-budget summary
- reproducibility metadata

## 6. Promotion relation

Evaluation success alone does not imply promotion.

Promotion additionally requires:
- champion superiority on the same surface
- complete artifact record
- paper/sim evidence
- acceptable-tolerance reproducibility
