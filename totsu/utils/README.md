# Totsu Internal Utilities

This directory contains the internal algorithmic and diagnostic engines that power Totsu.

While the main project focuses on structural diagnosis and visualization,
these utilities implement the mathematical foundations.

---

# Contents

## 1. SimplexSolver

An educational yet robust implementation of the Simplex algorithm.

### Features

* Two-phase Simplex (Phase I / Phase II)
* Slack and artificial variable handling
* Pivot rules (including anti-cycling strategies)
* Degeneracy detection
* Basis tracking and snapshot support
* Integration with Pyomo models

### Design Goals

* Transparent tableau representation
* Debuggable pivot mechanics
* Educational clarity without sacrificing robustness

This solver is designed to expose structural behavior rather than hide it behind a black-box interface.

---

## 2. Tableau

Encapsulates tableau operations used by the SimplexSolver.

### Responsibilities

* Pivot operations
* Basis updates
* Reduced cost inspection
* Feasibility detection
* Objective progression tracking

The Tableau class enables:

* Visualization of pivot paths
* Debugging degeneracy
* Structural inspection of LP geometry

Tableau Visualization
![Tableau Visualization](../../resources/tableau_vis.gif)

---

## 3. ElasticFeasibilityTool

Implements controlled relaxation of constraints to measure structural tension.

### Capabilities

* Supports `<=`, `>=`, `==`, and ranged constraints
* Configurable objective modes:

  * `violation_only`
  * `original_plus_violation`
  * `keep_original`
* Organizes elastic variables under `model.elastic`
* Provides structured violation summaries

This tool transforms infeasible models into measurable systems.

It is the foundation of Phase 1: Elastic Diagnosis.

---

## 4. Sensitivity & Structural Analysis Utilities

Includes tools for:

* Dual value inspection
* Reduced cost analysis
* Constraint tightness evaluation
* Post-solve structural interpretation

These tools help answer:

* Which constraints are binding?
* Which variables are structurally inactive?
* How sensitive is the objective to perturbations?

Sensitivity Analysis
![Sensitivity Analysis](../../resources/sensitivity_analysis.gif)

---

# Architectural Role

The utilities in this directory serve as:

* The computational core of Totsu
* Educational infrastructure for understanding LP/MILP behavior
* Building blocks for higher-level diagnostic tools

They are intentionally modular to allow:

* Custom solver experimentation
* Alternative pivot strategies
* Integration with visualization tools

---

# Relationship to Project Phases

| Phase                        | Role of utils                              |
| ---------------------------- | ------------------------------------------ |
| Phase 1 – Elastic Diagnosis  | ElasticFeasibilityTool + Simplex internals |
| Phase 2 – MILP Visualization | Tableau + branch-and-bound instrumentation |
| Phase 3 – Dual Stacking      | Sensitivity + structural metadata          |

---

# Intended Audience

This directory is primarily for:

* Advanced users
* Researchers
* Contributors
* Those interested in solver internals

If you are looking for high-level usage examples, see:

```
totsu/examples/
```

---

# Notes

These utilities prioritize clarity and structural insight over raw performance.

They are designed to make optimization behavior visible and interpretable.
