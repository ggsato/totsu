### *What a Generic Elastic-Feasibility Tool Does (Independent of Any Domain)*

This document describes the **generic functionality** of an elastic-feasibility tool in mathematical optimization.
It operates purely on *structure*, *variables*, and *constraints* - not on domain semantics.

---

# **1. Purpose**

A generic elasticity tool automatically transforms **hard constraints** into **soft constraints** by adding:

* **Aux variables** (unpenalized, internal balancing variables)
* **Violation variables** (penalized, measured structural break)
* **Penalty terms** in the objective

This lets a model stay **feasible**, even if the original constraints are impossible to satisfy, while still guiding the solver to **minimize violation**.

---

# **2. Terminology (Public Semantics)**

Public-facing outputs and docs should use:

* **violation**: non-negative constraint violation amount
* **margin**: non-negative satisfied-side slack (feasible-side headroom)
* **residual** (optional/internal): signed structural difference

For a constraint with body `a·x` and bound/right-hand side `b`, this document uses the unified residual convention `residual(x) = rhs - body`.

## LE constraint: `a·x <= b`

* `violation(x) = max(0, a·x - b)`
* `margin(x) = max(0, b - a·x)`
* `residual(x) = b - a·x`

## GE constraint: `a·x >= b`

* `violation(x) = max(0, b - a·x)`
* `margin(x) = max(0, a·x - b)`
* `residual(x) = b - a·x`

## EQ constraint: `a·x = b`

* `violation(x) = |a·x - b|`
* `margin(x)`: not directionally defined for equality; use `None` in public reports (or `0` only when a numeric placeholder is required)
* `residual(x) = b - a·x` (follows the same `rhs - body` convention)

Terminology note:

* The Elastic tool may introduce internal helpers during transformation.
* Those helpers are **not** public "margin" values.
* Avoid calling those helpers "slack" in public Elastic docs, because Pyomo uses "slack" broadly for generic bound-distance concepts.

---

# **Why the Tool Relaxed Demand (and How to Control It)**

When all constraints are elastic, the solver chooses the **cheapest** feasibility repair under the assigned penalties.  
If demand shortfall is cheaper than adding supply, it may relax demand first.

To get realistic repairs, explicitly define:

* Which constraints are elastic targets
* Penalties (weights/costs) by constraint group and index

In short: the tool supplies mechanics, while you define policy.

---

# **3. Inputs**

The generic tool receives:

* A Pyomo model (Concrete or Abstract)
* A list or selection of constraints to be elastified
* Penalty coefficients (optional)
* A policy for how violations are costed

Nothing in this step depends on domain knowledge.

---

# **4. Transformations Performed Automatically**

## ✔ **Step 1 - Detect constraint types**

For each constraint:

* `expr <= rhs` -> LE type
* `expr >= rhs` -> GE type
* `expr = rhs` -> EQ type

The tool does not interpret *what the constraint means* - only its algebraic form.

---

## ✔ **Step 2 - Introduce aux + violation variables**

| Original Type | Variables Added | Meaning |
| ------------- | --------------- | ------- |
| `<=` | Aux `aux >= 0`, Violation `viol >= 0` | `aux` balances equality form; `viol` measures upper-bound break |
| `>=` | Aux `aux >= 0`, Violation `viol >= 0` | `aux` balances equality form; `viol` measures lower-bound break |
| `=` | `viol_pos >= 0`, `viol_neg >= 0` | Two-sided equality violation decomposition |

### Why Elastic has aux variables

The aux variable is a modeling artifact used to convert inequalities to equalities while keeping violation variables one-sided and non-negative.
`aux` does not represent satisfied-side headroom (margin); it is purely an algebraic balancing artifact.

* `aux` is internal and unpenalized.
* `viol*` variables are penalized and reported as violations.
* `margin` is not an Elastic decision variable; it is computed after solving from the original constraint and solution `x*`.

### Example conversion (canonical pattern)

```
LE: body <= ub   ->   body + aux - viol = ub
GE: body >= lb   ->   body - aux + viol = lb
EQ: body = rhs   ->   body + viol_pos - viol_neg = rhs
```

Only `viol*` terms are penalized and reported as violation; `aux` is not.

The tool ensures:

* All transformed constraints are represented as equalities
* Violation variables are non-negative

---

## ✔ **Step 3 - Add penalties to the objective**

Generic form:

```
min  f(x)  +  Σ (penalty_i · violation_i)
```

The tool:

* Does **not** decide the penalty size
* Allows linear or weighted penalties
* Can apply L1, L2, or more complex norms (optional)

---

## ✔ **Step 4 - Provide a unified report**

The output includes:

* A solvable, always-feasible model
* A violation summary for each elastified constraint
* A mapping from reported violations -> original constraints

This ensures interpretability without knowing domain semantics.

---

# **5. Outputs Provided by the Tool**

The generic tool guarantees:

1. **Feasible model** even if original constraints conflict
2. **Penalty decomposition** for each constraint
3. **Violation diagnostics** (`violation`; optional derived `margin_amount`)
4. **Reconstructable original constraints**
5. Completely domain-agnostic logs and structure

---

# **6. What It *Does Not* Do**

The tool deliberately **avoids domain logic**:

* Does *not* decide which constraints should be elastic
* Does *not* assign meaningful penalties
* Does *not* interpret meaning (e.g., "workload" or "flex house")
* Does *not* ensure correctness of domain semantics

Those belong to domain experts and are described in the second file.

---

# **7. Generic Output Contract (No Domain Metadata Required)**

The generic API/CLI reports each top relaxation with:

* Fully-qualified Pyomo constraint name (`constraint_name`)
* Constraint index (`index`)
* Violation amount (`violation`)
* Violation cost (`cost`)
* Structural sense/bound metadata (`sense`, `bound`)
* Optional derived margin amount (`margin_amount`)

This keeps output concise and interpretable even when only model structure is known.

---

# **8. What Needs Domain Metadata**

To make reports business-friendly, domain metadata can be layered on top:

* Constraint naming conventions in the model (`client_demand`, `worker_daily`, etc.)
* A `pretty_name(constraint_data) -> str` mapper for human labels

Example:

* Raw: `demand_con[D2]`
* Pretty: `Demand requirement at destination D2`

The elastic mechanics remain generic; only label rendering uses domain metadata.

---

## Next: Operational Patterns / Domain Guidance

For policy patterns and assignment-specific design choices, continue to [docs/elastic/ELASTIC_TOOL_DOMAIN_EXPERTS.md](ELASTIC_TOOL_DOMAIN_EXPERTS.md).

Worked example entry point: [totsu/examples/model_building_imp/examples/chp5_2_transportation/transportation_elasticity_analysis.py](../../totsu/examples/model_building_imp/examples/chp5_2_transportation/transportation_elasticity_analysis.py)
