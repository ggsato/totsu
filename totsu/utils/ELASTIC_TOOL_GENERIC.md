### *What a Generic Elastic-Feasibility Tool Does (Independent of Any Domain)*

This document describes the **generic functionality** of an elastic-feasibility tool in mathematical optimization.
It operates purely on *structure*, *variables*, and *constraints* — not on domain semantics.

---

# **1. Purpose**

A generic elasticity tool automatically transforms **hard constraints** into **soft constraints** by adding:

* **Slack variables** (positive deviation below a requirement)
* **Excess variables** (positive deviation above a limit)
* **Penalty terms** in the objective

This lets a model stay **feasible**, even if the original constraints are impossible to satisfy, while still guiding the solver to **minimize violation**.

---

# **2. Inputs**

The generic tool receives:

* A Pyomo model (Concrete or Abstract)
* A list or selection of constraints to be elastified
* Penalty coefficients (optional)
* A policy for how violations are costed

Nothing in this step depends on domain knowledge.

---

# **3. Transformations Performed Automatically**

## ✔ **Step 1 — Detect constraint types**

For each constraint:

* `expr ≤ rhs` → LE type
* `expr ≥ rhs` → GE type
* `expr = rhs` → EQ type

The tool does not interpret *what the constraint means* — only its algebraic form.

---

## ✔ **Step 2 — Introduce deviation variables**

| Original Type | Slack/Excess Added                | Meaning                    |
| ------------- | --------------------------------- | -------------------------- |
| `≤`           | Slack `s ≥ 0`                     | Amount the LHS falls short |
| `≥`           | Excess `e ≥ 0`                    | Amount LHS exceeds the RHS |
| `=`           | Slack+Excess or 2-sided deviation | Equality violation         |

### Example conversion:

```
a·x ≤ b   →   a·x + s = b
a·x ≥ b   →   a·x - e = b
a·x = b   →   a·x + s - e = b
```

The tool ensures:

* All constraints become **equalities**
* Deviation variables are **non-negative**

---

## ✔ **Step 3 — Add penalties to the objective**

Generic form:

```
min  f(x)  +  Σ (penalty_i · deviation_i)
```

The tool:

* Does **not** decide the penalty size
* Allows linear or weighted penalties
* Can apply L1, L2, or more complex norms (optional)

---

## ✔ **Step 4 — Provide a unified report**

The output includes:

* A solvable, always-feasible model
* A deviation summary for each elastified constraint
* A mapping from deviations → original constraints

This ensures interpretability without knowing domain semantics.

---

# **4. Outputs Provided by the Tool**

The generic tool guarantees:

1. **Feasible model** even if original constraints conflict
2. **Penalty decomposition** for each constraint
3. **Violation diagnostics** (slack/excess values)
4. **Reconstructable original constraints**
5. Completely domain-agnostic logs and structure

---

# **5. What It *Does Not* Do**

The tool deliberately **avoids domain logic**:

* Does *not* decide which constraints should be elastic
* Does *not* assign meaningful penalties
* Does *not* interpret meaning (e.g., “workload” or “flex house”)
* Does *not* ensure correctness of domain semantics

Those belong to domain experts and are described in the second file.

---

# **6. Generic Output Contract (No Domain Metadata Required)**

The generic API/CLI reports each top relaxation with:

* Fully-qualified Pyomo constraint name (`constraint_name`)
* Constraint index (`index`)
* Deviation magnitude (`deviation`)
* Violation cost (`cost`)
* Direction hint (`direction`)

Direction is structural (derived from bounds), not domain-specific:

* Upper-bound constraints (`<=`): `relax upper bound by +<deviation>`
* Lower-bound constraints (`>=`): `relax lower bound by -<deviation>`
* Equalities (`==`): `relax by ±<deviation>`

This keeps output concise and interpretable even when only model structure is known.

---

# **7. What Needs Domain Metadata**

To make reports business-friendly, domain metadata can be layered on top:

* Constraint naming conventions in the model (`client_demand`, `worker_daily`, etc.)
* A `pretty_name(constraint_data) -> str` mapper for human labels

Example:

* Raw: `demand_con[D2]`
* Pretty: `Demand requirement at destination D2`

The elastic mechanics remain generic; only label rendering uses domain metadata.
