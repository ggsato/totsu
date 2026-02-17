### How Domain Experts Apply Elastic Feasibility (4-Way Assignment with Time Windows)

This document explains how **domain experts** apply elasticity to a realistic but fully public Pyomo model.

We use a simple yet expressive problem:

> Assign **Workers** to **Clients** on **Days** and **Shifts**,
> respecting capacity, availability windows, and “almost hard” rules.

---

## 1. Problem: 4-Way Assignment with Availability Windows

We assign:

* Workers `w ∈ W`
* Clients `c ∈ C`
* Days `d ∈ D`
* Shifts `s ∈ S`

Binary decision variable:

```python
x[w,c,d,s] = 1  if worker w is assigned to client c on day d, shift s
```

We also have:

* `start[c]` — earliest day client `c` can be served
* `end[c]` — latest day client `c` can be served

This pattern appears in:

* Healthcare visits
* Home services
* Store audit rounds
* Tutoring schedules
* General field service logistics

It’s generic enough to publish and teach, but rich enough to show elastic feasibility.

---

## 2. Base Model (Before Elasticity)

Here is a compact, public base model that can easily become infeasible.

```python
from pyomo.environ import *

def build_model():
    m = ConcreteModel()

    # Sets
    m.W = Set(initialize=["w1", "w2"])
    m.C = Set(initialize=["c1", "c2"])
    m.D = Set(initialize=[1,2,3])
    m.S = Set(initialize=[1,2])

    # Availability windows
    m.start = Param(m.C, initialize={"c1": 1, "c2": 2})
    m.end   = Param(m.C, initialize={"c1": 3, "c2": 3})

    # Forbidden assignment example (public & simple)
    m.forbidden_pairs = Set(within=m.W * m.C,
        initialize=[("w1", "c2")]
    )

    # Decision variable
    m.x = Var(m.W, m.C, m.D, m.S, within=Binary)

    # 1. Client must be served exactly once per day
    m.client_demand = ConstraintList()
    for c in m.C:
        for d in m.D:
            m.client_demand.add(
                sum(m.x[w,c,d,s] for w in m.W for s in m.S) == 1
            )

    # 2. Worker can do at most 1 shift per day
    worker_capacity = {"w1": 1, "w2": 1}
    m.worker_daily = ConstraintList()
    for w in m.W:
        for d in m.D:
            m.worker_daily.add(
                sum(m.x[w,c,d,s] for c in m.C for s in m.S) <= worker_capacity[w]
            )

    # 3. Hard forbidden assignment (before elasticity)
    m.forbidden_constraint = ConstraintList()
    for (w,c) in m.forbidden_pairs:
        for d in m.D:
            for s in m.S:
                m.forbidden_constraint.add(m.x[w,c,d,s] == 0)

    return m
```

At this stage:

* Each client must be served every day
* Each worker has a tight capacity (1 shift/day)
* Some worker–client pairs are forbidden
* Availability windows (`start[c], end[c]`) are *not yet enforced*

If we now try to enforce availability windows as hard rules, it becomes very easy to create **infeasible data**. That’s where elastic feasibility comes in.

---

## 3. Domain Expert’s Role: Decide What Can Bend

The **generic elastic tool** knows how to:

* Add aux + violation variables
* Turn inequalities into equalities
* Add penalty terms to the objective

But it does **not** know:

* Which constraints can safely be relaxed
* How bad different violations are in the real world

That is the **domain expert’s job**.

For this example, a domain expert might say:

1. **Client demand**

   * “We really want to serve each client every day, but if we must miss a day, that’s bad but not impossible.”
   * → Make shortfall elastic, with a large penalty.

2. **Worker capacity**

   * “Overtime is allowed, but should be clearly penalized.”
   * → Make capacity elastic via overtime variables.

3. **Forbidden assignments**

   * “This is practically ‘never’, but in real emergencies we might break the rule.”
   * → Make forbidden rules elastic with *very* high penalty.

4. **Availability windows (start/end)**

   * “Serving outside the requested window is undesirable but better than not serving at all.”
   * → Make early/late service elastic with a penalty between shortfall and overtime.

---

## 4. Elastic Formulation

### 4.1. New Violation Variables

We introduce penalized **violation variables**:

* `s_short[c,d] ≥ 0` — client `c` shortfall violation on day `d`
* `e_ot[w,d] ≥ 0` — overtime violation for worker `w` on day `d`
* `e_forbidden[w,c,d,s] ≥ 0` — forbidden-pair violation for `(w,c)`
* `e_early[c,d] ≥ 0` — early-window violation: serving client `c` **before** `start[c]` on day `d`
* `e_late[c,d] ≥ 0` — late-window violation: serving client `c` **after** `end[c]` on day `d`

Then we give them penalties, e.g.:

```python
penalties = {
    "shortfall": 1000,
    "overtime": 100,
    "forbidden": 5000,
    "early": 800,
    "late": 800,
}
```

This defines a **priority hierarchy**:

1. Try to avoid **forbidden** and **shortfall**
2. Then avoid **early/late**
3. Finally avoid mild **overtime**

---

### 4.2. Elastic Constraints

```python
from pyomo.environ import *

def apply_elasticity(m, penalties):
    # Deviation variables
    m.s_short = Var(m.C, m.D, within=NonNegativeReals)
    m.e_ot = Var(m.W, m.D, within=NonNegativeReals)
    m.e_forbidden = Var(m.W, m.C, m.D, m.S, within=NonNegativeReals)

    m.e_early = Var(m.C, m.D, within=NonNegativeReals)
    m.e_late  = Var(m.C, m.D, within=NonNegativeReals)

    # 1. Elastic client demand
    m.client_demand_elastic = ConstraintList()
    for c in m.C:
        for d in m.D:
            m.client_demand_elastic.add(
                sum(m.x[w,c,d,s] for w in m.W for s in m.S)
                + m.s_short[c,d] == 1
            )

    # 2. Elastic worker capacity
    worker_capacity = {"w1": 1, "w2": 1}
    m.worker_daily_elastic = ConstraintList()
    for w in m.W:
        for d in m.D:
            m.worker_daily_elastic.add(
                sum(m.x[w,c,d,s] for c in m.C for s in m.S)
                - m.e_ot[w,d] <= worker_capacity[w]
            )

    # 3. Elastic forbidden assignments
    m.forbidden_elastic = ConstraintList()
    for (w,c) in m.forbidden_pairs:
        for d in m.D:
            for s in m.S:
                m.forbidden_elastic.add(
                    m.x[w,c,d,s] <= m.e_forbidden[w,c,d,s]
                )

    # 4. Elastic availability windows (start / end)
    m.window_elastic = ConstraintList()
    for c in m.C:
        for d in m.D:
            assigned = sum(m.x[w,c,d,s] for w in m.W for s in m.S)

            # Early: d < start[c]
            if d < m.start[c]:
                m.window_elastic.add(assigned <= m.e_early[c,d])

            # Late: d > end[c]
            if d > m.end[c]:
                m.window_elastic.add(assigned <= m.e_late[c,d])

    # 5. Elastic objective: minimize total violation
    m.obj_elastic = Objective(
        expr =
            penalties["shortfall"] * sum(m.s_short[c,d] for c in m.C for d in m.D)
          + penalties["overtime"] * sum(m.e_ot[w,d] for w in m.W for d in m.D)
          + penalties["forbidden"] * sum(
                m.e_forbidden[w,c,d,s]
                for w in m.W for c in m.C for d in m.D for s in m.S
            )
          + penalties["early"] * sum(m.e_early[c,d] for c in m.C for d in m.D)
          + penalties["late"]  * sum(m.e_late[c,d]  for c in m.C for d in m.D),
        sense=minimize
    )

    return m
```

This is exactly the point where the **generic elastic tool** and **domain expert** meet:

* The tool can systematically add these deviations and objective terms.
* The expert chooses *which constraints* to elasticize and *what weights* to use.

---

## 5. Example Run and Interpretation

If we build and solve the elastic model:

```python
if __name__ == "__main__":
    penalties = {
        "shortfall": 1000,
        "overtime": 100,
        "forbidden": 5000,
        "early": 800,
        "late": 800,
    }

    m = build_model()
    m = apply_elasticity(m, penalties)

    # You can use any MILP solver here.
    # Example: CBC, GLPK, or your own AdvancedBranchAndBoundSolver.
    # solver = SolverFactory("cbc")
    # results = solver.solve(m, tee=True)

    from totsu.core.advanced_branch_and_bound_solver import AdvancedBranchAndBoundSolver
    solver = AdvancedBranchAndBoundSolver()
    solver.solve(m)  # assuming this updates m.x values

    print("=== Solution ===")
    for w in m.W:
        for c in m.C:
            for d in m.D:
                for s in m.S:
                    if m.x[w,c,d,s].value and m.x[w,c,d,s].value > 0.5:
                        print(f"x[{w},{c},d{d},s{s}] = 1")

    print("\n=== Early/Late Violations ===")
    for c in m.C:
        for d in m.D:
            if m.e_early[c,d].value and m.e_early[c,d].value > 1e-6:
                print(f"Early: {c} on day {d} => {m.e_early[c,d].value}")
            if m.e_late[c,d].value and m.e_late[c,d].value > 1e-6:
                print(f"Late: {c} on day {d} => {m.e_late[c,d].value}")
```

A typical optimal solution (as you observed with your `AdvancedBranchAndBoundSolver`) looks like:

```text
=== Solution ===
x[w1,c1,d1,s1] = 1
x[w1,c1,d2,s1] = 1
x[w1,c1,d3,s1] = 1
x[w2,c2,d1,s1] = 1
x[w2,c2,d2,s1] = 1
x[w2,c2,d3,s1] = 1

=== Early/Late Violations ===
Early: c2 on day 1 => 1.0
```

### 5.1. How to Read This

* **Capacity:**

  * Each worker does exactly 1 shift per day → no overtime (`e_ot = 0`).

* **Client demand:**

  * Every client is served every day → no shortfall (`s_short = 0`).

* **Forbidden assignments:**

  * `w1` never serves `c2` → no forbidden violations (`e_forbidden = 0`).

* **Availability windows:**

  * `c1` has window days 1–3 → all assignments inside window.
  * `c2` has window days 2–3 → but is also served on day 1.

    * Day 1 is **before** `start[c2] = 2`, so:

      * `assigned(c2, day1) = 1`
      * Constraint forces `e_early[c2,1] ≥ 1`
      * Penalty: `early_penalty * 1 = 800`

There is **no way** to get total violation cost below 800:

* If we don’t serve `c2` on day 1:

  * `s_short[c2,1] = 1` → shortfall cost 1000 (worse than early).
* Serving `c2` later doesn’t help; they also need day 2 and 3.
* Using a forbidden worker also costs `5000` via `e_forbidden`.

So the elastic model chooses:

* **Serve everyone every day**
* Accept **one early violation** (`c2` on day 1)
* Total cost = 800, which is optimal under the given penalties.

This is exactly the kind of trade-off **elastic feasibility** is meant to model.

---

## 6. Worked Example: Transportation

For a compact transportation case where supply-vs-demand elasticity is intentionally designed, see:

* Module: `python -m totsu.examples.model_building_imp.examples.chp5_2_transportation.transportation_elasticity_analysis`
* Source: [totsu/examples/model_building_imp/examples/chp5_2_transportation/transportation_elasticity_analysis.py](../../totsu/examples/model_building_imp/examples/chp5_2_transportation/transportation_elasticity_analysis.py)

This example shows how penalty and target choices produce a domain-specific repair policy instead of a purely structural one.

Back to generic guidance: [docs/elastic/ELASTIC_TOOL_GENERIC.md](ELASTIC_TOOL_GENERIC.md)

## 7. Summary: Who Does What?

### Generic Elastic Tool

* Adds aux + violation variables (`s_short`, `e_ot`, `e_forbidden`, `e_early`, `e_late`, …)
* Converts inequalities into equalities using internal aux terms while penalizing only violation variables
* Builds the aggregate penalty objective
* Guarantees: *model is always solvable*

### Domain Expert

* Chooses **which** constraints become elastic
* Chooses **penalty weights** and relative importance
* Interprets outputs:

  * “We served `c2` 1 day earlier than requested.”
  * “We had 0 shortfalls, 0 overtime, 0 forbidden violations.”
* Adjusts penalties or data based on business rules

Together, they produce a model that is:

* **Robust to infeasibility**
* **Interpretable in domain language**
* **Safe to expose to non-expert users** (e.g., via a Totsu-powered GUI or course demo)

---

## 8. Naming and `pretty_name` Guidance

Use consistent, domain-meaningful constraint names so diagnostics stay readable:

* Prefer semantic names: `client_demand`, `worker_daily`, `forbidden_constraint`, `window`.
* Keep index order aligned with business keys (for example `(client, day)` everywhere).
* Avoid opaque names like `c1`, `c2`, `rule_17` for user-facing reports.

When using the API, you can add optional display labels with:

```python
analyze_infeasibility(..., pretty_name=pretty_name_fn)
```

`pretty_name_fn` receives Pyomo `ConstraintData` and returns a short string:

```python
def pretty_name_fn(con):
    comp = con.parent_component().name
    idx = con.index()
    if comp == "client_demand":
        c, d = idx
        return f"Client {c} demand on day {d}"
    if comp == "worker_daily":
        w, d = idx
        return f"Worker {w} capacity on day {d}"
    return f"{comp}{idx}"
```

Recommended output pattern:

* Always keep the raw fully-qualified name for traceability.
* Show the pretty label next to it for domain users.
