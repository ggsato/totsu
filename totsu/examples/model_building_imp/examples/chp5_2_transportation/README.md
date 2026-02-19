# Transportation Example (Elastic Feasibility Demonstration)

This example demonstrates how Totsu’s Elastic mechanism works on a classical transportation problem.

It shows:

1. Original feasible solution
2. How infeasibility is created
3. Public API behavior (all constraints elastic)
4. Tool-level behavior (policy-controlled elasticity)
5. Clear violation / margin explanations in table form

---

# 1. Original Transportation Model (Feasible Case)

Objective value:

    25919.00

Shipment table:

|    | T1 | T2 | T3 | T4 |
|----|----|----|----|----|
| S1 | 0  | 0  | 39 | 87 |
| S2 | 56 | 0  | 0  | 0  |
| S3 | 6  | 83 | 0  | 4  |

All constraints satisfied.

Constraint diagnostics:

| Constraint       | Bound | Actual | Violation | Margin     |
|-----------------|-------|--------|----------|------------|
| Supply S1 ≤ 135 | 135   | 126    | 0        | 135 − 126  |
| Supply S2 ≤ 56 | 56   | 56     | 0        | 0 (tight)   |
| Supply S3 ≤ 93 | 93   | 93    | 0        | 0 (tight)  |
| Demand T1 = 62 | 62   | 62     | 0        | 0 (tight)   |
| Demand T2 = 83 | 83   | 83     | 0        | 0 (tight)  |
| Demand T3 = 39 | 39   | 39     | 0        | 0 (tight)  |

This is the normal feasible LP world.

---

# 2. Making the Model Infeasible

We increase total demand beyond total supply.

```
T2 demand = 83 * 2 = 166
```

Original solver output:

    LP HAS NO PRIMAL FEASIBLE SOLUTION

Now:

- Original model feasible: False
- No shipment table available

---

# 3. Public API Result (All Constraints Elastic, TRY_YOUR_MODEL Path)

File:
    transportation_elastic_api.py

Minimal usage:

``` python
from totsu.elastic import analyze_infeasibility

result = analyze_infeasibility(
    model,
    violation_only=False,
    default_penalty=1000.0,
)
result.print_summary()
```

This corresponds to the documented public entry point in
`TRY_YOUR_MODEL.md`.

Using:

    analyze_infeasibility(model, violation_only=False)

Example result:

    Solver: glpk
    Original model feasible: False
    Elastic model feasible: True
    Top relaxations:
      - demand_constraints[2]: violation=73, cost=73000
      - demand_constraints[1]: violation=1, cost=1000

Elastic Shipment Table (API):

|    | T1 | T2 | T3 | T4 |
|----|----|----|----|----|
| S1 | 5  | 0  | 39 | 91 |
| S2 | 56 | 0  | 0  | 0  |
| S3 | 0  | 93 | 0  | 0  |

Top relaxations (API):

| Constraint            | Violation | Cost  |
|----------------------|-----------|-------|
| demand_constraints[2] | 73        | 73000 |
| demand_constraints[1] | 1         | 1000  |

Violations:

| Constraint      | Required | Actual    | Violation | Meaning         |
|----------------|----------|-----------|----------|-----------------|
| Demand T2 = 166 | 166      | 166 − 73  | 73       | 73 units unmet  |
| Demand T1 = 62 | 62      | 62 − 1   | 1        | 1 unit unmet    |

Interpretation:

- Demand constraints were violated.
- Because all constraints are elastic and demand penalties are uniform, the solver may choose to violate demand instead of supply depending on cost structure.

API = fast structural repair, no policy control.

---

# 4. Tool-Level Result (Generic Elastic Tool)

File:

    transportation_elasticity_analysis.py

This script allows:

-   Explicit objective mode control
-   Selection of elastic constraint targets
-   Custom penalty configuration
-   Detailed violation diagnostics
-   Variable contribution analysis

This corresponds to the generic elastic transformation described in
`ELASTIC_TOOL_GENERIC.md`.

## violation_only

    objective value = 740.00
    total_violation_cost = 740.0
    original_objective_value = None

Elastic Shipment Table (Tool, violation_only):

|    | T1 | T2  | T3 | T4 |
|----|----|-----|----|----|
| S1 | 62 | 0   | 39 | 34 |
| S2 | 0  | 56  | 0  | 0  |
| S3 | 0  | 110 | 0  | 57 |

Total violation cost: 740

Violations (Tool, violation_only):

| Constraint       | Bound | Actual | Violation | Margin |
|-----------------|-------|--------|----------|--------|
| Supply S3 ≤ 93  | 93    | 167    | 74       | 0      |

Interpretation:

- Only supply constraints are allowed to violate.
- Demand constraints are kept hard in this configuration.
- Shipping cost is ignored.
- The solver purely minimizes total violation.
- Result may be structurally valid but economically unrealistic.

## original_plus_violation

    objective value = 33990.00
    original_objective_value = 33200.0
    violation_objective_value = 790.0
    combined_objective_value = 33990.0

Elastic Shipment Table (Tool, original_plus_violation):

|    | T1 | T2  | T3 | T4 |
|----|----|-----|----|----|
| S1 | 0  | 0   | 39 | 91 |
| S2 | 62 | 0   | 0  | 0  |
| S3 | 0  | 166 | 0  | 0  |

Violations (Tool, original_plus_violation):

| Constraint       | Bound | Actual | Violation | Margin |
|-----------------|-------|--------|----------|--------|
| Supply S3 ≤ 93  | 93    | 166    | 73       | 0      |
| Supply S2 ≤ 56  | 56    | 62     | 6        | 0      |

Interpretation:

- Demand constraints satisfied
- Only supply exceeded
- Shipping cost influences routing
- Violations minimized while respecting business objective

Tool = policy-driven repair planning.


---

# 5. Key Difference

| Aspect                  | Public API | Elastic Tool              |
|------------------------|------------|---------------------------|
| Which constraints elastic? | All     | Selected                  |
| Demand can be violated?    | Yes     | No (if configured hard)   |
| Policy control             | No      | Yes                       |
| Intended use               | Quick diagnosis | Real repair planning |

---

# ⚠️ Important Practical Tip: Not All Constraints Should Be Elastic

When using the public API:

``` python
result = analyze_infeasibility(model, violation_only=False)
```

All constraints are treated as elastic targets by default.

That means:

-   Supply constraints can be violated
-   Demand constraints can also be violated
-   The solver chooses the cheapest repair under the penalty structure

------------------------------------------------------------------------

## Why This Can Be Problematic

In the transportation example:

-   Violating a supply constraint may represent overtime production or
    emergency sourcing.
-   Violating a demand constraint means not satisfying customer demand.

In many real systems, demand is not under your control.

However, with the public API, the solver may decide:

"Relax demand --- it's cheaper."

This may be mathematically valid, but operationally meaningless.

------------------------------------------------------------------------

# Why Elasticity Analysis Exists

The tool-level script allows you to:

-   Select only supply constraints as elastic targets
-   Keep demand constraints hard
-   Assign different penalties per constraint group
-   Control repair policy explicitly

The tool provides mechanics.\
You define policy.

------------------------------------------------------------------------

# Practical Modeling Principle

In real applications:

-   Ask: Which constraints represent controllable flexibility?
-   Only those should be elastic.
-   Everything else should remain hard.

Elasticity without policy control can produce mathematically valid, but
operationally unrealistic repairs.

------------------------------------------------------------------------

# 6. Regression Testing Philosophy

For regression, we do not test exact numbers.

Instead, we confirm:

-   Original model is infeasible.
-   Elastic model is feasible.
-   `objective_mode = original_plus_violation`
-   Violation diagnostics are present.
-   `combined_objective_value = original + violation`

We test the contract, not solver numerics.

------------------------------------------------------------------------

This example demonstrates the difference between:

-   Measuring infeasibility
-   Designing a repair policy
-   Computing a realistic recovery plan

Elastic is not about breaking constraints.
It is about making trade-offs visible.

That difference is the core value of Totsu Elastic.
