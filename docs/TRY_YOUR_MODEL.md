# Try It With Your Own Model

If you already use Pyomo with any solver available in your environment, you can run Totsu infeasibility analysis directly on your model.

No extra solver installation is required when you already have one of these available to Pyomo: HiGHS, CBC, or GLPK.

## Minimal Example

```python
from totsu.elastic import analyze_infeasibility

result = analyze_infeasibility(model)
result.print_summary()
```

## Optional: Friendlier Names In Output

```python
from totsu.elastic import analyze_infeasibility

def pretty_name(con):
    comp = con.parent_component().name
    idx = con.index()
    return f"{comp}{idx}"

result = analyze_infeasibility(
    model,
    solver="auto",
    violation_only=False,
    default_penalty=1000.0,  # important for original_plus_violation
    max_items=10,
    pretty_name=pretty_name,  # optional
)
result.print_summary()
```

`violation_only=False` (original + violation) combines business cost and relaxation cost.
Set `default_penalty` large enough so violations are not cheaper than normal operation.

## What The Output Means

- `solver_name`: which solver was used (`auto` picks the first available in order `highs -> cbc -> glpk`).
- `is_feasible_original`: whether your original model solved as feasible.
- `is_feasible_elastic`: whether the elasticized model solved after allowing controlled relaxations.
- `top_relaxations`: highest-impact relaxations with `constraint_name`, `index`, `deviation`, `cost`, and `direction`.
  - `direction` is generic guidance based on constraint type:
  - upper bound: `relax upper bound by +...`
  - lower bound: `relax lower bound by -...`
  - equality: `relax by ±...`
- `pretty_name` (optional): when provided, shows a domain-friendly label next to raw constraint names.
- `print_summary()`: compact terminal view; `to_dict()` gives structured output for scripts or notebooks.

## Next step: Make the repair policy explicit

By default, if all constraints are elastic targets, the solver picks the cheapest feasibility repair. That can relax demand instead of supply unless you define a policy for what can move and how much each violation should cost.

Continue to [docs/elastic/ELASTIC_TOOL_GENERIC.md](elastic/ELASTIC_TOOL_GENERIC.md).
