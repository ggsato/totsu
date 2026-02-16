# Try It With Your Own Model

If you already use Pyomo with any solver available in your environment, you can run Totsu infeasibility analysis directly on your model.

No extra solver installation is required when you already have one of these available to Pyomo: HiGHS, CBC, or GLPK.

## Minimal Example

```python
from totsu.elastic import analyze_infeasibility

result = analyze_infeasibility(model)
result.print_summary()
```

## What The Output Means

- `solver_name`: which solver was used (`auto` picks the first available in order `highs -> cbc -> glpk`).
- `is_feasible_original`: whether your original model solved as feasible.
- `is_feasible_elastic`: whether the elasticized model solved after allowing controlled relaxations.
- `top_relaxations`: the highest-impact constraint relaxations (deviation, penalty, cost).
- `print_summary()`: compact terminal view; `to_dict()` gives structured output for scripts or notebooks.
