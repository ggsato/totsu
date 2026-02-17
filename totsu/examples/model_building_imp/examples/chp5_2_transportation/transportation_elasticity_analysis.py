import argparse

from .transportation import create_model, capacities, requirements
from .....utils.elastic_feasibility_tool import ElasticFeasibilityTool
from totsu.core.super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError
from pyomo.environ import Objective, SolverFactory


def _solve_with_selected_solver(solver_name, solver, model, logfile):
    if solver_name == "glpk":
        return solver.solve(
            model,
            tee=True,
            keepfiles=True,
            symbolic_solver_labels=True,
            logfile=logfile,
        )
    return solver.solve(model)


def main():
    parser = argparse.ArgumentParser(
        description="Run transportation elasticity analysis with selectable solver."
    )
    parser.add_argument(
        "--solver",
        choices=("glpk", "super_simplex"),
        default="glpk",
        help="Solver to use for elastic analysis.",
    )
    parser.add_argument(
        "--include-margin",
        action="store_true",
        help="Include derived margin/tightness diagnostics.",
    )
    parser.add_argument(
        "--margin-scope",
        choices=("non_elastic", "elastic", "all"),
        default="non_elastic",
        help="Margin reporting scope.",
    )
    parser.add_argument(
        "--margin-max-items",
        type=int,
        default=10,
        help="Maximum margin rows to show.",
    )
    args = parser.parse_args()

    # increase the requirements of T2 by 100%
    requirements["T2"] *= 2

    # 1) Create the infeasible model
    model = create_model()

    tool = ElasticFeasibilityTool(default_penalty=1.0, tol=1e-8)

    # 2) Specify which constraints to relax and how to relax them
    #    - ConstraintList / Constraint component / ConstraintData / 文字列名 が使えます
    result = tool.apply(
        model,
        constraints=["supply_constraints"],       # pass a list of constraints to relax
        penalty_map={"supply_constraints": 10.0},  # specify the penalty for each constraint (default is 1.0)
        objective_mode="violation_only",          # recommended
        clone=True,                               # default: True, modify the model in-place if False
        include_margin=args.include_margin,
        margin_scope=args.margin_scope,
        margin_max_items=args.margin_max_items,
    )

    elastic_model = result.model
    print("top-level objectives:", list(elastic_model.component_objects(Objective, active=True)))
    print("block objectives:", list(elastic_model.elastic.component_objects(Objective, active=True)))

    # 3) solve the elastic model and observe the results
    if args.solver == "super_simplex":
        solver = SuperSimplexSolver()
    else:
        solver = SolverFactory("glpk")
    print(f"using solver: {args.solver}")

    print("solving the original model (infeasible)")
    try:
        res = _solve_with_selected_solver(args.solver, solver, model, "solver_original.log")
        if args.solver == "super_simplex":
            print(f"original solve variable count: {len(res)}")
    except InfeasibleProblemError as ex:
        print("Original model infeasible (expected for this scenario):", str(ex))
    except Exception as ex:
        print("An error occurred while solving the original model:", str(ex))

    print("solving the elastic model with objective_mode='violation_only'")
    try:
        _solve_with_selected_solver(args.solver, solver, elastic_model, "solver.log")
        # 供給合計
        total_shipped = sum(
            (elastic_model.x[s, t].value or 0.0)
            for s in elastic_model.S for t in elastic_model.T
        )
        print(f"total shipped: {total_shipped}")

    except Exception as ex:
        print("An error occurred while solving the elastic model:", str(ex))
        elastic_model.display()

    # 4) populate the violation summary and print the results
    ElasticFeasibilityTool.populate_violation_summary(result, tol=1e-8, include_variable_contributions=True)
    if args.include_margin:
        ElasticFeasibilityTool.populate_margin_summary(
            result,
            model=elastic_model,
            tol=1e-8,
            scope=args.margin_scope,
            max_items=args.margin_max_items,
        )

    print("total_violation_cost =", result.total_violation_cost)
    print("objective_mode =", result.objective_mode)
    print("active_objective_value =", result.active_objective_value)
    print("original_objective_value =", result.original_objective_value)
    print("violation_objective_value =", result.violation_objective_value)
    print("combined_objective_value =", result.combined_objective_value)
    for row in result.violation_breakdown[:10]:
        print(row)
    if args.include_margin:
        print("top tight constraints (by margin):")
        for row in result.margin_summary[: args.margin_max_items]:
            print(row)

    # 5) optional: run original_plus_violation mode
    print("building elastic model with objective_mode='original_plus_violation'")
    try:
        result_plus = tool.apply(
            model,
            constraints=["supply_constraints"],
            penalty_map={"supply_constraints": 10.0},
            objective_mode="original_plus_violation",
            original_objective_weight=1.0,
            clone=True,
            include_margin=args.include_margin,
            margin_scope=args.margin_scope,
            margin_max_items=args.margin_max_items,
        )
        elastic_model_plus = result_plus.model
        print("solving the elastic model with objective_mode='original_plus_violation'")
        _solve_with_selected_solver(
            args.solver, solver, elastic_model_plus, "solver_original_plus.log"
        )
        ElasticFeasibilityTool.populate_violation_summary(
            result_plus, tol=1e-8, include_variable_contributions=True
        )
        if args.include_margin:
            ElasticFeasibilityTool.populate_margin_summary(
                result_plus,
                model=elastic_model_plus,
                tol=1e-8,
                scope=args.margin_scope,
                max_items=args.margin_max_items,
            )
        print("original_plus total_violation_cost =", result_plus.total_violation_cost)
        print("objective_mode =", result_plus.objective_mode)
        print("active_objective_value =", result_plus.active_objective_value)
        print("original_objective_value =", result_plus.original_objective_value)
        print("violation_objective_value =", result_plus.violation_objective_value)
        print("combined_objective_value =", result_plus.combined_objective_value)
        for row in result_plus.violation_breakdown[:10]:
            print(row)
        if args.include_margin:
            print("original_plus top tight constraints (by margin):")
            for row in result_plus.margin_summary[: args.margin_max_items]:
                print(row)
    except ValueError as ex:
        print("original_plus_violation validation failed:", str(ex))
    except Exception as ex:
        print("An error occurred while solving original_plus_violation model:", str(ex))

if __name__ == "__main__":
    main()
