from .transportation import create_model, capacities, requirements
from .....utils.elastic_feasibility_tool import ElasticFeasibilityTool
from totsu.core.super_simplex_solver import SuperSimplexSolver, InfeasibleProblemError
from pyomo.environ import Objective, ConstraintList, SolverFactory

def main():
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
    )

    elastic_model = result.model
    print("top-level objectives:", list(elastic_model.component_objects(Objective, active=True)))
    print("block objectives:", list(elastic_model.elastic.component_objects(Objective, active=True)))

    # 3) solve the elastic model and observe the results
    #solver = SuperSimplexSolver()
    solver = SolverFactory("glpk")

    print("solving the original model (infeasible)")
    try:
        #solver.solve(model)
        res = solver.solve(
            model,
            tee=True,                # display solver output
            keepfiles=True,          # keep .lp / .mps files
            symbolic_solver_labels=True,  
            logfile="solver_original.log",    # keep log file
        )
    except Exception as ex:
        print("An error occurred while solving the original model:", str(ex))

    print("solving the elastic model with objective_mode='violation_only'")
    try:
        #solver.solve(elastic_model)
        res = solver.solve(
            elastic_model,
            tee=True,                
            keepfiles=True,          # keep .lp / .mps
            symbolic_solver_labels=True,  
            logfile="solver.log",    # keep log file
        )
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

    print("total_violation_cost =", result.total_violation_cost)
    print("objective_mode =", result.objective_mode)
    print("active_objective_value =", result.active_objective_value)
    print("original_objective_value =", result.original_objective_value)
    print("violation_objective_value =", result.violation_objective_value)
    print("combined_objective_value =", result.combined_objective_value)
    for row in result.violation_breakdown[:10]:
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
        )
        elastic_model_plus = result_plus.model
        print("solving the elastic model with objective_mode='original_plus_violation'")
        solver.solve(
            elastic_model_plus,
            tee=True,
            keepfiles=True,
            symbolic_solver_labels=True,
            logfile="solver_original_plus.log",
        )
        ElasticFeasibilityTool.populate_violation_summary(
            result_plus, tol=1e-8, include_variable_contributions=True
        )
        print("original_plus total_violation_cost =", result_plus.total_violation_cost)
        print("objective_mode =", result_plus.objective_mode)
        print("active_objective_value =", result_plus.active_objective_value)
        print("original_objective_value =", result_plus.original_objective_value)
        print("violation_objective_value =", result_plus.violation_objective_value)
        print("combined_objective_value =", result_plus.combined_objective_value)
        for row in result_plus.violation_breakdown[:10]:
            print(row)
    except ValueError as ex:
        print("original_plus_violation validation failed:", str(ex))
    except Exception as ex:
        print("An error occurred while solving original_plus_violation model:", str(ex))

if __name__ == "__main__":
    main()
