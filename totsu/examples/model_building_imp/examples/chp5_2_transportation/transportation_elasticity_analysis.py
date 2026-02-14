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
        constraints=["demand_constraints"],       # pass a list of constraints to relax
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
            elastic_model,
            tee=True,                
            keepfiles=True,          # keep .lp / .mps
            symbolic_solver_labels=True,
            logfile="solver.log",    # keep log file
        )
    except InfeasibleProblemError as ex:
        print("The original model is infeasible as expected.")

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
        # 未達（posdev）合計（変数名はdisplayの通り）
        print("total posdev:",
            sum(getattr(elastic_model.elastic, v).value
                for v in [
                    "elastic_dev_demand_constraints_1_posdev_0",
                    "elastic_dev_demand_constraints_2_posdev_1",
                    "elastic_dev_demand_constraints_3_posdev_2",
                    "elastic_dev_demand_constraints_4_posdev_3",
                ]))

    except InfeasibleProblemError as ex:
        print("The elastic model is infeasible.")
        elastic_model.display()
        return

    # 4) populate the violation summary and print the results
    ElasticFeasibilityTool.populate_violation_summary(result, tol=1e-8)

    print("total_violation_cost =", result.total_violation_cost)
    for row in result.violation_breakdown[:10]:
        print(row)

if __name__ == "__main__":
    main()
