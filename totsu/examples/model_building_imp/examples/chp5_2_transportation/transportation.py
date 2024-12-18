from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver

# Define data
suppliers = ["S1", "S2", "S3"]
capacities = {"S1": 135, "S2": 56, "S3": 93}
customers = ["T1", "T2", "T3", "T4"]
requirements = {"T1": 63, "T2": 83, "T3": 39, "T4": 91}
cost = {
    "S1": {"T1": 132, "T2": None, "T3": 97, "T4": 103},
    "S2": {"T1": 85, "T2": 91, "T3": None, "T4": None},
    "S3": {"T1": 106, "T2": 89, "T3": 100, "T4": 98},
}

def create_model():
    # Create a Pyomo model
    model = ConcreteModel()

    # Sets
    model.S = Set(initialize=suppliers)
    model.T = Set(initialize=customers)

    # Parameters
    model.capacities = Param(model.S, initialize=capacities)
    model.requirements = Param(model.T, initialize=requirements)
    model.cost = Param(model.S, model.T, initialize=lambda model, s, t: cost[s][t] if cost[s][t] is not None else float('inf'))

    # Variables
    model.x = Var(model.S, model.T, domain=NonNegativeReals)

    # Objective: Minimize cost
    model.obj = Objective(
        expr=sum(model.x[s, t] * model.cost[s, t] for s in model.S for t in model.T), sense=minimize
    )

    # Constraints: Supply cannot exceed capacity
    model.supply_constraints = ConstraintList()
    for s in model.S:
        model.supply_constraints.add(sum(model.x[s, t] for t in model.T if cost[s][t] is not None) <= model.capacities[s])

    # Constraints: Demand must be satisfied
    model.demand_constraints = ConstraintList()
    for t in model.T:
        model.demand_constraints.add(sum(model.x[s, t] for s in model.S if cost[s][t] is not None) >= model.requirements[t])
    return model

if __name__ == "__main__":
    import sys, traceback
    try:
        print("solving the model")
        model = create_model()
        solver = SuperSimplexSolver()
        solution = solver.solve(model)

        print(f"objective value = {solver.get_current_objective_value()}") # 26030

        for s in suppliers:
            for t in customers:
                print(f"x[{s}][{t}] = {solution[f"x[{s},{t}]"]:.2f}")

    except Exception as ex:
        traceback.print_exception(ex)