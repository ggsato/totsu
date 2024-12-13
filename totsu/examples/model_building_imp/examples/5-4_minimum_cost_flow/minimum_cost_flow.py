from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver
from pyomo.environ import SolverFactory

# Define data
nodes = [i for i in range(8)]
# sources
availabilities = {0:10, 1:15}
# sinks
requirements = {5:-9, 6:-10, 7:-6}
# arc costs
costs = {(i, j): None for i in nodes for j in nodes}
costs[0,2] = 5
costs[1,3] = 4
costs[2,5] = 5
costs[2,3] = 2
costs[2,4] = 6
costs[3,4] = 1
costs[3,7] = 2
costs[4,2] = 4
costs[4,5] = 6
costs[4,6] = 3
costs[7,6] = 4

def create_model():
    # Create a Pyomo model
    model = ConcreteModel()

    # Sets
    model.nodes = Set(initialize=nodes)
    model.arcs = Set(initialize=[(i, j) for (i, j), c in costs.items() if c is not None])

    # Parameters
    model.costs = Param(model.arcs, initialize={k: v for k, v in costs.items() if v is not None})
    model.availabilities = Param(model.nodes, initialize=availabilities, default=0)
    model.requirements = Param(model.nodes, initialize=requirements, default=0)

    # Variables
    model.flow = Var(model.arcs, within=NonNegativeReals)

    # Constraints
    def flow_balance_rule(model, node):
        inflow = sum(model.flow[i, node] for i in model.nodes if (i, node) in model.arcs)
        outflow = sum(model.flow[node, j] for j in model.nodes if (node, j) in model.arcs)
        return inflow +  model.availabilities[node] ==  outflow + (-1 * model.requirements[node])

    model.flow_balance = Constraint(model.nodes, rule=flow_balance_rule)

    # Objective: Minimize the total cost
    def objective_rule(model):
        return summation(model.costs, model.flow)

    model.objective = Objective(rule=objective_rule, sense=minimize)

    return model

if __name__ == "__main__":
    import sys, traceback
    try:
        print("solving the model")
        use_glpk = False
        model = create_model()
        model.pprint()
        if use_glpk:
            solver = SolverFactory("glpk")
        else:
            solver = SuperSimplexSolver()

        solution = solver.solve(model)

        if use_glpk:
            print(f"objective value = {model.objective()}")
            # Print results
            print("Optimal Flows:")
            for (i, j) in model.arcs:
                if model.flow[i, j].value > 0:
                    print(f"Flow from {i} to {j}: {model.flow[i, j].value}")
        else:
            print(f"objective value = {solver.get_current_objective_value()}")     

    except Exception as ex:
        traceback.print_exception(ex)