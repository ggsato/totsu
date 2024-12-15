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

def create_minimum_model():
    # Create a Pyomo model
    model = ConcreteModel()

    min_nodes = [i for i in range(3)]
    min_costs = {(0, 1): 1, (0, 2): 3, (1, 2): 1} # 0 -> 1 -> 2: 1 + 1 = 2, 0 -> 2: 3
    min_availabilities = {0: 2}
    min_requirements = {2: -2}

    # Sets
    model.nodes = Set(initialize=min_nodes)
    model.arcs = Set(initialize=[(i, j) for (i, j), c in min_costs.items() if c is not None])

    # Parameters
    model.costs = Param(model.arcs, initialize={k: v for k, v in min_costs.items() if v is not None})
    model.availabilities = Param(model.nodes, initialize=min_availabilities, default=0)
    model.requirements = Param(model.nodes, initialize=min_requirements, default=0)

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

def create_minimum_model2():
    # Create a Pyomo model
    model = ConcreteModel()

    min_nodes = [i for i in range(4)]
    min_costs = {(0, 1): 1, (0, 2): 3, (1, 2): 1, (2,3): 1} # 0 -> 1 -> 2: 1 + 1 = 2, 0 -> 2: 3
    min_availabilities = {0: 2}
    min_requirements = {3: -2}

    # Sets
    model.nodes = Set(initialize=min_nodes)
    model.arcs = Set(initialize=[(i, j) for (i, j), c in min_costs.items() if c is not None])

    # Parameters
    model.costs = Param(model.arcs, initialize={k: v for k, v in min_costs.items() if v is not None})
    model.availabilities = Param(model.nodes, initialize=min_availabilities, default=0)
    model.requirements = Param(model.nodes, initialize=min_requirements, default=0)

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

def create_minimal_fixed_variable_model():
    model = ConcreteModel()

    # Define variables
    model.flow_1 = Var(within=NonNegativeReals)
    model.flow_2 = Var(within=NonNegativeReals, bounds=(2.0, 2.0))  # Fixed via bounds

    # Define constraint
    model.eq = Constraint(expr=model.flow_1 + model.flow_2 == 2)

    # Define objective
    model.objective = Objective(expr=model.flow_1, sense=minimize)

    return model

if __name__ == "__main__":
    import sys, traceback
    try:
        print("solving the model")
        use_glpk = False
        model = create_minimum_model2()
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
            print(f"tableau objective value = {solver.get_current_objective_value()}")
            print(f"final objective value = {model.objective()}")
            for var in solution:
                print(f"{var}: {solution[var]}")

    except Exception as ex:
        traceback.print_exception(ex)