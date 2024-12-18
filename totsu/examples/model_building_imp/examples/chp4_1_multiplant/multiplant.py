from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver

factories = ["Factory A", "Factory B"]
products = ["Standard", "Deluxe"]
profits = {products[0]: 10, products[1]: 15}
processes = ["Grinding", "Polishing"]
processing = {
    processes[0]: {
        factories[0]: 
        {products[0]: 4, products[1]: 2},
        factories[1]: 
        {products[0]: 5, products[1]: 3}
    },
    processes[1]: {
        factories[0]: 
        {products[0]: 2, products[1]: 5},
        factories[1]: 
        {products[0]: 5, products[1]: 6}
    }
}
maximum_capacities = {
        factories[0]: {
            processes[0]: 80, processes[1]: 60
        },
        factories[1]: {
            processes[0]: 60, processes[1]: 75
        }
}

raw_available = 120
raw_allocation = {factories[0]: 75, factories[1]: 45}
raw_per_unit = 4

def create_model(factory_name):
    model = ConcreteModel()

    # Variables
    # the number of products that should be produced in a week
    model.x1 = Var(bounds=(0, None), initialize=0)  # Standard
    model.x2 = Var(bounds=(0, None), initialize=0)  # Deluxe

    # Objective
    # Maximize the total profits
    model.obj = Objective(expr=
                          profits[products[0]] * model.x1 + 
                          profits[products[1]] * model.x2,  
                          sense=maximize)

    # Constraints
    # Grinding capacity
    model.grinding_capacity = Constraint(expr=
                              processing[processes[0]][factory_name][products[0]] * model.x1 + 
                              processing[processes[0]][factory_name][products[1]] * model.x2 <= maximum_capacities[factory_name][processes[0]])
    # Polishing capacity
    model.polishing_capacity = Constraint(expr=
                              processing[processes[1]][factory_name][products[0]] * model.x1 + 
                              processing[processes[1]][factory_name][products[1]] * model.x2 <= maximum_capacities[factory_name][processes[1]])
    # Raw capacity
    model.raw_capacity = Constraint(expr=
                              raw_per_unit * model.x1 + raw_per_unit * model.x2 <= raw_allocation[factory_name])
    return model

def create_company_model():
    model = ConcreteModel()

    # Variables
    # the number of products that should be produced in a week
    model.x1 = Var(bounds=(0, None), initialize=0)  # A: Standard
    model.x2 = Var(bounds=(0, None), initialize=0)  # A: Deluxe
    model.x3 = Var(bounds=(0, None), initialize=0)  # B: Standard
    model.x4 = Var(bounds=(0, None), initialize=0)  # B: Deluxe

    # Objective
    # Maximize the total profits
    model.obj = Objective(expr=
                          profits[products[0]] * model.x1 + 
                          profits[products[1]] * model.x2 +
                          profits[products[0]] * model.x3 + 
                          profits[products[1]] * model.x4,  
                          sense=maximize)

    # Constraints
    # Grinding capacity
    model.grinding_capacity_A = Constraint(expr=
                              processing[processes[0]][factories[0]][products[0]] * model.x1 + 
                              processing[processes[0]][factories[0]][products[1]] * model.x2 <= maximum_capacities[factories[0]][processes[0]])
    model.grinding_capacity_B = Constraint(expr=
                              processing[processes[0]][factories[1]][products[0]] * model.x3 + 
                              processing[processes[0]][factories[1]][products[1]] * model.x4 <= maximum_capacities[factories[1]][processes[0]])
    # Polishing capacity
    model.polishing_capacity_A = Constraint(expr=
                              processing[processes[1]][factories[0]][products[0]] * model.x1 + 
                              processing[processes[1]][factories[0]][products[1]] * model.x2 <= maximum_capacities[factories[0]][processes[1]])
    model.polishing_capacity_B = Constraint(expr=
                              processing[processes[1]][factories[1]][products[0]] * model.x3 + 
                              processing[processes[1]][factories[1]][products[1]] * model.x4 <= maximum_capacities[factories[1]][processes[1]])
    # Raw capacity
    model.raw_capacity = Constraint(expr=
                              raw_per_unit * model.x1 + raw_per_unit * model.x2 + 
                              raw_per_unit * model.x3 + raw_per_unit * model.x4 <= raw_available)
    return model

if __name__ == "__main__":
    import sys, traceback
    try:
        for factory_name in factories:
            print(f"solving the model of {factory_name}")
            model = create_model(factory_name)
            solver = SuperSimplexSolver()
            solution = solver.solve(model)

            print(f"objective value = {solver.get_current_objective_value()}") # A, B = 225, 168.75
            print(f"x1 = {solution["x1"]:.2f}") # A, B = 11.25, 0
            print(f"x2 = {solution["x2"]:.2f}") # A, B = 7.5, 11.25

            # update the model
            model.x1 = solution["x1"]
            model.x2 = solution["x2"]

            print(f"Grinding usage: {model.grinding_capacity.body()} <= {maximum_capacities[factory_name][processes[0]]}")
            print(f"Polishing usage: {model.polishing_capacity.body()} <= {maximum_capacities[factory_name][processes[1]]}")
            print(f"Raw usage: {model.raw_capacity.body()} <= {raw_allocation[factory_name]}")

        print("solving the company model")
        model = create_company_model()
        solver = SuperSimplexSolver()
        solution = solver.solve(model)

        print(f"objective value = {solver.get_current_objective_value()}") # 404.17
        print(f"x1 = {solution["x1"]:.2f}") # 9.17
        print(f"x2 = {solution["x2"]:.2f}") # 8.33
        print(f"x3 = {solution["x3"]:.2f}") # 0
        print(f"x4 = {solution["x4"]:.2f}") # 12.5

        # update the model
        model.x1 = solution["x1"]
        model.x2 = solution["x2"]
        model.x3 = solution["x3"]
        model.x4 = solution["x4"]

        print(f"Grinding usage(A): {model.grinding_capacity_A.body()} <= {maximum_capacities[factories[0]][processes[0]]}")
        print(f"Polishing usage(A): {model.polishing_capacity_A.body()} <= {maximum_capacities[factories[0]][processes[1]]}")
        print(f"Grinding usage(B): {model.grinding_capacity_B.body()} <= {maximum_capacities[factories[1]][processes[0]]}")
        print(f"Polishing usage(B): {model.polishing_capacity_B.body()} <= {maximum_capacities[factories[1]][processes[1]]}")
        print(f"Raw usage: {model.raw_capacity.body()} <= {raw_available}")

    except Exception as ex:
        traceback.print_exception(ex)