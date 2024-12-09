from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver

oil_types = ["Vegetable oils", "Non-vegetable oils"] # max 200 tons, 250 tons
refining_max_capacity = {oil_types[0]: 200, oil_types[1]: 250}
products = ["VEG1", "VEG2", "OIL1", "OIL2", "OIL3"]
costs = {products[0]: 110, products[1]: 120, products[2]: 130, products[3]: 110, products[4]: 115}
hardness = {products[0]: 8.8, products[1]: 6.1, products[2]: 2.0, products[3]: 4.2, products[4]: 5.0} # between 3 and 6
hardness_range = {"MIN": 3, "MAX": 6}
price = 150 # per ton

def create_model():
    model = ConcreteModel()

    # Variables
    # the number of PROD X produced in a month
    model.x1 = Var(bounds=(0, None), initialize=0)  # VEG1
    model.x2 = Var(bounds=(0, None), initialize=0)  # VEG2
    model.x3 = Var(bounds=(0, None), initialize=0)  # OIL1
    model.x4 = Var(bounds=(0, None), initialize=0)  # OIL2
    model.x5 = Var(bounds=(0, None), initialize=0)  # OIL3
    model.y = Var(bounds=(0, None), initialize=0)   # FINAL PRODUCT

    # Objective
    # Maximize the net profits
    model.obj = Objective(expr=
                          -costs[products[0]] * model.x1 + 
                          -costs[products[1]] * model.x2 + 
                          -costs[products[2]] * model.x3 + 
                          -costs[products[3]] * model.x4 + 
                          -costs[products[4]] * model.x5 +
                          price * model.y,
                          sense=maximize)

    # Constraints
    # Refining veg capacity
    model.refining_capacity_veg = Constraint(expr=
                                    model.x1 + model.x2 <= refining_max_capacity[oil_types[0]])
    # Refining veg capacity
    model.refining_capacity_nonveg = Constraint(expr=
                                    model.x3 + model.x4 + model.x5 <= refining_max_capacity[oil_types[1]])
    # Hardness
    model.hardness_max = Constraint(expr=
                              hardness[products[0]] * model.x1 +
                              hardness[products[1]] * model.x2 + 
                              hardness[products[2]] * model.x3 +
                              hardness[products[3]] * model.x4 + 
                              hardness[products[4]] * model.x5 -
                              hardness_range["MAX"] * model.y <= 0)

    model.hardness_min = Constraint(expr=
                              hardness[products[0]] * model.x1 +
                              hardness[products[1]] * model.x2 + 
                              hardness[products[2]] * model.x3 +
                              hardness[products[3]] * model.x4 + 
                              hardness[products[4]] * model.x5 -
                              hardness_range["MIN"] * model.y >= 0)

    # weight
    model.weight = Constraint(expr=model.x1 + model.x2 + model.x3 + model.x4 + model.x5 - model.y == 0)

    return model

if __name__ == "__main__":
    import sys, traceback
    try:
        model = create_model()
        solver = SuperSimplexSolver()
        solution = solver.solve(model)

        print(f"objective value = {solver.get_current_objective_value()}") # 17592
        print(f"x1 = {solution["x1"]:.1f}") # 159.3
        print(f"x2 = {solution["x2"]:.1f}") # 40.7
        print(f"x3 = {solution["x3"]:.1f}") # 0
        print(f"x4 = {solution["x4"]:.1f}") # 250
        print(f"x5 = {solution["x5"]:.1f}") # 0
        print(f"y = {solution["y"]:.1f}") # 450

        # update the model
        model.x1 = solution["x1"]
        model.x2 = solution["x2"]
        model.x3 = solution["x3"]
        model.x4 = solution["x4"]
        model.x5 = solution["x5"]
        model.y = solution["y"]
        print(f"Refining capacity usage: veg={model.refining_capacity_veg.body()}, non veg={model.refining_capacity_nonveg.body()}")
        print(f"Hardness range: min={model.hardness_min.body()}, max={model.hardness_max.body()}")
        print(f"Weight = {model.weight.body()}")

    except Exception as ex:
        traceback.print_exception(ex)