from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver

def create_model():
    model = ConcreteModel()

    # Variables
    # the total quantities of INPUTs
    model.x_c = Var(bounds=(0, None), initialize=0)  # Coal
    model.x_s = Var(bounds=(0, None), initialize=0)  # Steel
    model.x_t = Var(bounds=(0, None), initialize=0)  # Transport
    # Achievable bills of goods
    model.y_c = Var(bounds=(0, None), initialize=0)  # Coal
    model.y_s = Var(bounds=(0, None), initialize=0)  # Steel
    model.y_t = Var(bounds=(0, None), initialize=0)  # Transport

    # Objective
    # Maximize the total output
    model.obj = Objective(expr= model.x_c + model.x_s + model.x_t, sense=maximize)

    # Constraints
    # Coal process
    model.coal_process = Constraint(expr=
                              0.9 * model.x_c - 0.5 * model.x_s - 0.4 * model.x_t == model.y_c)
    # Streel process
    model.steel_process = Constraint(expr=
                              -0.1 * model.x_c + 0.9 * model.x_s - 0.2 * model.x_t == model.y_s)
    # Transport process
    model.transport_process = Constraint(expr=
                              -0.2 * model.x_c - 0.1 * model.x_s + 0.8 * model.x_t == model.y_t)
    # Labor limitation
    model.labor_limitation = Constraint(expr=
                              0.6 * model.x_c + 0.3 * model.x_s + 0.2 * model.x_t <= 40)
    return model

def create_alternative_model():
    model = ConcreteModel()

    # Variables
    # the total quantities of INPUTs
    model.x_c1 = Var(bounds=(0, None), initialize=0)  # Coal
    model.x_s1 = Var(bounds=(0, None), initialize=0)  # Steel
    model.x_t1 = Var(bounds=(0, None), initialize=0)  # Transport
    # the total quantities of alternative INPUTs
    model.x_c2 = Var(bounds=(0, None), initialize=0)  # Coal
    model.x_s2 = Var(bounds=(0, None), initialize=0)  # Steel
    model.x_t2 = Var(bounds=(0, None), initialize=0)  # Transport

    # Objective
    # Maximize the total output
    model.obj = Objective(expr= model.x_c1 + model.x_s1 + model.x_t1 + 
                                model.x_c2 + model.x_s2 + model.x_t2, sense=maximize)

    # Constraints
    # Coal process
    model.coal_process = Constraint(expr=
                              0.9 * model.x_c1 - 0.5 * model.x_s1 - 0.4 * model.x_t1 +
                              0.8 * model.x_c2 - 0.6 * model.x_s2 - 0.6 * model.x_t2 == 20)
    # Streel process
    model.steel_process = Constraint(expr=
                              -0.1 * model.x_c1 + 0.9 * model.x_s1 - 0.2 * model.x_t1 + 
                              -0.0 * model.x_c2 + 0.9 * model.x_s2 - 0.2 * model.x_t2 == 5)
    # Transport process
    model.transport_process = Constraint(expr=
                              -0.2 * model.x_c1 - 0.1 * model.x_s1 + 0.8 * model.x_t1 + 
                              -0.1 * model.x_c2 - 0.0 * model.x_s2 + 0.95 * model.x_t2 == 25)
    # Labor limitation
    model.labor_limitation = Constraint(expr=
                              0.6 * model.x_c1 + 0.3 * model.x_s1 + 0.2 * model.x_t1 + 
                              0.7 * model.x_c2 + 0.3 * model.x_s2 + 0.15 * model.x_t2 <= 60)
    return model

if __name__ == "__main__":
    import sys, traceback
    try:
        print("solving the model")
        model = create_model()
        solver = SuperSimplexSolver()
        solution = solver.solve(model)

        print(f"objective value = {solver.get_current_objective_value()}") # 116.60

        # 6 variables under 4 constraints lead to two zeros
        print(f"xc = {solution["x_c"]:.2f}") # 37.25
        print(f"xs = {solution["x_s"]:.2f}") # 17.81
        print(f"xt = {solution["x_t"]:.2f}") # 61.54
        print(f"yc = {solution["y_c"]:.2f}") # 0
        print(f"ys = {solution["y_s"]:.2f}") # 0
        print(f"yt = {solution["y_t"]:.2f}") # 40.00

        print("solving the alternative model")
        model = create_alternative_model()
        solver = SuperSimplexSolver()
        solution = solver.solve(model)

        print(f"objective value = {solver.get_current_objective_value()}") # 146.60087719298247

        # 6 variables under 4 constraints lead to two zeros
        print(f"xc1 = {solution["x_c1"]:.2f}") # 56.07
        print(f"xs1 = {solution["x_s1"]:.2f}") # 22.47
        print(f"xt1 = {solution["x_t1"]:.2f}") # 48.08
        print(f"xc2 = {solution["x_c2"]:.2f}") # 0
        print(f"xs2 = {solution["x_s2"]:.2f}") # 0
        print(f"xt2 = {solution["x_t2"]:.2f}") # 0

        model.x_c1 = solution["x_c1"]
        model.x_s1 = solution["x_s1"]
        model.x_t1 = solution["x_t1"]
        model.x_c2 = solution["x_c2"]
        model.x_s2 = solution["x_s2"]
        model.x_t2 = solution["x_t2"]

        print(f"coal process = {model.coal_process.body()}")
        print(f"steel process = {model.steel_process.body()}")
        print(f"transport process = {model.transport_process.body()}")
        print(f"labor limitation = {model.labor_limitation.body()}")

        # The optimal solution in the book
        print(f"checking the optimal solution in the book if constraints are met")
        model.x_c1 = 64.6
        model.x_s1 = 22.6
        model.x_t1 = 0
        model.x_c1 = 0
        model.x_s1 = 0
        model.x_t2 = 44.6

        print(f"coal process = {model.coal_process.body()}")
        print(f"steel process = {model.steel_process.body()}")
        print(f"transport process = {model.transport_process.body()}")
        print(f"labor limitation = {model.labor_limitation.body()}")

    except Exception as ex:
        traceback.print_exception(ex)