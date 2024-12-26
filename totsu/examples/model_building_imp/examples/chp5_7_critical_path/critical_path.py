from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver
from pyomo.environ import SolverFactory

def create_model():
    # Create a Pyomo model
    model = ConcreteModel()

    # Variables
    model.t_0 = Var(within=NonNegativeReals) # start time for activities 0-1, 0-3 and 0-2
    model.t_1 = Var(within=NonNegativeReals) # start time for activity 1-3
    model.t_2 = Var(within=NonNegativeReals) # start time for activity 2-5
    model.t_3 = Var(within=NonNegativeReals) # start time for activities 3-4
    model.t_4 = Var(within=NonNegativeReals) # start time for activities 4-2, 4-5
    model.t_5 = Var(within=NonNegativeReals) # start time for activities 5-6
    model.z = Var(within=NonNegativeReals) # finish time for the project

    # Constraints
    model.con0_1 = Constraint(expr=-model.t_0 + model.t_1 >= 4)
    model.con0_2 = Constraint(expr=-model.t_0 + model.t_2 >= 12)
    model.con0_3 = Constraint(expr=-model.t_0 + model.t_3 >= 7)
    model.con1_3 = Constraint(expr=-model.t_1 + model.t_3 >= 2)
    model.con3_4 = Constraint(expr=-model.t_3 + model.t_4 >= 10)
    model.con4_2 = Constraint(expr=-model.t_4 + model.t_2 >= 0)
    model.con2_5 = Constraint(expr=-model.t_2 + model.t_5 >= 5)
    model.con4_5 = Constraint(expr=-model.t_4 + model.t_5 >= 3)
    model.con5_z = Constraint(expr=-model.t_5 + model.z >= 4)

    # Objective: Minimize the total cost
    model.objective = Objective(rule=model.z, sense=minimize)

    return model

def main(use_glpk):
    import sys, traceback
    try:
        print(f"solving the model - use_glpk = {use_glpk}")
        model = create_model()
        if use_glpk:
            solver = SolverFactory("glpk")
        else:
            solver = SuperSimplexSolver()

        solution = solver.solve(model)

        print(f"objective value = {model.objective()}")
        print(f"t_0 = {model.t_0()}") # 0
        print(f"t_1 = {model.t_1()}") # 4
        print(f"t_2 = {model.t_2()}") # 12
        print(f"t_3 = {model.t_3()}") # 6
        print(f"t_4 = {model.t_4()}") # 16
        print(f"t_5 = {model.t_5()}") # 19
        print(f"z = {model.z()}") # 23

    except Exception as ex:
        traceback.print_exception(ex)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Minimum Cost Flow")
    parser.add_argument("--use_glpk", action='store_true', help="If set, use the glpk solver, SuperSimplexSolver otherwise.")
    args = parser.parse_args()
    main(args.use_glpk)