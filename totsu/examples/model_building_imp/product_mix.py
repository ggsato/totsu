from pyomo.environ import *
from totsu.core.super_simplex_solver import SuperSimplexSolver

products = ["PROD1", "PROD2", "PROD3", "PROD4", "PROD5"]
profits = {products[0]: 550, products[1]: 600, products[2]: 350, products[3]: 400, products[4]: 200}
processes = ["Grinding", "Drilling", "Manpower"]
processing = {
    processes[0]: {products[0]: 12, products[1]: 20, products[2]: None, products[3]: 25, products[4]: 15},
    processes[1]: {products[0]: 10, products[1]: 8, products[2]: 16, products[3]: None, products[4]: None},
    processes[2]: {products[0]: 20, products[1]: 20, products[2]: 20, products[3]: 20, products[4]: 20}
}
maximum_capacities = {processes[0]: 288, processes[1]: 192, processes[2]: 384}

def create_model():
    model = ConcreteModel()

    # Variables
    # the number of PROD X that should be produced in a week
    model.x1 = Var(bounds=(0, None), initialize=0)  # PROD 1
    model.x2 = Var(bounds=(0, None), initialize=0)  # PROD 2
    model.x3 = Var(bounds=(0, None), initialize=0)  # PROD 3
    model.x4 = Var(bounds=(0, None), initialize=0)  # PROD 4
    model.x5 = Var(bounds=(0, None), initialize=0)  # PROD 5

    # Objective
    # Maximize the total profits
    model.obj = Objective(expr=
                          profits[products[0]] * model.x1 + 
                          profits[products[1]] * model.x2 + 
                          profits[products[2]] * model.x3 + 
                          profits[products[3]] * model.x4 + 
                          profits[products[4]] * model.x5,  
                          sense=maximize)

    # Constraints
    # Grinding capacity
    model.grinding_capacity = Constraint(expr=
                              processing[processes[0]][products[0]] * model.x1 + 
                              processing[processes[0]][products[1]] * model.x2 + 
                              processing[processes[0]][products[3]] * model.x4 + 
                              processing[processes[0]][products[4]] * model.x5 <= maximum_capacities[processes[0]])
    # Drilling capacity
    model.drilling_capacity = Constraint(expr=
                              processing[processes[1]][products[0]] * model.x1 + 
                              processing[processes[1]][products[1]] * model.x2 + 
                              processing[processes[1]][products[2]] * model.x3 <= maximum_capacities[processes[1]])
    # Manpower capacity
    model.manpower_capacity = Constraint(expr=
                              processing[processes[2]][products[0]] * model.x1 + 
                              processing[processes[2]][products[1]] * model.x2 + 
                              processing[processes[2]][products[2]] * model.x3 + 
                              processing[processes[2]][products[3]] * model.x4 + 
                              processing[processes[2]][products[4]] * model.x5 <= maximum_capacities[processes[2]])
    return model

def create_dual_model():
    model = ConcreteModel()

    # Dual Variables
    # the valuations for each hour of each of the capacities
    model.y1 = Var(bounds=(0, None), initialize=0)  # Grinding
    model.y2 = Var(bounds=(0, None), initialize=0)  # Drilling
    model.y3 = Var(bounds=(0, None), initialize=0)  # Manpower

    # Objective
    # Minimize the total valuations(costs)
    model.obj = Objective(expr=
                          maximum_capacities[processes[0]] * model.y1 + 
                          maximum_capacities[processes[1]] * model.y2 + 
                          maximum_capacities[processes[2]] * model.y3,  
                          sense=minimize)
    
    # Constraints
    # PROD 1
    model.con6_11 = Constraint(expr=
                              processing[processes[0]][products[0]] * model.y1 + 
                              processing[processes[1]][products[0]] * model.y2 + 
                              processing[processes[2]][products[0]] * model.y3 >= profits[products[0]])
    
    # PROD 2
    model.con6_12 = Constraint(expr=
                              processing[processes[0]][products[1]] * model.y1 + 
                              processing[processes[1]][products[1]] * model.y2 + 
                              processing[processes[2]][products[1]] * model.y3 >= profits[products[1]])
    
    # PROD 3
    model.con6_13 = Constraint(expr=
                              processing[processes[1]][products[2]] * model.y2 + 
                              processing[processes[2]][products[2]] * model.y3 >= profits[products[2]])
    
    # PROD 4
    model.con6_14 = Constraint(expr=
                              processing[processes[0]][products[3]] * model.y1 + 
                              processing[processes[2]][products[3]] * model.y3 >= profits[products[3]])
    
    # PROD 5
    model.con6_15 = Constraint(expr=
                              processing[processes[0]][products[4]] * model.y1 + 
                              processing[processes[2]][products[4]] * model.y3 >= profits[products[4]])

    return model

if __name__ == "__main__":
    import sys, traceback
    try:
        model = create_model()
        solver = SuperSimplexSolver()
        solution = solver.solve(model)

        print(f"objective value = {solution["objective_value"]}") # 10,920
        print(f"x1 = {solution["x1"]:.1f}") # 12
        print(f"x2 = {solution["x2"]:.1f}") # 7.2
        print(f"x3 = {solution["x3"]:.1f}") # 0
        print(f"x4 = {solution["x4"]:.1f}") # 0
        print(f"x5 = {solution["x5"]:.1f}") # 0

        # update the model
        model.x1 = solution["x1"]
        model.x2 = solution["x2"]
        model.x3 = solution["x3"]
        model.x4 = solution["x4"]
        model.x5 = solution["x5"]
        print(f"Grinding usage: {model.con1_2.body()} <= {maximum_capacities[processes[0]]}") # 288.0 <= 288, binding
        print(f"Drilling usage: {model.con1_3.body()} <= {maximum_capacities[processes[1]]}") # 177.6 <= 192, not binding
        print(f"Manpower usage: {model.con1_4.body()} <= {maximum_capacities[processes[2]]}") # 384.0 <= 384, binding

        dual_model = create_dual_model()
        solution_dual = solver.solve(dual_model)

        print(f"objective value = {solution_dual["objective_value"]}") # 10,920, the same value of the primal model(duality theorem)
        # Dual variables, Shadow Prices
        print(f"y1 = {solution_dual["y1"]:.2f}") # 6.25
        print(f"y2 = {solution_dual["y2"]:.2f}") # 0
        print(f"y3 = {solution_dual["y3"]:.2f}") # 23.75

        # update the dual model
        dual_model.y1 = solution_dual["y1"]
        dual_model.y2 = solution_dual["y2"]
        dual_model.y3 = solution_dual["y3"]
        # Reduced Costs
        print(f"PROD 1 cost: {dual_model.con6_11.body()} >= {profits[products[0]]}, reduced cost: {dual_model.con6_11.body() - profits[products[0]]}") # 550 >= 550
        print(f"PROD 2 cost: {dual_model.con6_12.body()} >= {profits[products[1]]}, reduced cost: {dual_model.con6_12.body() - profits[products[1]]}") # 600 >= 600
        print(f"PROD 3 cost: {dual_model.con6_13.body()} >= {profits[products[2]]}, reduced cost: {dual_model.con6_13.body() - profits[products[2]]}") # 475 >= 350, negative resultant profit
        print(f"PROD 4 cost: {dual_model.con6_14.body()} >= {profits[products[3]]}, reduced cost: {dual_model.con6_14.body() - profits[products[3]]}") # 631 >= 400, negative resultant profit
        print(f"PROD 5 cost: {dual_model.con6_15.body()} >= {profits[products[4]]}, reduced cost: {dual_model.con6_15.body() - profits[products[4]]}") # 568.75 >= 200, negative resultant profit

    except Exception as ex:
        traceback.print_exception(ex)