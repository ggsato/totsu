from pyomo.environ import SolverFactory
from .minimum_cost_flow import create_model
from .....utils.sensitivity_analyzer import SensitivityAnalyzer
from .....core.totsu_simplex_solver import TotsuSimplexSolver # required to register

def main():
    # Create the model
    model = create_model()
    solver = SolverFactory("totsu")

    visualizer = SensitivityAnalyzer(model, solver)
    if visualizer.solve_primal():
        visualizer.show_analyzer()
    else:
        print("Failed to solve the primal model.")

if __name__ == "__main__":
    main()