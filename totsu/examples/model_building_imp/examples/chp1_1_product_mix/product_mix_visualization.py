from pyomo.environ import SolverFactory
from .product_mix import create_model, create_dual_model
from .....utils.sensitivity_analyzer import SensitivityAnalyzer
from .....core.totsu_simplex_solver import TotsuSimplexSolver # required to register

def main(use_dual):
    # Create the model
    if use_dual:
        model = create_dual_model()
    else:
        model = create_model()
    solver = SolverFactory("totsu")

    visualizer = SensitivityAnalyzer(model, solver)
    if visualizer.solve_primal():
        visualizer.show_analyzer()
    else:
        print("Failed to solve the primal model.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show Visualizer of product mix models")
    parser.add_argument("--use_dual", action='store_true', help="If set, use the dual model, the primal model otherwise.")
    args = parser.parse_args()
    main(args.use_dual)