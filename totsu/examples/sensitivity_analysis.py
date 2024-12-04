from pyomo.environ import SolverFactory
from ..utils.model_builder import ModelBuilder
from ..utils.sensitivity_analyzer import SensitivityAnalyzer
from ..core.totsu_simplex_solver import TotsuSimplexSolver # required to register

def main(model_name):
    # Create the primal model
    primal_model = ModelBuilder.build_model_by_name(model_name)
    solver = SolverFactory("totsu")

    visualizer = SensitivityAnalyzer(primal_model, solver)
    if visualizer.solve_primal():
        visualizer.show_analyzer()
    else:
        print("Failed to solve the primal model.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show Tableau Visualizer by using ModelBuilder")
    parser.add_argument("model", type=str, help="The model name among simple, pivot, optimal, infeasible, feasible, cyclic, negative_rhs, unbounded")
    args = parser.parse_args()
    main(args.model)