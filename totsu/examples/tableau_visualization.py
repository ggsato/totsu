from ..utils.model_builder import ModelBuilder
from ..utils.tableau_visualizer import TableauVisualizer
from ..core.super_simplex_solver import SuperSimplexSolver

def main(model_name):
    # Create the primal model
    primal_model = ModelBuilder.build_model_by_name(model_name)

    # Initialize the solver
    solver = SuperSimplexSolver()

    # Visualize the solution
    visualizer = TableauVisualizer(primal_model, solver)
    try:
        visualizer.solve_model()
        visualizer.show_tableau_visualization()
    except:
        print("Failed to solve the model and show the visualization")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show Tableau Visualizer by using ModelBuilder")
    parser.add_argument("model", type=str, help="The model name among simple, pivot, optimal, infeasible, feasible, cyclic, negative_rhs, unbounded")
    args = parser.parse_args()
    main(args.model)