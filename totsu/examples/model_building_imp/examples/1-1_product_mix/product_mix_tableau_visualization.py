from .product_mix import create_model, create_dual_model
from .....utils.tableau_visualizer import TableauVisualizer
from .....core.super_simplex_solver import SuperSimplexSolver

def main(use_dual):
    # Create the model
    if use_dual:
        model = create_dual_model()
    else:
        model = create_model()

    # Initialize the solver
    solver = SuperSimplexSolver()

    # Visualize the solution
    visualizer = TableauVisualizer(model, solver)
    try:
        visualizer.solve_model()
        visualizer.show_tableau_visualization()
    except:
        print("Failed to solve the model and show the visualization")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show Tableau Visualizer of product mix models")
    parser.add_argument("--use_dual", action='store_true', help="If set, use the dual model, the primal model otherwise.")
    args = parser.parse_args()
    main(args.use_dual)
