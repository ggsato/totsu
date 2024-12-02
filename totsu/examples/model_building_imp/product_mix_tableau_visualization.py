from .product_mix import create_model
from ...utils.tableau_visualizer import TableauVisualizer
from ...core.super_simplex_solver import SuperSimplexSolver

def main():
    # Create the primal model
    primal_model = create_model()

    # Initialize the solver
    solver = SuperSimplexSolver()

    # Visualize the solution
    visualizer = TableauVisualizer(primal_model, solver)
    if visualizer.solve_model():
        visualizer.show_tableau_visualization()
    else:
        print("Failed to solve the model.")

if __name__ == "__main__":
    main()
