from .minimum_cost_flow import create_model
from .....utils.tableau_visualizer import TableauVisualizer
from .....core.super_simplex_solver import SuperSimplexSolver

def main():
    # Create the model
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
    main()
