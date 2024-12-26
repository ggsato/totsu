from .transportation import create_model
from .....utils.tableau_visualizer import TableauVisualizer

def main():
    # Create the model
    model = create_model()

    # Visualize the solution
    visualizer = TableauVisualizer(model)
    try:
        visualizer.solve_model()
        visualizer.show_tableau_visualization()
    except:
        print("Failed to solve the model and show the visualization")

if __name__ == "__main__":
    main()
