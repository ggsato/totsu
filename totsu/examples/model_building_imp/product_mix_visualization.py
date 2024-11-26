from .product_mix import create_model
from ...utils.sensitivity_analyzer import SensitivityAnalyzer

def main():
    # Create the primal model
    primal_model = create_model()

    visualizer = SensitivityAnalyzer(primal_model)
    if visualizer.solve_primal():
        visualizer.show_analyzer()
    else:
        print("Failed to solve the primal model.")

if __name__ == "__main__":
    main()