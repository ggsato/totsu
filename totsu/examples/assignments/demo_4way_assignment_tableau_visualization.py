from .demo_4way_assignment import build_toy_model
from ...utils.tableau_visualizer import TableauVisualizer
from ...utils.degeneracy_profiler import DegeneracyProfiler, extract_matrix

def main():
    # Create the model
    model = build_toy_model()

    # Visualize the solution
    visualizer = TableauVisualizer(model)
    visualizer.solve_model()

    md = extract_matrix(model)
    prof = DegeneracyProfiler(md)
    struct = prof.analyze_structure()
    print(prof.report(struct))

    visualizer.show_tableau_visualization()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show Tableau Visualizer of 4-way assignment problem.")
    args = parser.parse_args()
    main()
