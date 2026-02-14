from .demo_4way_assignment_scaled import build_large_model, build_medium_model
#from ...utils.tableau_visualizer import TableauVisualizer
from ...utils.degeneracy_profiler import DegeneracyProfiler, extract_matrix

def main(exact_cover: bool):
    # Create the model
    model = build_medium_model(exact_cover=exact_cover, W=4, C=4, D=4, S=2)

    # Visualize the solution
    #visualizer = TableauVisualizer(model)
    #visualizer.solve_model()

    md = extract_matrix(model)
    prof = DegeneracyProfiler(md)
    struct = prof.analyze_structure()
    print(prof.report(struct))

    #visualizer.show_tableau_visualization()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show Tableau Visualizer of 4-way assignment large problem.")
    parser.add_argument("--exact-cover", action="store_true",
                    help="Use exact cover on (c,d,s)==1; else soft cover with penalty.")
    args = parser.parse_args()
    main(args.exact_cover)
