from totsu.elastic import analyze_infeasibility
from .transportation import create_model, capacities, requirements, print_results_from_model


def _build_pretty_name(model):
    suppliers = tuple(model.S)
    customers = tuple(model.T)

    def pretty_name(con):
        comp = con.parent_component().name
        idx = con.index()

        if comp == "demand_constraints" and isinstance(idx, int) and 1 <= idx <= len(customers):
            return f"demand requirement at {customers[idx - 1]}"
        if comp == "supply_constraints" and isinstance(idx, int) and 1 <= idx <= len(suppliers):
            return f"supply capacity at {suppliers[idx - 1]}"
        return None

    return pretty_name


def main():
    # increase the requirements of T2 by 100%
    requirements["T2"] *= 2

    # 1) Create the infeasible model
    model = create_model()
    pretty_name = _build_pretty_name(model)

    # 2) Analyze the infeasibility
    analysis_result = analyze_infeasibility(model, pretty_name=pretty_name)

    # 3) Print the results
    print("=== Original Model Shipment Table ===")
    print_results_from_model(model)
    print("=== Elastic Model Shipment Table (violation_only) ===")
    print_results_from_model(analysis_result.elastic_model)
    analysis_result.print_summary()

    # 4) Analyze the infeasibility(with the original objective included in the elastic formulation)
    # In original_plus_violation mode, use a large violation penalty so
    # "serve demand by shipping" dominates "drop demand with cheap violations".
    analysis_result = analyze_infeasibility(
        model,
        violation_only=False,
        default_penalty=1000.0,
        pretty_name=pretty_name,
    )

    # 5) Print the results(with the original objective included in the elastic formulation)
    print("=== Original Model Shipment Table ===")
    print_results_from_model(model)
    print("=== Elastic Model Shipment Table (original_plus_violation) ===")
    print_results_from_model(analysis_result.elastic_model)
    analysis_result.print_summary()

if __name__ == "__main__":
    main()
