from pyomo.environ import ConcreteModel, Var, Constraint, Objective, NonNegativeReals, minimize, maximize, value
    
class ModelBuilder:
    model_map = {
        "simple": "build_simple_lp_model",
        "pivot": "build_simple_requiring_pivot_model",
        "optimal": "build_simple_optimal_model",
        "infeasible": "build_simple_infeasible_model",
        "feasible": "build_simple_feasible_model",
        "cyclic": "build_simple_cyclic_model",
        "negative_rhs": "build_negative_rhs_model",
        "unbounded": "build_unbounded_model"
    }

    @staticmethod
    def build_model_by_name(model_name, params=None):
        # Assuming this method builds a model based on the model_name and parameters
        if model_name in ModelBuilder.model_map:
            method_name = ModelBuilder.model_map[model_name]
            return getattr(ModelBuilder, method_name)()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def build_simple_lp_model():
        # Create a simple LP model
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)
        model.y = Var(domain=NonNegativeReals)
        model.con1 = Constraint(expr=2 * model.x + model.y <= 10)
        model.con2 = Constraint(expr=model.x + 3 * model.y <= 15)
        model.obj = Objective(expr=3 * model.x + 2 * model.y, sense=minimize)
        return model
    
    @staticmethod
    def build_simple_requiring_pivot_model():
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)
        model.y = Var(domain=NonNegativeReals)
        model.con1 = Constraint(expr=model.x + model.y <= 5)
        model.con2 = Constraint(expr=model.x + 2 * model.y <= 8)
        model.obj = Objective(expr=3 * model.x + 5 * model.y, sense=minimize)
        return model
    
    @staticmethod
    def build_simple_optimal_model():
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)
        model.con1 = Constraint(expr=model.x >= 3)
        model.obj = Objective(expr=model.x, sense=minimize)
        return model

    @staticmethod
    def build_simple_infeasible_model():
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)
        model.con1 = Constraint(expr=model.x >= 5)
        model.con2 = Constraint(expr=model.x <= 3)
        model.obj = Objective(expr=model.x, sense=minimize)
        return model
    
    @staticmethod
    def build_simple_feasible_model():
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)
        model.y = Var(domain=NonNegativeReals)
        model.con1 = Constraint(expr=model.x + model.y == 10)
        model.obj = Objective(expr=model.x + 2 * model.y, sense=minimize)
        return model
    
    @staticmethod
    def build_simple_cyclic_model():
        model = ConcreteModel()
        model.x1 = Var(domain=NonNegativeReals)
        model.x2 = Var(domain=NonNegativeReals)
        model.x3 = Var(domain=NonNegativeReals)
        model.con1 = Constraint(expr=10 * model.x1 - 20 * model.x2 + 10 * model.x3 == 0)
        model.con2 = Constraint(expr=-20 * model.x1 + 10 * model.x2 + 10 * model.x3 == 0)
        model.obj = Objective(expr=model.x1 + model.x2 + model.x3, sense=minimize)
        return model
    
    @staticmethod
    def build_negative_rhs_model():
        # Create a model with negative rhs values
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)
        model.y = Var(domain=NonNegativeReals)

        # Constraints with negative rhs (adjusted to be feasible)
        model.constraint1 = Constraint(expr=2 * model.x + model.y >= -4)  # Always satisfied
        model.constraint2 = Constraint(expr=model.x + 2 * model.y <= 6)   # Adjusted RHS

        # Objective function
        model.objective = Objective(expr=model.x + model.y, sense=minimize)
        return model
    
    @staticmethod
    def build_unbounded_model():
        # Create a model that is unbounded
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)

        # Constraint
        model.constraint = Constraint(expr=-model.x <= -1)
        
        # Objective function (maximize without upper bound)
        model.objective = Objective(expr=model.x, sense=maximize)
        return model