from pyomo.environ import AbstractModel, Set, Param, Var, Objective, Constraint, SolverFactory, maximize, minimize, Binary, Any

POKEMON_TYPES = ["Normal", "Fire", "Water", "Grass", "Electric", "Fighting", "Poison", "Ground", "Flying", 
                 "Psychic", "Bug", "Rock", "Ghost", "Ice", "Dragon", "Dark", "Steel", "Fairy"]
OBJECTIVE_TYPES = ["attack", "defense", "balanced"]
EFFECTIVENESS = {
    "S": 1.5, # Super effective
    "O": 1.0, # Normal effectiveness
    "N": 0.5, # Not very effective
    "I": 0.25 # Immune
}
# See: Type Effectiveness in Battle
# https://niantic.helpshift.com/hc/en/6-pokemon-go/faq/2132-type-effectiveness-in-battle/
MATCHUP_MATRIX = {
    "Normal":   {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "N", 
                 "Ghost": "I", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Fire":     {"Normal": "O", "Fire": "N", "Water": "N", "Grass": "S", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "S", "Rock": "N", 
                 "Ghost": "O", "Ice": "S", "Dragon": "N", "Dark": "O", "Steel": "S", "Fairy": "O"},
    "Water":    {"Normal": "O", "Fire": "S", "Water": "N", "Grass": "N", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "S", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "S", 
                 "Ghost": "O", "Ice": "O", "Dragon": "N", "Dark": "O", "Steel": "O", "Fairy": "O"},
    "Grass":    {"Normal": "O", "Fire": "N", "Water": "O", "Grass": "N", "Electric": "O", "Fighting": "O", 
                 "Poison": "N", "Ground": "O", "Flying": "N", "Psychic": "O", "Bug": "N", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "N", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Electric":    {"Normal": "O", "Fire": "O", "Water": "S", "Grass": "N", "Electric": "N", "Fighting": "O", 
                 "Poison": "O", "Ground": "I", "Flying": "S", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "N", "Dark": "O", "Steel": "O", "Fairy": "O"},
    "Fighting":    {"Normal": "S", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "N", "Ground": "O", "Flying": "N", "Psychic": "N", "Bug": "N", "Rock": "S", 
                 "Ghost": "O", "Ice": "S", "Dragon": "O", "Dark": "S", "Steel": "S", "Fairy": "N"},
    "Poison":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "I", "Fairy": "O"},
    "Ground":    {"Normal": "O", "Fire": "S", "Water": "O", "Grass": "N", "Electric": "S", "Fighting": "O", 
                 "Poison": "S", "Ground": "O", "Flying": "I", "Psychic": "O", "Bug": "N", "Rock": "S", 
                 "Ghost": "O", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "S", "Fairy": "O"},
    "Flying":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "S", "Electric": "N", "Fighting": "S", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "S", "Rock": "N", 
                 "Ghost": "O", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Psychic":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "S", 
                 "Poison": "S", "Ground": "O", "Flying": "O", "Psychic": "N", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "O", "Dark": "I", "Steel": "N", "Fairy": "O"},
    "Bug":    {"Normal": "O", "Fire": "N", "Water": "O", "Grass": "S", "Electric": "O", "Fighting": "N", 
                 "Poison": "N", "Ground": "O", "Flying": "N", "Psychic": "S", "Bug": "O", "Rock": "O", 
                 "Ghost": "N", "Ice": "O", "Dragon": "O", "Dark": "S", "Steel": "N", "Fairy": "N"},
    "Rock":    {"Normal": "O", "Fire": "S", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "N", 
                 "Poison": "O", "Ground": "N", "Flying": "S", "Psychic": "O", "Bug": "S", "Rock": "O", 
                 "Ghost": "O", "Ice": "S", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Ghost":    {"Normal": "I", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "S", "Bug": "O", "Rock": "O", 
                 "Ghost": "S", "Ice": "O", "Dragon": "O", "Dark": "N", "Steel": "O", "Fairy": "O"},
    "Ice":    {"Normal": "O", "Fire": "N", "Water": "N", "Grass": "S", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "S", "Flying": "S", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "N", "Dragon": "S", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Dragon":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "S", "Dark": "O", "Steel": "N", "Fairy": "I"},
    "Dark":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "N", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "S", "Bug": "O", "Rock": "O", 
                 "Ghost": "S", "Ice": "O", "Dragon": "O", "Dark": "N", "Steel": "O", "Fairy": "N"},
    "Steel":    {"Normal": "O", "Fire": "N", "Water": "N", "Grass": "O", "Electric": "N", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "S", 
                 "Ghost": "O", "Ice": "S", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "S"},
    "Fairy":    {"Normal": "O", "Fire": "N", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "S", 
                 "Poison": "N", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "S", "Dark": "S", "Steel": "N", "Fairy": "O"},
}

class PokemonTeamSelection:
    def __init__(self, objective_type='attack', balance_lambda=0.5):
        """
        objective_type: 'attack', 'defense', or 'balanced'.
        balance_lambda: Weight for balanced objective (0-1, where 1 favors attack and 0 favors defense).
        """
        self.objective_type = objective_type
        self.balance_lambda = balance_lambda
        
        # Create Pyomo model
        self.model = self.create_model()

    def create_model(self):
        model = AbstractModel()
        
        # Sets
        # All 18 types + Dual-Types
        model.T = Set(initialize=POKEMON_TYPES, ordered=True)  # Single-type only
        model.T_MIXED = Set(within=Any, initialize=[(t, None) for t in POKEMON_TYPES] + [(t1, t2) for t1 in POKEMON_TYPES for t2 in POKEMON_TYPES if t1 != t2])

        # Single-type Pokémon → ("Fire", None)
        # Dual-type Pokémon → ("Fire", "Flying")
        model.S = Set(within=model.T_MIXED, dimen=2)
        
        # Parameters
        def defense_effectiveness_rule(model, atk, *defn):
            """Handles both single-type and dual-type defenders."""
            if len(defn) == 1:
                return EFFECTIVENESS[MATCHUP_MATRIX[atk][defn[0]]]
            elif len(defn) == 2:  # Dual-type case
                if defn[1] is None:
                    return EFFECTIVENESS[MATCHUP_MATRIX[atk][defn[0]]]  # Single type case
                return EFFECTIVENESS[MATCHUP_MATRIX[atk][defn[0]]] * EFFECTIVENESS[MATCHUP_MATRIX[atk][defn[1]]]
            else:
                raise ValueError(f"Invalid type combination: {defn}")

        model.A = Param(model.T, model.T | model.T_MIXED, initialize=defense_effectiveness_rule)
        model.N = Param() # the number of selection
        model.balance_lambda = Param()
        
        # Decision Variables
        model.x = Var(model.T, within=Binary)  # 1 if type i is selected, else 0
        
        # Constraints
        def team_size_rule(model):
            return Constraint.Feasible if not model.T else sum(model.x[i] for i in model.T) == model.N
        model.team_size = Constraint(rule=team_size_rule)
        
        # Objective Function
        def attack_objective_rule(model):
            return sum(model.x[i] * sum(model.A[i, j] for j in model.S) for i in model.T)

        def defense_objective_rule(model):
            total_defense = 0
            for j in model.S:  # Opponent types (single or dual)
                for i in model.T:  # Attackers are always single-type
                    if isinstance(j, tuple) and j[1] is not None:  # Dual-type defender
                        total_defense += model.A[i, j] * model.x[i]
                    else:  # Single-type defender
                        if isinstance(j, tuple) and j[1] is not None:  # Dual-type defender
                            total_defense += model.A[i, j] * model.x[i]
                        else:  # Single-type defender (ensure it's not already a tuple)
                            total_defense += model.A[i, (j if isinstance(j, str) else j[0], None)] * model.x[i]
            return total_defense

        def balanced_objective_rule(model):
            total_attack = 0
            total_defense = 0

            # Attack term
            for i in model.T:  # Single-type attackers
                for j in model.S:  # Opponent types (single or dual)
                    total_attack += model.A[i, j] * model.x[i]

            # Defense term
            for j in model.S:  # Opponent types (single or dual)
                for i in model.T:  # Single-type attackers
                    if isinstance(j, tuple) and j[1] is not None:  # Dual-type defender
                        total_defense += model.A[i, j] * model.x[i]
                    else:  # Single-type defender
                        if isinstance(j, tuple) and j[1] is not None:  # Dual-type defender
                            total_defense += model.A[i, j] * model.x[i]
                        else:  # Single-type defender (ensure it's not already a tuple)
                            total_defense += model.A[i, (j if isinstance(j, str) else j[0], None)] * model.x[i]

            return model.balance_lambda * total_attack - (1 - model.balance_lambda) * total_defense
        
        if self.objective_type == 'attack':
            model.obj = Objective(rule=attack_objective_rule, sense=maximize)
        elif self.objective_type == 'defense':
            model.obj = Objective(rule=defense_objective_rule, sense=minimize)
        elif self.objective_type == 'balanced':
            model.obj = Objective(rule=balanced_objective_rule, sense=maximize)
        else:
            raise ValueError("Invalid objective type. Choose from 'attack', 'defense', or 'balanced'.")
        
        return model

    def solve(self, data, solver_name='glpk'):
        """Solve the ILP problem using the specified solver."""
        instance = self.model.create_instance(data)
        solver = SolverFactory(solver_name)
        result = solver.solve(instance, tee=True)
        instance.display()
        selected_types = [pokemon_type for pokemon_type in instance.T if instance.x[pokemon_type].value == 1]
        return selected_types, result

def main(opponent_types, objective_type, team_size):
    print(f"chosen opponent types: {opponent_types}")
    print(f"chosen objective type: {objective_type}")

    # the validity of given options
    parsed_opponent_types = parse_opponent_types(args.opponent_types)

    if not objective_type in OBJECTIVE_TYPES:
        print(f"{objective_type} is not a valid type")
        return

    pokemon_model = PokemonTeamSelection(objective_type)  # or 'defense', 'balanced'
    selected_team, result = pokemon_model.solve(create_input_data(parsed_opponent_types, team_size))
    print("Selected Pokémon Types:", selected_team)

def create_input_data(opponent_types, team_size):
    if team_size < 1 or team_size > 6:
        raise ValueError(f"team_size should be between 1 and 6, not {team_size}")

    if len(opponent_types) == 0 or len(opponent_types) > 3:
        raise ValueError(f"Chose 1~3 opponent types, not {len(opponent_types)}")

    input_data = {None: {
        'S': format_opponent_types(opponent_types),  # Opponent Pokémon
        'N': {None: team_size},
        'balance_lambda': {None: 0.5}
    }}
    return input_data

def parse_opponent_types(opponent_types):
    parsed_types = []
    for opt in opponent_types:
        if "-" in opt:  # Mixed type
            types = tuple(opt.split("-"))
            if len(types) == 2 and all(t in POKEMON_TYPES for t in types):
                parsed_types.append(types)
            else:
                raise ValueError(f"Invalid mixed type: {opt}")
        else:  # Single type
            if opt in POKEMON_TYPES:
                parsed_types.append(opt)
            else:
                raise ValueError(f"Invalid type: {opt}")
    return parsed_types

def format_opponent_types(opponent_types):
    return [(t, None) if isinstance(t, str) else t for t in opponent_types]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show the best team selection for pokemon battle")
    parser.add_argument("objective_type", type=str, help=f"Either of {OBJECTIVE_TYPES}")
    parser.add_argument("opponent_types", nargs="+", help="Types of opponents. Choose up to 3 among single or mixed types (e.g., Ice, Fire-Flying, Ghost)")
    parser.add_argument("--team_size", type=int, default=3, help="The number of team members selected")
    args = parser.parse_args()
    main(args.opponent_types, args.objective_type, args.team_size)
