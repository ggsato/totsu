import pytest
from pyomo.environ import SolverFactory
from totsu.examples.pokemon.pokemon_team_selection import PokemonTeamSelection, create_input_data, parse_opponent_types

# Define test data
SINGLE_OPPONENT = ["Fire"]
DUAL_OPPONENT = ["Water-Electric"]
MULTI_OPPONENT = ["Grass", "Psychic", "Steel"]
INVALID_OPPONENT = ["Unknown"]
TEAM_SIZE = 3

@pytest.fixture
def model_attack():
    return PokemonTeamSelection(objective_type="attack")

@pytest.fixture
def model_defense():
    return PokemonTeamSelection(objective_type="defense")

@pytest.fixture
def model_balanced():
    return PokemonTeamSelection(objective_type="balanced")

@pytest.fixture
def solver():
    return SolverFactory("glpk")

def test_valid_single_opponent(model_attack, solver):
    input_data = create_input_data(parse_opponent_types(SINGLE_OPPONENT), TEAM_SIZE)
    selected_team, result, _ = model_attack.solve(input_data, solver_name="totsubb")
    assert isinstance(selected_team, list)
    assert len(selected_team) == TEAM_SIZE

def test_valid_dual_opponent(model_defense, solver):
    input_data = create_input_data(parse_opponent_types(DUAL_OPPONENT), TEAM_SIZE)
    selected_team, result, _ = model_defense.solve(input_data, solver_name="totsubb")
    assert isinstance(selected_team, list)
    assert len(selected_team) == TEAM_SIZE

def test_valid_multi_opponent(model_balanced, solver):
    input_data = create_input_data(parse_opponent_types(MULTI_OPPONENT), TEAM_SIZE)
    selected_team, result, _ = model_balanced.solve(input_data, solver_name="totsubb")
    assert isinstance(selected_team, list)
    assert len(selected_team) == TEAM_SIZE

def test_invalid_opponent():
    with pytest.raises(ValueError, match="Invalid type"):
        parse_opponent_types(INVALID_OPPONENT)

def test_invalid_team_size(model_attack):
    with pytest.raises(ValueError):
        create_input_data(parse_opponent_types(SINGLE_OPPONENT), 0)

def test_solution_consistency(model_attack, model_defense, model_balanced, solver):
    input_data = create_input_data(parse_opponent_types(SINGLE_OPPONENT), TEAM_SIZE)
    
    attack_team, _, _ = model_attack.solve(input_data, solver_name="totsubb")
    defense_team, _, _ = model_defense.solve(input_data, solver_name="totsubb")
    balanced_team, _, _ = model_balanced.solve(input_data, solver_name="totsubb")

    assert attack_team != defense_team  # Attack and Defense should differ
    assert attack_team != balanced_team or defense_team != balanced_team  # Balanced should be distinct

if __name__ == "__main__":
    pytest.main()
