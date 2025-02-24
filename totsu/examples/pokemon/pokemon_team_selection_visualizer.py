import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash.dash_table as dt
import argparse
from .pokemon_team_selection import POKEMON_TYPES, PokemonTeamSelection, create_input_data
import dash.dash_table as dt
import warnings

# Japanese translations for Pokémon types (simplified for younger children)
POKEMON_TYPES_JA = ["ノーマル", "ほのお", "みず", "くさ", "でんき", "かくとう", "どく", "じめん", "ひこう", 
                     "エスパー", "むし", "いわ", "ゴースト", "こおり", "ドラゴン", "あく", "はがね", "フェアリー"]

# Dash application setup
app = dash.Dash(__name__, suppress_callback_exceptions=True)

def layout():
    return html.Div([
        html.H1("Pokémon Team Selection - Step 3 / ポケモンチームえらび"),
        html.Label("Select opponent Pokémon types / あいてのポケモンのタイプをえらぶ"),
        
        html.Div([
            dcc.Dropdown(
                id="slot-1",
                options=[{"label": f"{en} ({ja})", "value": en} for en, ja in zip(POKEMON_TYPES, POKEMON_TYPES_JA)],
                multi=True,
                placeholder="Select up to 2 types for Slot 1 / スロット1のタイプをえらぶ"
            ),
            dcc.Dropdown(
                id="slot-2",
                options=[{"label": f"{en} ({ja})", "value": en} for en, ja in zip(POKEMON_TYPES, POKEMON_TYPES_JA)],
                multi=True,
                placeholder="Select up to 2 types for Slot 2 / スロット2のタイプをえらぶ"
            ),
            dcc.Dropdown(
                id="slot-3",
                options=[{"label": f"{en} ({ja})", "value": en} for en, ja in zip(POKEMON_TYPES, POKEMON_TYPES_JA)],
                multi=True,
                placeholder="Select up to 2 types for Slot 3 / スロット3のタイプをえらぶ"
            ),
        ], className="slot-container"),
        
        html.Div(id="warning-message", style={"color": "red"}),
        
        html.Div([
            html.Label("Team Size / チームのかず"),
            dcc.Input(id="team-size", type="number", value=3, min=1, max=6),
        ], className="team-size-container"),
        
        html.Div([
            html.Label("Objective Function / もくひょう"),
            dcc.Dropdown(
                id="objective-dropdown",
                options=[
                    {"label": "Maximize Attack / こうげきがさいだい", "value": "attack"},
                    {"label": "Maximize Defense / ぼうぎょがさいだい", "value": "defense"},
                    {"label": "Balanced Strategy / バランス", "value": "balanced"}
                ],
                value="attack",
                clearable=False
            ),
        ], className="objective-container"),
        
        html.Button("Run Optimization / さいてきかする", id="run-optimization", n_clicks=0),
        
        # Stacked bar chart replacing the previous two graphs
        dcc.Graph(id="stacked-effectiveness-chart"),

        # Matchup Details Table
        html.H3("Matchup Details / たいしょうひょう"),
        dt.DataTable(
            id="matchup-table",
            columns=[
                {"name": "Opponent Type", "id": "opponent"},
                {"name": "Selected Pokémon 1", "id": "pokemon1"},
                {"name": "Selected Pokémon 2", "id": "pokemon2"},
                {"name": "Selected Pokémon 3", "id": "pokemon3"}
            ],
            data=[],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center"},
        ),
        # Store selected Pokémon (hidden)
        dcc.Store(id="selected-pokemon", data=[]),
    ])

app.layout = layout

# **Callback 1: Show Stacked Effectiveness Chart when Button is Pressed**
@app.callback(
    Output("stacked-effectiveness-chart", "figure"),
    Input("run-optimization", "n_clicks"),
    [State("slot-1", "value"),
     State("slot-2", "value"),
     State("slot-3", "value"),
     State("team-size", "value"),
     State("objective-dropdown", "value")]
)
def show_stacked_chart(n_clicks, slot1, slot2, slot3, team_size, objective):
    if n_clicks == 0:
        return go.Figure()  # Return empty figure initially

    # **Process Opponent Pokémon Selections**
    def process_slot(slot):
        if not slot:
            return None
        if len(slot) > 2:
            slot = slot[:2]  # Limit to 2 types
        return tuple(slot) if len(slot) == 2 else (slot[0], None)

    opponent_types = [process_slot(slot) for slot in [slot1, slot2, slot3] if process_slot(slot) is not None]
    if not opponent_types:
        return go.Figure()

    # **Create Model Instance (Without Solving)**
    input_data = create_input_data(opponent_types, team_size)
    instance = PokemonTeamSelection(objective).model.create_instance(input_data)

    # **Compute Attack & Defense Effectiveness**
    attack_scores = {ptype: sum(instance.A[ptype, opp] for opp in opponent_types) for ptype in instance.T}
    defense_scores = {ptype: sum(instance.A[opp[0], (ptype, None)] for opp in opponent_types) for ptype in instance.T}

    # **Sort Pokémon Types Based on Objective**
    if objective == "attack":
        sorted_types = sorted(instance.T, key=lambda x: -attack_scores.get(x, 0))
    elif objective == "defense":
        sorted_types = sorted(instance.T, key=lambda x: defense_scores.get(x, 0))
    else:  # Balanced objective
        sorted_types = sorted(instance.T, key=lambda x: defense_scores.get(x, 0) - attack_scores.get(x, 0))

    sorted_types = [str(t) for t in sorted_types]

    # **Create Stacked Bar Chart**
    attack_traces = []
    defense_traces = []

    for opp in opponent_types:
        attack_traces.append(go.Bar(
            x=sorted_types,
            y=[instance.A[t, opp] for t in sorted_types],
            name=f"Attack vs {opp}",
        ))

        defense_traces.append(go.Bar(
            x=sorted_types,
            y=[-instance.A[opp[0], (t, None)] for t in sorted_types],
            name=f"Defense vs {opp}",
        ))

    fig = go.Figure()
    for trace in attack_traces:
        fig.add_trace(trace)
    for trace in defense_traces:
        fig.add_trace(trace)

    fig.update_layout(
        title="Stacked Effectiveness Chart (Sorted by Objective)",
        barmode="relative",
        xaxis=dict(title="Pokémon Type"),
        yaxis=dict(title="Effectiveness Score"),
        legend_title="Effectiveness Type"
    )

    return fig

# **Callback 2: Handle Pokémon Selection from Stacked Chart**
@app.callback(
    Output("selected-pokemon", "data"),
    Input("stacked-effectiveness-chart", "clickData"),
    State("selected-pokemon", "data")
)
def update_selected_pokemon(click_data, selected_pokemon):
    if not click_data:
        return selected_pokemon  # Keep previous selection if no click event

    clicked_pokemon = click_data["points"][0]["x"]

    # Toggle selection (add/remove clicked Pokémon)
    if clicked_pokemon in selected_pokemon:
        selected_pokemon.remove(clicked_pokemon)
    else:
        if len(selected_pokemon) < 3:
            selected_pokemon.append(clicked_pokemon)
    return selected_pokemon

# **Callback 3: Update Matchup Table Based on Selected Pokémon**
@app.callback(
    [Output("matchup-table", "columns"),
     Output("matchup-table", "data")],
    [Input("selected-pokemon", "data"),
     Input("run-optimization", "n_clicks")],
    [State("slot-1", "value"),
     State("slot-2", "value"),
     State("slot-3", "value"),
     State("team-size", "value"),
     State("objective-dropdown", "value")]
)
def update_matchup_table(selected_pokemon, n_clicks, slot1, slot2, slot3, team_size, objective):
    if n_clicks == 0 or not selected_pokemon:
        return [{"name": "Opponent Type", "id": "opponent"}], []

    # **Process Opponent Pokémon Selections**
    def process_slot(slot):
        if not slot:
            return None
        if len(slot) > 2:
            slot = slot[:2]  # Limit to 2 types
        return tuple(slot) if len(slot) == 2 else (slot[0], None)

    opponent_types = [process_slot(slot) for slot in [slot1, slot2, slot3] if process_slot(slot) is not None]
    if not opponent_types:
        return [{"name": "Opponent Type", "id": "opponent"}], []

    # **Create Model Instance (Without Solving)**
    input_data = create_input_data(opponent_types, team_size)
    instance = PokemonTeamSelection(objective).model.create_instance(input_data)

    # **Define Helper Function for Colored Circles**
    def get_color_circle(value, is_defense=False):
        """Returns color circle based on effectiveness. Defense colors are reversed (lower is better)."""
        if is_defense:  # Defense represents **damage taken**, so flip color logic
            if value <= -1.5:
                return "🔴"  # **Bad Defense (High Damage Taken)**
            elif value >= -0.5:
                return "🟢"  # **Good Defense (Low Damage Taken)**
        else:  # Normal attack effectiveness
            if value >= 1.5:
                return "🟢"  # **Strong Attack**
            elif value <= 0.5:
                return "🔴"  # **Weak Attack**
        return "🟡"  # **Neutral**

    # **Create Matchup Table with Correct Defense Representation**
    matchup_data = []
    for i, opp in enumerate(opponent_types):
        row = {"Opponent Type": str(opp)}
        for pokemon in selected_pokemon:
            attack_value = instance.A[pokemon, opp]  # Normal attack effectiveness
            defense_value = -1 * instance.A[opp[0], (pokemon, None)]

            row[pokemon] = f"🗡️ {get_color_circle(attack_value)} {attack_value:.2f}  🛡️ {get_color_circle(defense_value, is_defense=True)} {defense_value:.2f}"

        matchup_data.append(row)

    # **Table Columns: Opponent, (P1), (P2), (P3)**
    columns = [{"name": "Opponent Type", "id": "Opponent Type"}] + [
        {"name": pokemon, "id": pokemon} for pokemon in selected_pokemon
    ]

    return columns, matchup_data


if __name__ == "__main__":
    app.run_server(debug=True)
