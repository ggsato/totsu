import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyomo.environ import SolverFactory
from ..utils.logger import totsu_logger

class TableauVisualizer:
    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
        self.history = []
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    def solve_model(self):
        """Solves the model and stores tableau history for visualization."""
        try:
            solution = self.solver.solve(self.model)
            self.history = self.solver.get_history()
            self.setup_dash_layout()
        except Exception as e:
            totsu_logger.error(f"Error solving model: {e}")
            raise  # Re-raise the exception after logging

    def show_tableau_visualization(self):
        """Launches the Dash app for tableau visualization."""
        self.app.run_server(debug=True)

    def setup_dash_layout(self):
        """Configures the Dash app layout and callbacks."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row(
                dbc.Col(
                    html.H2("Simplex Tableau Visualization"),
                    className="text-center my-2"
                )
            ),
            # Iteration Slider, Buttons, and Objective Progress Chart
            dbc.Row([
                dbc.Col([
                    html.Label("Select Iteration:", className="font-weight-bold"),
                    dbc.InputGroup([
                        dbc.Button("Previous", id="prev-button", n_clicks=0),
                        dbc.Button("Next", id="next-button", n_clicks=0),
                    ], size="sm", className="mb-2"),
                    dcc.Slider(
                        id="iteration-slider",
                        min=0,
                        max=max(len(self.history) - 1, 0),
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    dbc.Button(
                        "Toggle Explanation",
                        id="explanation-toggle",
                        color="primary",
                        size="sm",
                        className="mt-2",
                        n_clicks=0
                    ),
                    dbc.Collapse(
                        html.Div(id="iteration-explanation"),
                        id='explanation-collapse',
                        is_open=False
                    )
                ], width=4),
                dbc.Col(
                    dcc.Graph(id="objective-progress-chart", config={'displayModeBar': False}, style={'height': '250px'}),
                    width=8
                )
            ], align="center", className="my-2"),
            # Combined Figure with All Components
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id="combined-figure", config={'displayModeBar': False}, style={'height': '600px'}),
                    width=12
                ),
                className="my-2"
            )
        ], fluid=True)
        self.register_callbacks()

    def register_callbacks(self):
        """Defines the callbacks for interactive components."""
        @self.app.callback(
            [
                Output("iteration-slider", "value"),
                Output("prev-button", "disabled"),
                Output("next-button", "disabled"),
            ],
            [
                Input("prev-button", "n_clicks"),
                Input("next-button", "n_clicks"),
                Input("iteration-slider", "value")
            ],
            [State("iteration-slider", "value")]
        )
        def update_iteration(n_clicks_prev, n_clicks_next, slider_value, slider_state):
            ctx = dash.callback_context
            if not ctx.triggered:
                return slider_value, True, False
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            total_iterations = len(self.history) - 1
            if button_id == "prev-button":
                new_value = max(slider_state - 1, 0)
            elif button_id == "next-button":
                new_value = min(slider_state + 1, total_iterations)
            else:
                new_value = slider_value

            # Disable buttons appropriately
            disable_prev = new_value == 0
            disable_next = new_value == total_iterations

            return new_value, disable_prev, disable_next

        @self.app.callback(
            [
                Output("objective-progress-chart", "figure"),
                Output("combined-figure", "figure"),
                Output("iteration-explanation", "children"),
            ],
            [Input("iteration-slider", "value")]
        )
        def update_visualizations(selected_iteration):
            return self.update_visualizations(selected_iteration)
        
        @self.app.callback(
            Output("explanation-collapse", "is_open"),
            [Input("explanation-toggle", "n_clicks")],
            [State("explanation-collapse", "is_open")],
        )
        def toggle_explanation(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open

    def update_visualizations(self, selected_iteration):
        """Callback to update all visualizations based on selected iteration."""
        if not self.history:
            return {}, {}, {}
        
        snapshot = self.history[selected_iteration]
        
        totsu_logger.info(f"snapshot at iteration {selected_iteration}\n{snapshot}")
        
        # Update the combined figure with all components
        combined_fig = self.create_combined_figure(snapshot)
        
        return (
            self.create_objective_progress_chart(selected_iteration),
            combined_fig,
            self.update_explanation(snapshot, selected_iteration)
        )

    def create_objective_progress_chart(self, selected_iteration):
        if not self.history:
            return go.Figure()

        iterations = list(range(len(self.history)))
        obj_values_phase1 = []
        obj_values_phase2 = []
        phases = []

        for idx, snap in enumerate(self.history):
            phase = snap.get('phase', 1)  # Default to Phase 1 if not specified
            phases.append(phase)
            if phase == 1:
                obj_values_phase1.append(snap["objective_value"])
                obj_values_phase2.append(None)
            else:
                obj_values_phase1.append(None)
                obj_values_phase2.append(snap["objective_value"])

        fig = go.Figure()

        # Plot Phase 1 Objective Values
        fig.add_trace(go.Scatter(
            x=iterations,
            y=obj_values_phase1,
            mode="lines+markers",
            name="Phase 1 Objective",
            line=dict(color='blue'),
        ))

        # Plot Phase 2 Objective Values
        fig.add_trace(go.Scatter(
            x=iterations,
            y=obj_values_phase2,
            mode="lines+markers",
            name="Phase 2 Objective",
            line=dict(color='green'),
        ))

        # Highlight current iteration
        fig.add_vline(x=selected_iteration, line_dash="dash", line_color="red")

        # Indicate Phase Transition
        if 2 in phases:
            phase_transition_idx = phases.index(2)
            fig.add_vline(
                x=phase_transition_idx - 0.5,
                line_dash="dot",
                line_color="black",
                annotation_text="Phase 2 Begins",
                annotation_position="top left"
            )

        fig.update_layout(
            title="Objective Value Progress",
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            legend_title="Phases"
        )
        return fig

    def create_combined_figure(self, snapshot):
        # Extract necessary data
        objective_row = snapshot["objective_row"]
        entering_var_idx = snapshot["entering_var_idx"]
        variables = snapshot["variable_names"]
        tableau = snapshot["tableau"]
        pivot_row = snapshot["pivot_row"]
        pivot_col = snapshot["pivot_col"]
        basic_var_indices = snapshot['basis_vars']
        ratios = snapshot["ratios"]  # List of (ratio, row_index)
        phase = snapshot["phase"]

        # -- Basic Variable Names --
        basic_var_names = [variables[idx] for idx in basic_var_indices]

        # -- Entering Variable Bar Chart Data --
        colors_entering = ['green' if coef >= 0 else 'red' for coef in objective_row]

        # -- Leaving Variable Bar Chart Data --
        # Prepare ratios aligned with basic variables
        ratio_dict = {row_idx: ratio for ratio, row_idx in ratios}
        aligned_ratios = [ratio_dict.get(i, None) for i in range(len(basic_var_names))]

        # Reverse data to match display order if necessary
        basic_var_names_reversed = basic_var_names[::-1]
        aligned_ratios_reversed = aligned_ratios[::-1]
        colors_leaving = ['#1f77b4' if i == pivot_row else '#cccccc' for i in range(len(basic_var_names))]
        colors_leaving_reversed = colors_leaving[::-1]

        # Create a subplot figure with 2 columns and 2 rows
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=False,
            shared_yaxes=False,
            row_heights=[0.3, 0.7],
            column_widths=[0.25, 0.75],
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
            specs=[
                [None, {"type": "xy"}],
                [{"type": "xy"}, {"type": "heatmap"}]
            ]
        )

        # -- Entering Variable Bar Chart --
        fig.add_trace(
            go.Bar(
                x=variables,
                y=objective_row,
                marker_color=colors_entering,
                name="Objective Coefficients"
            ),
            row=1, col=2
        )

        # Highlight entering variable
        if entering_var_idx is not None:
            fig.add_trace(
                go.Scatter(
                    x=[variables[entering_var_idx]],
                    y=[objective_row[entering_var_idx]],
                    mode="markers+text",
                    marker=dict(size=12, color="blue", symbol='diamond'),
                    text=["Entering Variable"],
                    textposition="top center",
                    name="Entering Variable"
                ),
                row=1, col=2
            )

        # -- Leaving Variable Bar Chart --
        fig.add_trace(
            go.Bar(
                x=aligned_ratios_reversed,
                y=basic_var_names_reversed,
                orientation="h",
                marker_color=colors_leaving_reversed,
                name="Minimum Ratios"
            ),
            row=2, col=1
        )

        # Highlight leaving variable
        if pivot_row is not None:
            idx_in_reversed = len(basic_var_names) - 1 - pivot_row
            fig.add_trace(
                go.Scatter(
                    x=[aligned_ratios_reversed[idx_in_reversed]],
                    y=[basic_var_names_reversed[idx_in_reversed]],
                    mode="markers+text",
                    marker=dict(size=12, color="red", symbol='diamond'),
                    text=["Leaving Variable"],
                    textposition="middle right",
                    name="Leaving Variable"
                ),
                row=2, col=1
            )

        # -- Tableau Heatmap --
        fig.add_trace(
            go.Heatmap(
                z=tableau,
                x=variables,
                y=basic_var_names_reversed,
                colorscale="Viridis",
                text=tableau.round(2),
                hoverinfo="text",
                colorbar=dict(title="Values"),
                showscale=True
            ),
            row=2, col=2
        )

        # Highlight the pivot element
        if pivot_row is not None and pivot_col is not None:
            idx_in_reversed = len(basic_var_names) - 1 - pivot_row
            fig.add_trace(
                go.Scatter(
                    x=[variables[pivot_col]],
                    y=[basic_var_names_reversed[idx_in_reversed]],
                    mode="markers+text",
                    marker=dict(size=14, color="yellow", symbol='diamond'),
                    text=["Pivot"],
                    textposition="middle center",
                    name="Pivot Element"
                ),
                row=2, col=2
            )

        # -- Update Layout --
        fig.update_layout(
            height=600,
            showlegend=False,
            xaxis=dict(title="Variables", showticklabels=False),
            xaxis2=dict(title="Ratios"),
            xaxis3=dict(title="Variables", tickangle=-45),
            yaxis=dict(title="Objective Coefficients")
        )

        # Update y-axes configurations
        fig.update_yaxes(
            title_text="Basic Variables",
            type='category',
            categoryorder='array',
            categoryarray=basic_var_names_reversed,
            showticklabels=True,
            row=2, col=1
        )

        fig.update_yaxes(
            type='category',
            categoryorder='array',
            categoryarray=basic_var_names_reversed,
            showticklabels=False,
            row=2, col=2
        )

        return fig

    def update_explanation(self, snapshot, selected_iteration):
        entering_var_idx = snapshot.get('entering_var_idx')
        pivot_row = snapshot.get('pivot_row')
        variables = snapshot['variable_names']
        phase = snapshot.get('phase', 1)  # Default to Phase 1 if not specified

        # Get entering variable name
        entering_var = variables[entering_var_idx] if entering_var_idx is not None else "None"

        # Get leaving variable name from previous snapshot
        if selected_iteration > 0 and pivot_row is not None:
            prev_snapshot = self.history[selected_iteration - 1]
            prev_basic_var_indices = prev_snapshot['basis_vars']
            prev_basic_var_names = [variables[idx] for idx in prev_basic_var_indices]
            leaving_var = prev_basic_var_names[pivot_row]
        else:
            leaving_var = "None"

        explanation = [
            html.H4(f"Iteration {selected_iteration} Explanation (Phase {phase}):"),
            html.P([
                html.Strong("Entering Variable: "),
                f"{entering_var} ",
                "(Selected to improve the objective function by entering the basis)"
            ]) if entering_var != "None" else html.P("No entering variable. Optimality reached."),
            html.P([
                html.Strong("Leaving Variable: "),
                f"{leaving_var} ",
                "(Selected to maintain feasibility by leaving the basis)"
            ]) if leaving_var != "None" else html.P("No leaving variable. Solution is infeasible."),
            # Replace the placeholder with specific explanations
            html.P(self.get_phase_explanation(phase))
        ]
        return explanation
    
    def get_phase_explanation(self, phase):
        if phase == 1:
            return ("Phase 1 focuses on finding an initial basic feasible solution. "
                    "Artificial variables are introduced to achieve feasibility. "
                    "The objective is to minimize the sum of artificial variables. "
                    "Once a feasible solution is found, the algorithm proceeds to Phase 2.")
        elif phase == 2:
            return ("Phase 2 optimizes the original objective function using the basic feasible solution "
                    "obtained from Phase 1. The algorithm performs pivot operations to improve the objective value "
                    "until optimality is reached.")
        else:
            return "Unknown phase."
