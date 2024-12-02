from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from pyomo.environ import SolverFactory
from ..utils.logger import totsu_logger

class TableauVisualizer:
    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
        self.history = []
        self.app = Dash(__name__)

    def solve_model(self):
        """Solves the model and stores tableau history for visualization."""
        try:
            solution = self.solver.solve(self.model)
            self.history = self.solver.get_history()
            self.setup_dash_layout()
        except Exception as e:
            print(f"Error solving model: {e}")
            return False
        
        return True

    def show_tableau_visualization(self):
        """Launches the Dash app for tableau visualization."""
        self.app.run_server(debug=True)

    def setup_dash_layout(self):
        """Configures the Dash app layout and callbacks."""
        self.app.layout = html.Div([
            html.H1("Simplex Tableau Visualization"),
            html.Div(id="iteration-explanation", style={"margin": "20px 0", "padding": "10px", "background-color": "#f9f9f9"}),
            html.Div([
                html.Label("Select Iteration:"),
                dcc.Slider(
                    id="iteration-slider",
                    min=0,
                    max=max(len(self.history) - 1, 0),
                    step=1,
                    value=0,
                    marks={i: f"Step {i}" for i in range(len(self.history))}
                )
            ], style={"margin": "20px 0"}),

            # Line chart of objective values
            html.Div([dcc.Graph(id="objective-progress-chart")], style={"margin-bottom": "40px"}),

            # Tableau and bar charts
            html.Div([
                html.Div([dcc.Graph(id="tableau-heatmap")], style={"width": "49%", "display": "inline-block"}),
                html.Div([dcc.Graph(id="entering-variable-bar")], style={"width": "49%", "display": "inline-block"})
            ], style={"margin-bottom": "40px"}),

            html.Div([
                html.Div([dcc.Graph(id="leaving-variable-bar")], style={"width": "100%"})
            ])
        ])

        # Define callbacks for interactivity
        self.app.callback(
            [
                Output("objective-progress-chart", "figure"),
                Output("tableau-heatmap", "figure"),
                Output("entering-variable-bar", "figure"),
                Output("leaving-variable-bar", "figure"),
                Output("iteration-explanation", "children"),
            ],
            [Input("iteration-slider", "value")]
        )(self.update_visualizations)

    def update_visualizations(self, selected_iteration):
        """Callback to update all visualizations based on selected iteration."""
        if not self.history:
            return {}, {}, {}, {}, {}

        snapshot = self.history[selected_iteration]

        totsu_logger.debug(f"snapshot at iteration {selected_iteration}\n{snapshot}")

        return (
            self.create_objective_progress_chart(),
            self.create_tableau_heatmap(snapshot),
            self.create_entering_variable_bar(snapshot),
            self.create_leaving_variable_bar(snapshot),
            self.update_explanation(snapshot, selected_iteration)
        )

    def create_objective_progress_chart(self):
        iterations = list(range(len(self.history)))
        obj_values = [snap["objective_value"] for snap in self.history]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=obj_values, mode="lines+markers", name="Objective Value"))
        fig.update_layout(
            title="Objective Value Progress",
            xaxis_title="Iteration",
            yaxis_title="Objective Value"
        )
        return fig

    def create_tableau_heatmap(self, snapshot):
        tableau = snapshot["tableau"]
        pivot_row = snapshot["pivot_row"]
        pivot_col = snapshot["pivot_col"]

        fig = go.Figure(data=go.Heatmap(
            z=tableau,
            colorscale="Viridis",
            text=tableau.round(2),
            hoverinfo="text"
        ))

        if pivot_row is not None and pivot_col is not None:
            fig.add_trace(go.Scatter(
                x=[pivot_col],
                y=[pivot_row],
                mode="markers+text",
                marker=dict(size=14, color="yellow"),
                text=["Pivot"],
                textposition="bottom center",
                name="Pivot Element"
            ))

            fig.update_layout(
                title="Tableau Heatmap with Pivot Element",
                xaxis_title="Variables",
                yaxis_title="Rows"
            )
        return fig

    def create_entering_variable_bar(self, snapshot):
        objective_row = snapshot["objective_row"]
        entering_var_idx = snapshot["entering_var_idx"]
        variables = snapshot["variable_names"]  # Use variable names from snapshot

        fig = go.Figure()
        fig.add_trace(go.Bar(x=variables, y=objective_row, name="Objective Coefficients"))

        if entering_var_idx is not None:
            fig.add_trace(go.Scatter(
                x=[variables[entering_var_idx]],
                y=[objective_row[entering_var_idx]],
                mode="markers+text",
                marker=dict(size=12, color="red"),
                text=["Selected Entering Variable"],
                textposition="top center",
                name="Entering Variable"
            ))

            fig.update_layout(
                title="Entering Variable Selection: Ensures Improvement of the Goal",
                xaxis_title="Variables",
                yaxis_title="Objective Coefficients",
                annotations=[
                    dict(
                        x=variables[entering_var_idx],
                        y=objective_row[entering_var_idx],
                        text=f"Entering: {variables[entering_var_idx]}",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40
                    )
                ]
            )
        return fig

    def create_leaving_variable_bar(self, snapshot):
        ratios = snapshot["ratios"]  # List of (ratio, row_index)
        pivot_row = snapshot["pivot_row"]

        # Extract ratio values and their corresponding basic variables
        ratio_values = [r[0] for r in ratios]
        row_indices = [r[1] for r in ratios]
        basic_vars = [snapshot["variable_names"][i] for i in snapshot["basis_vars"]]

        # Map valid ratios to their respective rows
        aligned_ratios = [ratio_values[row_indices.index(i)] if i in row_indices else float('inf') for i in range(len(basic_vars))]

        fig = go.Figure()
        fig.add_trace(go.Bar(y=basic_vars, x=aligned_ratios, orientation="h", name="Minimum Ratios"))
        # Add pivot row highlighting if applicable
        if pivot_row is not None:
            fig.add_trace(go.Scatter(
                y=[basic_vars[pivot_row]],
                x=[aligned_ratios[pivot_row]],
                mode="markers+text",
                marker=dict(size=12, color="blue"),
                text=["Selected Leaving Variable"],
                textposition="middle right",
                name="Leaving Variable"
            ))
        fig.update_layout(
            title="Leaving Variable Selection: Ensures Feasibility",
            xaxis_title="Ratios (RHS / Coefficients)",
            yaxis_title="Basic Variables",
        )
        return fig

    def update_explanation(self, snapshot, selected_iteration):
        entering_var = f"x{snapshot['entering_var_idx'] + 1}" if snapshot["entering_var_idx"] is not None else "None"
        leaving_var = f"BV{snapshot['pivot_row'] + 1}" if snapshot["pivot_row"] is not None else "None"
        explanation = [
            html.P(f"Iteration {selected_iteration} Explanation:"),
            html.P(f"Entering Variable: {entering_var} (Selected to Improve the Objective)"),
            html.P(f"Leaving Variable: {leaving_var} (Selected to Maintain Feasibility)")
        ]
        return explanation
