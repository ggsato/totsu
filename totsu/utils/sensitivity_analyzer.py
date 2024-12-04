import numpy as np
import plotly.graph_objects as go
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    maximize,
    minimize,
    NonNegativeReals,
    Suffix,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import threading
from ..utils.logger import totsu_logger
from ..utils.model_processor import ModelProcessor

class SensitivityAnalyzer:
    def __init__(self, model, solver):
        self.solver = solver
        self.model = model.clone()
        self.significant_constraints = None
        self.b_original = {}
        self.lp_solution_cache = {}
        self.computation_thread = None
        self.stop_computation = False
        self.computation_data = {
            "B": None,
            "Z": None,
            "progress": 0,
            "completed": False,
            "valid_ranges": {},
        }
        self.is_minimization = ModelProcessor.get_active_objective(model).sense == minimize

    def solve_primal(self):
        # Solve the primal model
        try:
            z_primal, shadow_prices = self.solve_lp(self.model.clone(), {})
        except ValueError as e:
            totsu_logger.info("Primal model is not optimal", e)
            return False  # Indicate failure

        # Identify significant constraints
        self.significant_constraints = self.get_significant_constraints(shadow_prices)

        if len(self.significant_constraints) == 1:
            totsu_logger.error(f"More than one constraints are required for this visualization")
            return False

        # Get original RHS values for significant constraints
        for constr_name in self.significant_constraints:
            constraint = getattr(self.model, constr_name)
            if constraint.has_ub():
                self.b_original[constr_name] = constraint.upper()
            elif constraint.has_lb():
                self.b_original[constr_name] = constraint.lower()
            else:
                self.b_original[constr_name] = value(constraint.body)

        return True  # Indicate success

    def solve_lp(self, model, rhs_adjustments):
        # Adjust RHS values of the constraints
        for constr_name, new_rhs in rhs_adjustments.items():
            constraint = getattr(model, constr_name)
            # Update the constraint with the new RHS value
            expr = constraint.body
            if constraint.has_lb() and constraint.has_ub():
                constraint.set_value(expr >= constraint.lower())
                constraint.set_value(expr <= new_rhs)
            elif constraint.has_lb():
                constraint.set_value(expr >= constraint.lower())
            elif constraint.has_ub():
                constraint.set_value(expr <= new_rhs)
            else:
                constraint.set_value(expr == new_rhs)

        # Declare dual suffixes to get shadow prices
        if hasattr(model, 'dual'):
            model.del_component(model.dual)
        model.dual = Suffix(direction=Suffix.IMPORT)

        # Solve the model using the given solver
        result = self.solver.solve(model, tee=False)

        # Check solver status
        if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
            raise ValueError("Solver did not find an optimal solution.")

        # Get the objective value
        objective_value = result.solver.objective

        # Extract shadow prices (dual values)
        shadow_prices = {}
        for c in model.component_objects(Constraint, active=True):
            for index in c:
                constr = c[index]
                dual_value = model.dual.get(constr, 0.0)
                shadow_prices[constr.name] = dual_value

        # Store result in cache
        cache_key = tuple(sorted(rhs_adjustments.items()))
        self.lp_solution_cache[cache_key] = (objective_value, shadow_prices)

        return objective_value, shadow_prices


    def get_significant_constraints(self, shadow_prices, num_constraints=2):
        # Sort constraints based on the absolute value of their dual variables
        sorted_constraints = sorted(shadow_prices.items(), key=lambda x: abs(x[1]), reverse=True)
        # Return the names of the top constraints
        significant_constraints = [name for name, value in sorted_constraints[:num_constraints]]
        return significant_constraints

    def compute_valid_range(self, model, constraint_name, b_original_value, other_constraints, b_original_values, delta=1.0):
        # Initialize variables
        allowable_increase = 0.0
        allowable_decrease = 0.0

        # Get the original shadow price and objective value
        rhs_adjustments = {name: b_original_values[name] for name in other_constraints}
        rhs_adjustments[constraint_name] = b_original_value
        z_original, shadow_prices_original = self.solve_lp(model.clone(), rhs_adjustments)
        original_shadow_price = shadow_prices_original[constraint_name]
        original_objective = z_original

        # Check increase direction
        b = b_original_value
        iteration = 0
        max_iterations = 1000  # Prevent infinite loops
        while iteration < max_iterations:
            iteration += 1
            b += delta
            rhs_adjustments[constraint_name] = b
            try:
                z_new, shadow_prices_new = self.solve_lp(model.clone(), rhs_adjustments)
                new_shadow_price = shadow_prices_new.get(constraint_name, 0.0)
                expected_objective = original_objective + original_shadow_price * (b - b_original_value)
                if (abs(new_shadow_price - original_shadow_price) > 1e-5 or
                    abs(z_new - expected_objective) > 1e-2):
                    totsu_logger.debug(f"no more points for {constraint_name} in the increase direction beyond {b} at iteration {iteration}")
                    totsu_logger.debug(f"original objective = {original_objective}, original shadow price = {original_shadow_price}, b original value = {b_original_value}")
                    totsu_logger.debug(f"z_new = {z_new}, new shadow price = {new_shadow_price}, expected objective = {expected_objective}")
                    break
                allowable_increase = b - b_original_value
            except:
                raise
        if iteration >= max_iterations:
            totsu_logger.warning(f"gave up on calculating valid ranges in the increase direction")

        # Check decrease direction
        b = b_original_value
        iteration = 0
        while iteration < max_iterations and b > 0:
            iteration += 1
            b -= delta
            if b < 0:
                b = 0
            rhs_adjustments[constraint_name] = b
            try:
                z_new, shadow_prices_new = self.solve_lp(model.clone(), rhs_adjustments)
                new_shadow_price = shadow_prices_new.get(constraint_name, 0.0)
                expected_objective = original_objective + original_shadow_price * (b - b_original_value)
                if (abs(new_shadow_price - original_shadow_price) > 1e-5 or
                    abs(z_new - expected_objective) > 1e-2):
                    totsu_logger.debug(f"no more points for {constraint_name} in the decrease direction beyond {b} at iteration {iteration}")
                    totsu_logger.debug(f"original objective = {original_objective}, original shadow price = {original_shadow_price}, b original value = {b_original_value}")
                    totsu_logger.debug(f"z_new = {z_new}, new shadow price = {new_shadow_price}, expected objective = {expected_objective}")
                    break
                allowable_decrease = b_original_value - b
            except:
                raise
        if iteration >= max_iterations:
            totsu_logger.warning(f"gave up on calculating valid ranges in the decrease direction")

        return allowable_increase, allowable_decrease

    def computation_function(self, significant_constraints, b_original, b_range_percentages):
        # Clear the cache at the start of the computation
        self.lp_solution_cache = {}

        # Calculate absolute ranges based on percentage changes
        b_ranges = {}
        for constr_name in significant_constraints:
            b_min = b_original[constr_name] * (1 + b_range_percentages[constr_name][0] / 100)
            b_max = b_original[constr_name] * (1 + b_range_percentages[constr_name][1] / 100)
            b_ranges[constr_name] = np.linspace(b_min, b_max, 20)

        # Create meshgrid for the two constraints
        B = np.meshgrid(*b_ranges.values())
        Z = np.empty(B[0].shape)

        total_points = B[0].size
        processed_points = 0

        # Map constraint names to their index in B
        constr_indices = {name: idx for idx, name in enumerate(significant_constraints)}

        # Iterate over the grid and solve LP at each point
        for idx in np.ndindex(B[0].shape):
            if self.stop_computation:
                totsu_logger.info("Computation stopped by user.")
                self.computation_data["completed"] = True
                return
            rhs_adjustments = {}
            for constr_name in significant_constraints:
                rhs_adjustments[constr_name] = B[constr_indices[constr_name]][idx]
            try:
                Z_value, _ = self.solve_lp(self.model.clone(), rhs_adjustments)
                Z[idx] = Z_value
            except:
                Z[idx] = np.nan  # Infeasible solution
            processed_points += 1
            self.computation_data["progress"] = processed_points / total_points * 100

        # Update computation_data with B and Z
        self.computation_data["B"] = B
        self.computation_data["Z"] = Z

        # Compute valid ranges for each significant constraint
        for constr_name in significant_constraints:
            allowable_increase, allowable_decrease = self.compute_valid_range(
                self.model, constr_name, b_original[constr_name], significant_constraints, b_original
            )
            self.computation_data['valid_ranges'][constr_name] = [
                b_original[constr_name] - allowable_decrease,
                b_original[constr_name] + allowable_increase
            ]

        totsu_logger.info("Computation completed.")
        self.computation_data["completed"] = True

    def compute_ridge_line(self, x_plot_min, x_plot_max, y_plot_min, y_plot_max, num_steps=100, step_size=1.0):
        x_constr, y_constr = self.significant_constraints

        def trace_path(x_start, y_start, step_size, direction_multiplier):
            x_values = [x_start]
            y_values = [y_start]
            z_values = []

            x_current = x_start
            y_current = y_start

            for _ in range(num_steps):
                # Solve LP at the current point
                try:
                    z_current, shadow_prices = self.solve_lp(self.model.clone(), {x_constr: x_current, y_constr: y_current})
                    z_values.append(z_current)
                except:
                    break  # Cannot solve LP at this point

                # Get the shadow prices
                shadow_price_x = shadow_prices.get(x_constr, 0.0)
                shadow_price_y = shadow_prices.get(y_constr, 0.0)

                # Define the direction vector based on shadow prices
                gradient = np.array([shadow_price_x, shadow_price_y])
                if self.is_minimization:
                    gradient = -gradient

                norm = np.linalg.norm(gradient)
                if norm == 0:
                    break  # Zero gradient

                direction = direction_multiplier * (gradient / norm)

                # Take a step in the direction
                x_next = x_current + step_size * direction[0]
                y_next = y_current + step_size * direction[1]

                # Check plotting area bounds
                if not (x_plot_min <= x_next <= x_plot_max):
                    break
                if not (y_plot_min <= y_next <= y_plot_max):
                    break

                x_current = x_next
                y_current = y_next

                x_values.append(x_current)
                y_values.append(y_current)

            return x_values, y_values, z_values

        # Start from the original point
        x_start = self.b_original[x_constr]
        y_start = self.b_original[y_constr]

        # Trace in the positive direction
        x_vals_pos, y_vals_pos, z_vals_pos = trace_path(x_start, y_start, step_size, direction_multiplier=1)

        # Trace in the negative direction
        x_vals_neg, y_vals_neg, z_vals_neg = trace_path(x_start, y_start, step_size, direction_multiplier=-1)

        # Combine the paths (excluding the starting point in one to avoid duplication)
        x_values = x_vals_neg[::-1][:-1] + x_vals_pos
        y_values = y_vals_neg[::-1][:-1] + y_vals_pos
        z_values = z_vals_neg[::-1][:-1] + z_vals_pos

        if len(x_values) < 2:
            return None, None, None, "Too few points"

        return x_values, y_values, z_values, None

    def show_analyzer(self):
        app = dash.Dash(__name__)

        # Build constraint sliders
        constraint_sliders = []
        for constr_name in self.significant_constraints:
            slider_id = {'type': 'range-slider', 'index': constr_name}
            constraint_sliders.append(html.Label(f"{constr_name} Change (%)"))
            constraint_sliders.append(
                dcc.RangeSlider(
                    id=slider_id,
                    min=-50,
                    max=50,
                    step=1,
                    value=[-20, 20],
                    marks={i: f'{i}%' for i in range(-50, 51, 10)},
                )
            )

        # Create initial figure
        initial_fig = self.create_initial_figure()

        app.layout = html.Div(
            [
                html.H1("LP Sensitivity Analysis Visualization"),
                html.Div(constraint_sliders, style={'width': '80%', 'margin': 'auto'}),
                html.Button("Start Exploration", id="start-button", n_clicks=0),
                html.Button("Stop Exploration", id="stop-button", n_clicks=0),
                dcc.Graph(id="lp-graph", figure=initial_fig),
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0, disabled=True),
                html.Div(id="progress-output"),
                dcc.Store(id='computation-status', data={'running': False}),
            ]
        )

        # Callbacks are defined within this method to access 'self'

        @app.callback(
            Output("progress-output", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_progress(n):
            if self.computation_thread is not None and self.computation_thread.is_alive():
                progress = self.computation_data.get("progress", 0)
                return f"Computation in progress: {progress:.1f}% completed."
            elif self.computation_data.get("completed", False):
                return "Computation completed."
            else:
                return ""

        @app.callback(
            Output("lp-graph", "figure"),
            [Input("interval-component", "n_intervals"),
            Input("computation-status", "data")],
            [State("lp-graph", "figure")]
        )
        def update_graph(n, computation_status, current_fig):
            if not computation_status.get('running', False) and self.computation_data.get("completed", False):
                fig = go.Figure(current_fig)

                # Unpack significant constraints
                x_constr, y_constr = self.significant_constraints

                # Define constr_indices here
                constr_indices = {name: idx for idx, name in enumerate(self.significant_constraints)}

                # Extract B and Z from computation_data
                B = self.computation_data["B"]
                Z = self.computation_data["Z"]
                x_values = B[constr_indices[x_constr]]
                y_values = B[constr_indices[y_constr]]

                # Clear existing data
                fig.data = []

                # Add the surface plot
                surface = go.Surface(
                    x=x_values,
                    y=y_values,
                    z=Z,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Objective Value"),
                    name="Objective Value Surface",
                    opacity=0.8,  # Set opacity to 80%
                    hovertemplate=f"{x_constr} RHS: %{{x:.1f}}<br>"
                                f"{y_constr} RHS: %{{y:.1f}}<br>"
                                f"Z: %{{z:.2f}}<extra></extra>",
                )
                fig.add_trace(surface)

                # Add the initial ridge line (empty data)
                ridge_line = go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode='lines',
                    name='Ridge Line',
                    line=dict(color='green', width=8),
                    showlegend=True,
                )
                fig.add_trace(ridge_line)

                # Add the initial marker (empty data)
                ridge_marker = go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode='markers',
                    name='Ridge Marker',
                    marker=dict(size=2, color='green'),
                    showlegend=False,
                )
                fig.add_trace(ridge_marker)

                # Update axis ranges based on data
                x_min = np.nanmin(x_values)
                x_max = np.nanmax(x_values)
                y_min = np.nanmin(y_values)
                y_max = np.nanmax(y_values)

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title=f"{x_constr} RHS Value", range=[x_min, x_max]),
                        yaxis=dict(title=f"{y_constr} RHS Value", range=[y_min, y_max]),
                        zaxis=dict(title="Objective Value (Z)"),
                    ),
                    uirevision='data_changed',  # Preserve user interactions
                )

                # Original RHS values
                x_original = self.b_original[x_constr]
                y_original = self.b_original[y_constr]
                rhs_adjustments = {x_constr: x_original, y_constr: y_original}
                Z_original, _ = self.solve_lp(self.model.clone(), rhs_adjustments)  # Use 'model' instead of 'primal_model'

                # Plot valid range for x_constr
                valid_range_x = self.computation_data['valid_ranges'].get(x_constr)
                if valid_range_x and valid_range_x[0] != valid_range_x[1]:
                    x_line = np.linspace(valid_range_x[0], valid_range_x[1], 100)
                    y_const = y_original
                    Z_x_line = []
                    for x_val in x_line:
                        rhs_adjustments = {x_constr: x_val, y_constr: y_const}
                        try:
                            z, _ = self.solve_lp(self.model.clone(), rhs_adjustments)
                            Z_x_line.append(z)
                        except:
                            Z_x_line.append(np.nan)
                    line = go.Scatter3d(
                        x=x_line,
                        y=[y_const]*len(x_line),
                        z=Z_x_line,
                        mode='lines',
                        name=f'Valid {x_constr} Range',
                        line=dict(color='red', width=4),
                        hovertemplate=f"{x_constr} RHS: %{{x:.1f}}<br>{y_constr} RHS: %{{y}}<br>Z: %{{z:.2f}}<extra></extra>",
                    )
                    fig.add_trace(line)
                else:
                    totsu_logger.info(f"No valid range for {x_constr} to plot.")

                # Plot valid range for y_constr
                valid_range_y = self.computation_data['valid_ranges'].get(y_constr)
                if valid_range_y and valid_range_y[0] != valid_range_y[1]:
                    y_line = np.linspace(valid_range_y[0], valid_range_y[1], 100)
                    x_const = x_original
                    Z_y_line = []
                    for y_val in y_line:
                        rhs_adjustments = {x_constr: x_const, y_constr: y_val}
                        try:
                            z, _ = self.solve_lp(self.model.clone(), rhs_adjustments)
                            Z_y_line.append(z)
                        except:
                            Z_y_line.append(np.nan)
                    line = go.Scatter3d(
                        x=[x_const]*len(y_line),
                        y=y_line,
                        z=Z_y_line,
                        mode='lines',
                        name=f'Valid {y_constr} Range',
                        line=dict(color='blue', width=4),
                        hovertemplate=f"{x_constr} RHS: %{{x}}<br>{y_constr} RHS: %{{y:.1f}}<br>Z: %{{z:.2f}}<extra></extra>",
                    )
                    fig.add_trace(line)
                else:
                    totsu_logger.info(f"No valid range for {y_constr} to plot.")

                # Optimal point (original capacities)
                optimal_point = go.Scatter3d(
                    x=[x_original],
                    y=[y_original],
                    z=[Z_original],
                    mode='markers+text',
                    name='Optimal Solution',
                    marker=dict(color='black', size=6, symbol='circle'),
                    text=['Optimal Solution'],
                    textposition='top center',
                    hovertemplate=f"Optimal Solution<br>{x_constr} RHS: %{{x}}<br>{y_constr} RHS: %{{y}}<br>Z: %{{z}}<extra></extra>"
                )
                fig.add_trace(optimal_point)

                # Compute and plot the ridge/valley line
                x_ridge, y_ridge, z_ridge, ridge_error = self.compute_ridge_line(x_min, x_max, y_min, y_max)
                if ridge_error is None:
                    # Create frames for the animation
                    frames = []
                    for i in range(len(x_ridge)):
                        frame = go.Frame(
                            data=[
                                # Update ridge line (trace index 1)
                                dict(
                                    x=x_ridge[:i + 1],
                                    y=y_ridge[:i + 1],
                                    z=z_ridge[:i + 1],
                                    mode='lines',
                                    line=dict(color='green', width=8),
                                    type='scatter3d',
                                ),
                                # Update marker (trace index 2)
                                dict(
                                    x=[x_ridge[i]],
                                    y=[y_ridge[i]],
                                    z=[z_ridge[i]],
                                    mode='markers',
                                    marker=dict(size=2, color='green'),
                                    type='scatter3d',
                                )
                            ],
                            traces=[1, 2],  # Indices of the traces being updated
                            name=str(i)
                        )
                        frames.append(frame)

                    fig.frames = frames

                    fig.update_layout(
                        updatemenus=[
                            dict(
                                type='buttons',
                                buttons=[
                                    dict(label='Play',
                                        method='animate',
                                        args=[None, {'frame': {'duration': 100, 'redraw': True},
                                                    'fromcurrent': True}]),
                                    dict(label='Pause',
                                        method='animate',
                                        args=[[None], {'frame': {'duration': 0, 'redraw': True},
                                                        'mode': 'immediate',
                                                        'transition': {'duration': 0}}])
                                ],
                                showactive=False,
                            )
                        ],
                        scene=dict(
                            xaxis=dict(title=f"{x_constr} RHS Value", range=[x_min, x_max]),
                            yaxis=dict(title=f"{y_constr} RHS Value", range=[y_min, y_max]),
                            zaxis=dict(title="Objective Value (Z)"),
                        ),
                        legend=dict(x=0.7, y=0.9),
                    )
                else:
                    totsu_logger.info(f"No ridge line to plot because {ridge_error}")

                return fig
            else:
                raise dash.exceptions.PreventUpdate

        @app.callback(
            Output("computation-status", "data"),
            [Input("start-button", "n_clicks"),
            Input("stop-button", "n_clicks"),
            Input("interval-component", "n_intervals")],
            [State({'type': 'range-slider', 'index': ALL}, 'value'),
            State('computation-status', 'data')]
        )
        def control_computation(start_clicks, stop_clicks, n_intervals, slider_values, computation_status):
            ctx = dash.callback_context

            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            else:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "start-button":
                if self.computation_thread is None or not self.computation_thread.is_alive():
                    self.stop_computation = False
                    self.computation_data["B"] = None
                    self.computation_data["Z"] = None
                    self.computation_data["progress"] = 0
                    self.computation_data["completed"] = False
                    self.computation_data["valid_ranges"] = {}
                    # Build a dictionary of percentage ranges for each constraint
                    b_range_percentages = {}
                    # Retrieve the constraint names from the slider IDs
                    for i, constr_name in enumerate(self.significant_constraints):
                        b_range_percentages[constr_name] = slider_values[i]
                    self.computation_thread = threading.Thread(
                        target=self.computation_function,
                        args=(self.significant_constraints, self.b_original, b_range_percentages)
                    )
                    self.computation_thread.start()
                    return {'running': True}
            elif button_id == "stop-button":
                self.stop_computation = True
                return {'running': False}
            elif button_id == "interval-component" and computation_status['running']:
                if self.computation_data.get("completed", False):
                    return {'running': False}
            return dash.no_update

        @app.callback(
            Output("interval-component", "disabled"),
            [Input("computation-status", "data")],
        )
        def update_interval(computation_status):
            return not computation_status.get('running', False)

        app.run_server(debug=True)

    def create_initial_figure(self):
        x_constr, y_constr = self.significant_constraints

        fig = go.Figure()
        fig.update_layout(
            title="Objective Value Exploration",
            scene=dict(
                xaxis=dict(title=f"{x_constr} RHS Value"),
                yaxis=dict(title=f"{y_constr} RHS Value"),
                zaxis=dict(title="Objective Value (Z)"),
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(x=0.7, y=0.9),
            uirevision='data_changed',  # Preserve user interactions
        )
        return fig
