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
from pyomo.core import inequality
from pyomo.core.base.constraint import ConstraintData, ConstraintList
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import threading
from ..utils.logger import totsu_logger
from ..utils.model_processor import ModelProcessor

class SensitivityAnalyzer:
    def __init__(self, model, solver, use_jupyter=False):
        self.solver = solver
        self.model = model.clone()
        self.significant_constraints = None
        self.all_constraints = self.get_all_constraints(self.model)
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
        self.last_termination_info = None
        self.infeasible_points = []
        self.unbounded_points = []
        self.baseline_shadow_prices = None
        self.use_jupyter = use_jupyter

    def get_all_constraints(self, model):
        # Extract all constraint names from the model.
        constraint_names = []
        for c in ModelProcessor.get_constraints(model):
            # Check if the constraint is indexed
            if c.is_indexed():
                # c has multiple indexed constraints
                for index in c:
                    constr = c[index]
                    constraint_names.append(constr.name)
            else:
                # c is a single (scalar) constraint
                constraint_names.append(c.name)
        return constraint_names

    def solve_primal(self):
        # Solve the primal model
        try:
            model = self.model.clone()
            z_primal, shadow_prices = self.solve_lp(model, {})
            if z_primal is None:
                totsu_logger.info("Primal model is not optimal or not feasible.")
                return False
            self.baseline_shadow_prices = shadow_prices
        except Exception as e:
            totsu_logger.info("Exception occurred while solving the primal model:", e)
            return False

        # Identify significant constraints
        self.significant_constraints = self.get_significant_constraints(shadow_prices)

        if len(self.significant_constraints) == 1:
            totsu_logger.error(f"More than one constraint is required for this visualization.")
            return False

        # Get original RHS values for significant constraints
        for constr_name in self.significant_constraints:
            constraint = self.get_constraint_by_name(model, constr_name)
            if constraint.has_ub():
                self.b_original[constr_name] = constraint.upper()
            elif constraint.has_lb():
                self.b_original[constr_name] = constraint.lower()
            else:
                self.b_original[constr_name] = value(constraint.body)

        return True

    def solve_lp(self, model, rhs_adjustments):
        # Adjust RHS values of the constraints
        for constr_name, new_rhs in rhs_adjustments.items():
            constraint = self.get_constraint_by_name(model, constr_name)
            expr = constraint.body

            if constraint.equality:
                constraint.set_value(expr == new_rhs)
            elif constraint.has_lb() and not constraint.has_ub():
                constraint.set_value(expr >= new_rhs)
            elif constraint.has_ub() and not constraint.has_lb():
                constraint.set_value(expr <= new_rhs)
            elif constraint.has_lb() and constraint.has_ub():
                # If it's a double-bounded constraint, assume adjusting lower bound for now.
                constraint.set_value(inequality(new_rhs, expr, constraint.upper()))
            else:
                # No explicit bounds? Set as equality.
                constraint.set_value(expr == new_rhs)

        # Declare dual suffixes to get shadow prices
        if hasattr(model, 'dual'):
            model.del_component(model.dual)
        model.dual = Suffix(direction=Suffix.IMPORT)

        result = self.solver.solve(model, tee=False)

        self.last_termination_info = (result.solver.status, result.solver.termination_condition)

        # Check solver status
        if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
            # Return None to indicate no optimal solution
            totsu_logger.warning("Solver did not find an optimal solution at given RHS adjustments.")
            return None, {}

        # Get the objective value
        objective_value = ModelProcessor.get_active_objective_value(model)

        # Extract shadow prices (dual values)
        shadow_prices = {}
        for c in ModelProcessor.get_constraints(model):
            if isinstance(c, Constraint):
                # Iterate over indices of the Constraint object
                for index in c:
                    constr = c[index]
                    dual_value = model.dual.get(constr, 0.0)
                    shadow_prices[constr.name] = dual_value
            elif isinstance(c, ConstraintList):
                # Iterate over all constraints in the ConstraintList
                for constr in c:
                    dual_value = model.dual.get(constr, 0.0)
                    shadow_prices[constr.name] = dual_value
            elif isinstance(c, ConstraintData):
                # Single constraint
                dual_value = model.dual.get(c, 0.0)
                shadow_prices[c.name] = dual_value

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

    def get_constraint_by_name(self, model, name):
        if '[' in name and ']' in name:
            # It's likely a ConstraintList item
            base_name, index_str = name.split('[', 1)
            index = int(index_str.replace(']', ''))
            constraint_list = getattr(model, base_name)
            return constraint_list[index]
        else:
            # It's a regular constraint or block attribute
            return getattr(model, name)

    def compute_valid_range(self, model, constraint_name, b_original_value, other_constraints, b_original_values, delta=1.0):
        # Initialize variables
        allowable_increase = 0.0
        allowable_decrease = 0.0

        # Get the original shadow price and objective value at original point
        rhs_adjustments = {name: b_original_values[name] for name in other_constraints}
        rhs_adjustments[constraint_name] = b_original_value
        z_original, shadow_prices_original = self.solve_lp(model.clone(), rhs_adjustments)
        if z_original is None:
            totsu_logger.info(f"Could not find a valid solution at original point for {constraint_name}.")
            return allowable_increase, allowable_decrease

        original_shadow_price = shadow_prices_original.get(constraint_name, 0.0)
        original_objective = z_original

        # Check increase direction
        b = b_original_value
        iteration = 0
        max_iterations = 1000
        while iteration < max_iterations:
            iteration += 1
            b += delta
            rhs_adjustments[constraint_name] = b
            z_new, shadow_prices_new = self.solve_lp(model.clone(), rhs_adjustments)
            if z_new is None:
                # No further feasible/optimal solutions in this direction
                break
            new_shadow_price = shadow_prices_new.get(constraint_name, 0.0)
            expected_objective = original_objective + original_shadow_price * (b - b_original_value)
            if (abs(new_shadow_price - original_shadow_price) > 1e-5 or
                abs(z_new - expected_objective) > 1e-2):
                break
            allowable_increase = b - b_original_value

        if iteration >= max_iterations:
            totsu_logger.warning(f"Gave up on calculating valid increase range for {constraint_name}")

        # Check decrease direction
        b = b_original_value
        iteration = 0
        while iteration < max_iterations and b > 0:
            iteration += 1
            b -= delta
            if b < 0:
                b = 0
            rhs_adjustments[constraint_name] = b
            z_new, shadow_prices_new = self.solve_lp(model.clone(), rhs_adjustments)
            if z_new is None:
                # No further feasible/optimal solutions in this direction
                break
            new_shadow_price = shadow_prices_new.get(constraint_name, 0.0)
            expected_objective = original_objective + original_shadow_price * (b - b_original_value)
            if (abs(new_shadow_price - original_shadow_price) > 1e-5 or
                abs(z_new - expected_objective) > 1e-2):
                break
            allowable_decrease = b_original_value - b

        if iteration >= max_iterations:
            totsu_logger.warning(f"Gave up on calculating valid decrease range for {constraint_name}")

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
        x_constr, y_constr = self.significant_constraints
        for idx in np.ndindex(B[0].shape):
            if self.stop_computation:
                totsu_logger.info("Computation stopped by user.")
                self.computation_data["completed"] = True
                return
            rhs_adjustments = {}
            for constr_name in significant_constraints:
                rhs_adjustments[constr_name] = B[constr_indices[constr_name]][idx]
            z_value, _ = self.solve_lp(self.model.clone(), rhs_adjustments)
            if z_value is None:
                # Determine if it is infeasible or unbounded by checking solver conditions
                # Assume you've modified solve_lp to return termination condition
                # For illustration:
                status, term_cond = self.last_termination_info
                if term_cond == TerminationCondition.infeasible:
                    self.infeasible_points.append((B[constr_indices[x_constr]][idx], B[constr_indices[y_constr]][idx]))
                elif term_cond == TerminationCondition.unbounded:
                    self.unbounded_points.append((B[constr_indices[x_constr]][idx], B[constr_indices[y_constr]][idx]))
                Z[idx] = np.nan
            else:
                Z[idx] = z_value
            processed_points += 1
            self.computation_data["progress"] = processed_points / total_points * 100

        is_nan = np.isnan(Z)
        not_nan_counts = np.sum(~is_nan, axis=1)
        totsu_logger.info(f"Computed {np.sum(not_nan_counts)} valid points out of {total_points}.")

        # Update computation_data with B and Z
        self.computation_data["B"] = B
        self.computation_data["Z"] = Z

        # Compute valid ranges for each significant constraint
        for constr_name in significant_constraints:
            try:
                allowable_increase, allowable_decrease = self.compute_valid_range(
                    self.model, constr_name, b_original[constr_name], significant_constraints, b_original
                )
                self.computation_data['valid_ranges'][constr_name] = [
                    b_original[constr_name] - allowable_decrease,
                    b_original[constr_name] + allowable_increase
                ]
            except Exception as e:
                totsu_logger.warning(f"Could not compute valid range for {constr_name}: {e}")
                # If we fail, set the range as the original only
                self.computation_data['valid_ranges'][constr_name] = [b_original[constr_name], b_original[constr_name]]

        totsu_logger.info("Computation completed.")
        self.computation_data["completed"] = True

    def compute_ridge_line(self, x_plot_min, x_plot_max, y_plot_min, y_plot_max, num_steps=100, step_size=1.0):
        x_constr, y_constr = self.significant_constraints

        def trace_path(x_start, y_start, step_size, direction_multiplier):
            x_values = []
            y_values = []
            z_values = []

            x_current = x_start
            y_current = y_start

            for _ in range(num_steps):
                z_current, shadow_prices = self.solve_lp(
                    self.model.clone(), {x_constr: x_current, y_constr: y_current}
                )
                if z_current is None:
                    # Cannot solve LP at this point
                    break
                x_values.append(x_current)
                y_values.append(y_current)
                z_values.append(z_current)

                # Get the shadow prices
                shadow_price_x = shadow_prices.get(x_constr, 0.0)
                shadow_price_y = shadow_prices.get(y_constr, 0.0)

                # Define direction vector based on shadow prices
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

            return x_values, y_values, z_values

        # Start from the original point
        x_start = self.b_original[x_constr]
        y_start = self.b_original[y_constr]

        # Trace in both directions
        x_vals_pos, y_vals_pos, z_vals_pos = trace_path(x_start, y_start, step_size, direction_multiplier=1)
        x_vals_neg, y_vals_neg, z_vals_neg = trace_path(x_start, y_start, step_size, direction_multiplier=-1)

        x_values = x_vals_neg[::-1][:-1] + x_vals_pos
        y_values = y_vals_neg[::-1][:-1] + y_vals_pos
        z_values = z_vals_neg[::-1][:-1] + z_vals_pos

        if len(x_values) < 2:
            return None, None, None, "Too few points"

        return x_values, y_values, z_values, None

    def show_analyzer(self):
        if not self.b_original:
            totsu_logger.error("Please call solve_primal before calling show_analyzer")
            return

        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Dropdown for constraint selection
        constraint_options = []
        for cname in self.all_constraints:
            dual_val = self.baseline_shadow_prices.get(cname, 0.0)  # If no dual val, default to 0.0
            constraint_options.append({
                'label': f"{cname} ({dual_val:.4f})",
                'value': cname
            })
        constraint_dropdown = dcc.Dropdown(
            id='constraint-dropdown',
            options=constraint_options,
            multi=True,
            placeholder="Select two constraints",
            value=self.significant_constraints,
            style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'middle'}  # Wider and inline
        )

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

        app.layout = html.Div([
            html.H1("LP Sensitivity Analysis Visualization"),
            html.Div([
                html.Label("Select Two Constraints for Visualization:", style={'display': 'inline-block', 'margin-right': '10px'}),
                constraint_dropdown,
                html.Button("Apply Constraints", id="apply-constraints-button", n_clicks=0, style={'margin-bottom': '10px'}),
            ], style={'margin-bottom': '20px', 'whiteSpace': 'nowrap'}),

            html.Div(id='status-message', style={'margin': 'auto', 'vertical-align': 'middle', 'margin-bottom': '20px'}),
            
            html.Div(id='slider-container', style={'width': '80%', 'margin': 'auto', 'margin-bottom': '20px'}),

            html.Button("Start Exploration", id="start-button", n_clicks=0),
            html.Button("Stop Exploration", id="stop-button", n_clicks=0),
            dcc.Graph(id="lp-graph", figure=initial_fig),
            dcc.Interval(id="interval-component", interval=1000, n_intervals=0, disabled=True),
            
            dcc.Store(id='computation-status', data={'running': False}),
            dcc.Store(id='selected-constraints', data=self.significant_constraints),
            dcc.Store(id='constraint-message', data=''),  # store for constraint application messages
            dcc.Store(id='computation-message', data='')   # store for computation progress messages
        ])

        # Callback to update significant constraints based on user selection
        @app.callback(
            [Output('selected-constraints', 'data'),
            Output('slider-container', 'children'),
            Output('constraint-message', 'data')],
            [Input('apply-constraints-button', 'n_clicks'),
            Input('constraint-dropdown', 'value')],
            [State('selected-constraints', 'data')]
        )
        def update_selected_constraints(n_clicks, selected_constraints_from_dropdown, current_selected_constraints):
            ctx = dash.callback_context
            if not ctx.triggered:
                # On initial load (no user interaction yet)
                default_msg = f"Using default constraints: {current_selected_constraints[0]} and {current_selected_constraints[1]}"
                return current_selected_constraints, generate_sliders(current_selected_constraints), default_msg
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                if button_id == 'apply-constraints-button':
                    if selected_constraints_from_dropdown is None or len(selected_constraints_from_dropdown) != 2:
                        return dash.no_update, dash.no_update, "Please select exactly two constraints."
                    # Constraints applied by user
                    return (
                        selected_constraints_from_dropdown,
                        generate_sliders(selected_constraints_from_dropdown),
                        f"Constraints '{selected_constraints_from_dropdown[0]}' and '{selected_constraints_from_dropdown[1]}' applied successfully!"
                    )
                elif button_id == 'constraint-dropdown':
                    # If user just changes dropdown without pressing apply, do not update the message
                    raise dash.exceptions.PreventUpdate

        def generate_sliders(constraints):
            slider_elements = []
            for constr_name in constraints:
                slider_id = {'type': 'range-slider', 'index': constr_name}
                slider_elements.append(html.Label(f"{constr_name} Change (%)"))
                slider_elements.append(
                    dcc.RangeSlider(
                        id=slider_id,
                        min=-50,
                        max=50,
                        step=1,
                        value=[-20, 20],
                        marks={i: f'{i}%' for i in range(-50, 51, 10)},
                    )
                )
            return slider_elements

        @app.callback(
            Output('constraint-dropdown', 'value'),
            [Input('constraint-dropdown', 'value')]
        )
        def limit_constraint_selection(selected_constraints):
            # If no selection or it's None, do nothing
            if selected_constraints is None:
                raise PreventUpdate

            # If more than two constraints have been selected,
            # truncate the list to the first two chosen
            if len(selected_constraints) > 2:
                return selected_constraints[:2]

            return selected_constraints

        @app.callback(
            Output("computation-message", "data"),
            [Input("interval-component", "n_intervals")],
            [State("computation-status", "data")]
        )
        def update_progress(n, computation_status):
            if self.computation_thread is not None and self.computation_thread.is_alive():
                progress = self.computation_data.get("progress", 0)
                return f"Computation in progress: {progress:.1f}% completed."
            elif self.computation_data.get("completed", False):
                return "Computation completed."
            else:
                return ""

        @app.callback(
            Output('status-message', 'children'),
            [Input('constraint-message', 'data'),
            Input('computation-message', 'data')]
        )
        def show_last_updated_message(constraint_msg, computation_msg):
            ctx = dash.callback_context
            if not ctx.triggered:
                return ""  # No messages yet

            # Identify which input triggered the callback
            triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

            if triggered_input == 'constraint-message':
                return constraint_msg  # Show only constraint message
            elif triggered_input == 'computation-message':
                return computation_msg  # Show only computation message

            return ""  # Default fallback if none match

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
                constr_indices = {name: idx for idx, name in enumerate(self.significant_constraints)}

                # Extract B and Z
                B = self.computation_data["B"]
                Z = self.computation_data["Z"]
                x_values = B[constr_indices[x_constr]]
                y_values = B[constr_indices[y_constr]]
                totsu_logger.debug(f"Plotting {len(x_values)} x {len(y_values)} grid points.")
                totsu_logger.debug(f"Z stats: {np.nanmin(Z)}, {np.nanmax(Z)}, {np.isnan(Z).sum()}")
                totsu_logger.debug(f"{x_values.shape}, {y_values.shape}, {Z.shape}")


                # Clear existing data
                fig.data = []

                # Add surface
                surface = go.Surface(
                    x=x_values,
                    y=y_values,
                    z=Z,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Objective Value"),
                    name="Objective Value Surface",
                    opacity=0.8,
                    hovertemplate=f"{x_constr} RHS: %{{x:.1f}}<br>"
                                  f"{y_constr} RHS: %{{y:.1f}}<br>"
                                  f"Z: %{{z:.2f}}<extra></extra>",
                )
                fig.add_trace(surface)

                # Add empty ridge line and marker
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

                # Axis ranges
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
                    uirevision='data_changed',
                )

                # Original point
                x_original = self.b_original[x_constr]
                y_original = self.b_original[y_constr]
                rhs_adjustments = {x_constr: x_original, y_constr: y_original}
                Z_original, _ = self.solve_lp(self.model.clone(), rhs_adjustments)

                # Valid ranges
                valid_range_x = self.computation_data['valid_ranges'].get(x_constr)
                if valid_range_x and valid_range_x[0] != valid_range_x[1]:
                    x_line = np.linspace(valid_range_x[0], valid_range_x[1], 100)
                    y_const = y_original
                    Z_x_line = []
                    for x_val in x_line:
                        rhs_adjustments = {x_constr: x_val, y_constr: y_const}
                        z_val, _ = self.solve_lp(self.model.clone(), rhs_adjustments)
                        Z_x_line.append(np.nan if z_val is None else z_val)

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

                valid_range_y = self.computation_data['valid_ranges'].get(y_constr)
                if valid_range_y and valid_range_y[0] != valid_range_y[1]:
                    y_line = np.linspace(valid_range_y[0], valid_range_y[1], 100)
                    x_const = x_original
                    Z_y_line = []
                    for y_val in y_line:
                        rhs_adjustments = {x_constr: x_const, y_constr: y_val}
                        z_val, _ = self.solve_lp(self.model.clone(), rhs_adjustments)
                        Z_y_line.append(np.nan if z_val is None else z_val)

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

                # Optimal point
                if Z_original is not None:
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

                # After computing all points
                x_infeas = [pt[0] for pt in self.infeasible_points]
                y_infeas = [pt[1] for pt in self.infeasible_points]
                # Choose a baseline z to plot these points, e.g., min(Z) - 10
                if np.all(np.isnan(Z)):
                    baseline_z = 0
                else:
                    baseline_z = np.nanmin(Z) - 10
                z_infeas = [baseline_z]*len(self.infeasible_points)

                infeasible_trace = go.Scatter3d(
                    x=x_infeas,
                    y=y_infeas,
                    z=z_infeas,
                    mode='markers',
                    name='Infeasible Region',
                    marker=dict(size=5, color='red', symbol='x'),
                    hovertemplate="Infeasible at<br>x: %{x}<br>y: %{y}<extra></extra>"
                )
                fig.add_trace(infeasible_trace)

                # Compute ridge line
                x_ridge, y_ridge, z_ridge, ridge_error = self.compute_ridge_line(x_min, x_max, y_min, y_max)
                if ridge_error is None and x_ridge is not None:
                    frames = []
                    for i in range(len(x_ridge)):
                        frames.append(
                            go.Frame(
                                data=[
                                    dict(
                                        x=x_ridge[:i + 1],
                                        y=y_ridge[:i + 1],
                                        z=z_ridge[:i + 1],
                                        mode='lines',
                                        line=dict(color='green', width=8),
                                        type='scatter3d'
                                    ),
                                    dict(
                                        x=[x_ridge[i]],
                                        y=[y_ridge[i]],
                                        z=[z_ridge[i]],
                                        mode='markers',
                                        marker=dict(size=2, color='green'),
                                        type='scatter3d'
                                    )
                                ],
                                traces=[1, 2],
                                name=str(i)
                            )
                        )

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
                    totsu_logger.error(f"No ridge line to plot because {ridge_error}")

                totsu_logger.debug(f"fig.data: {fig.data}")
                return fig
            else:
                raise dash.exceptions.PreventUpdate

        @app.callback(
            Output("computation-status", "data"),
            [Input("start-button", "n_clicks"),
             Input("stop-button", "n_clicks"),
             Input("interval-component", "n_intervals")],
            [State({'type': 'range-slider', 'index': ALL}, 'value'),
             State('computation-status', 'data'),
             State('selected-constraints', 'data')]
        )
        def control_computation(start_clicks, stop_clicks, n_intervals, slider_values, computation_status, selected_constraints):
            ctx = dash.callback_context

            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            else:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "start-button":
                # If user-selected constraints are available, use them
                if selected_constraints and len(selected_constraints) == 2:
                    self.significant_constraints = selected_constraints
                else:
                    # If none selected, fallback to previous logic (already chosen constraints)
                    pass

                # Now that self.significant_constraints is set, re-generate self.b_original if needed
                self.b_original = {}
                for constr_name in self.significant_constraints:
                    constraint = self.get_constraint_by_name(self.model, constr_name)
                    if constraint.has_ub():
                        self.b_original[constr_name] = constraint.upper()
                    elif constraint.has_lb():
                        self.b_original[constr_name] = constraint.lower()
                    else:
                        self.b_original[constr_name] = value(constraint.body)

                if self.computation_thread is None or not self.computation_thread.is_alive():
                    self.stop_computation = False
                    self.computation_data["B"] = None
                    self.computation_data["Z"] = None
                    self.computation_data["progress"] = 0
                    self.computation_data["completed"] = False
                    self.computation_data["valid_ranges"] = {}
                    # Build percentage range dictionary
                    b_range_percentages = {}
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

        if self.use_jupyter:
            app.run_server(mode='inline', debug=True)
        else:
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
            uirevision='data_changed',
        )
        return fig
