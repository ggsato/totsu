# Simplex Desgin Concepts

1. **Tableau Class**:
   - Design a `Tableau` class dedicated to the Simplex method, encapsulating basis management, variable mappings, and simplex operations.
   - Implement a rich set of methods to manage pivots, track basis/non-basis variables, and handle name/index mappings, focusing on robustness and educational clarity.

2. **Visualization with History**:
   - Track each step of the Simplex process, visualizing convexity, feasible regions, and tableau changes dynamically.
   - Maintain a history of tableau states to review the Simplex progression and support learning.

3. **Helper Class for Model Standardization**:
   - Create a Helper class to construct an initial standard tableau for the main Simplex class to start directly with Phase I, simplifying the design.
   - Detect and adjust non-standard constraints (e.g., equalities or "≥" inequalities), adding artificial variables as necessary.

4. **Pyomo Model Integration and Consistency in Branch and Bound**:
   - Make the passed Pyomo model immutable to prevent unintended modifications, ensuring consistency when Simplex is used within Branch and Bound or other solvers.
   - Use a converted model (via the Helper class) to manage dynamic constraints and bounds added by other solvers, synchronizing them with the tableau as needed.

# Visualization Design Concepts

- **Focus on the Dual Model:**
  - Utilize the dual linear programming model instead of the primal to simplify visualization.
  - Dual variables correspond directly to primal constraints, making relationships more straightforward.

- **Visualization Axes:**
  - Plot two dual variables on the x and y axes.
  - Represent the dual objective value on the z-axis.

- **Direct Relationship with Objective Value:**
  - Changes in dual variables directly affect the dual objective value.
  - Eliminates indirect complications associated with shadow prices and primal variables.

- **Interactive Exploration:**
  - Allow users to adjust dual variables and observe real-time changes in the objective value.
  - Enable selection of different dual variables for visualization as needed.

- **Feasibility and Sensitivity Analysis:**
  - Visualize the feasible region in the dual space.
  - Show how variations in constraints (primal right-hand side values) impact the optimal solution.

- **Educational Support:**
  - Provide explanations and tooltips to help users understand dual variables and their connection to the primal problem.
  - Include tutorials or guided examples within the tool.

- **Enhanced Clarity and Insight:**
  - By working in the dual space, offer a more intuitive understanding of how constraints influence the optimization.
  - Simplify the visualization, making it accessible even for those with basic knowledge of linear programming.