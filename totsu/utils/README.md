# Sensitivity Analyzer

- **Linear Programming Sensitivity Analysis:**
  - Evaluates how changes in the right-hand side (RHS) values of constraints affect the optimal solution and objective value of a linear programming (LP) model.

- **Identification of Significant Constraints:**
  - Uses shadow prices (dual values) to determine which constraints have the most significant impact on the objective function.
  - Selects the top constraints with the highest absolute shadow prices for detailed analysis.

- **Computation of Valid Ranges:**
  - Calculates the allowable increase and decrease for the RHS values of significant constraints.
  - Determines the ranges within which the current optimal basis remains valid, ensuring linearity and constant shadow prices.

- **Model Cloning and Safe Modification:**
  - Clones the original Pyomo model to safely modify constraint RHS values without affecting the original model.
  - Ensures that each LP solve operates on an independent model instance.

- **Objective Function Surface Visualization:**
  - Generates a 3D surface plot of the objective function value over the ranges of the significant constraints.
  - Provides a visual representation of how the objective value changes with variations in constraint capacities.

- **Ridge Line Computation and Visualization:**
  - Computes the ridge or valley line representing the path of maximum increase or decrease in the objective function.
  - Implements a gradient ascent (or descent) algorithm, updating shadow prices at each step to follow the true ridge on the surface.
  - Visualizes the ridge line on the 3D plot to indicate the optimal direction for adjusting constraint capacities.

- **Directional Indicators Along the Ridge Line:**
  - Enhances the visualization with directional markers, such as multiple cones or markers along the ridge line, to indicate the direction of improvement.
  - Optionally uses animation to dynamically show the progression along the ridge line.

- **Interactive Dash Application:**
  - Provides an interactive user interface using Dash for users to explore sensitivity analysis results.
  - Includes sliders for adjusting the percentage change ranges of significant constraints.
  - Updates the visualization in real-time as users interact with the controls.

- **Exception Handling and Robustness:**
  - Implements thorough exception handling to manage cases like division by zero, infeasible solutions, and zero shadow prices.
  - Ensures that the application provides meaningful feedback and remains stable under various scenarios.

- **Cache Management:**
  - Utilizes caching mechanisms to store LP solve results and avoid redundant computations.
  - Enhances performance by retrieving previously computed solutions when the same RHS adjustments are requested.

- **Modularity and Extensibility:**
  - Structures the code with clear separation of concerns, making it easier to maintain and extend.
  - Allows for customization, such as changing the number of significant constraints analyzed or integrating different solvers.

- **Educational and Decision-Making Aid:**
  - Serves as a tool to help users understand the sensitivity of their LP models.
  - Assists in making informed decisions about resource allocation and constraint adjustments to achieve optimal outcomes.

- **Visualization Enhancements:**
  - Incorporates features like adjustable opacity, color schemes, and annotations to improve the clarity and aesthetics of the plots.
  - Ensures that visual elements like the ridge line, markers, and surface are distinct and convey information effectively.

- **Integration with Pyomo and Plotly:**
  - Leverages Pyomo for LP modeling and solving, benefiting from its robustness and flexibility.
  - Uses Plotly for creating interactive and high-quality visualizations within the Dash framework.
