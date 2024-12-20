# Sensitivity Analyzer

## Technical key features

The `SensitivityAnalyzer` class is designed to explore and visualize how the objective value of a linear programming (LP) problem changes as the right-hand side (RHS) values of significant constraints vary. Below are the **key features and functionalities**:

---

### **1. Model Cloning and Solver Integration**
- Clones the original Pyomo model to ensure any modifications (like adjusting RHS values) do not affect the original model.
- Integrates with an external solver (e.g., GLPK, CBC) to compute solutions efficiently.
- Supports solving both primal and dual problems to retrieve objective values and shadow prices.

---

### **2. Identification of Significant Constraints**
- Determines **significant constraints** based on the absolute values of the shadow prices (dual values) after solving the primal model.
- By default, selects the top two constraints with the largest shadow prices for sensitivity analysis.

---

### **3. Sensitivity Surface Computation**
- Computes how the **objective value (Z)** changes across a grid of RHS adjustments for the selected significant constraints.
- Uses a meshgrid to explore the joint effects of varying RHS values for two constraints within user-defined ranges.
- Identifies infeasible and unbounded points in the grid and handles them by marking those regions explicitly.

---

### **4. Ridge Line Computation**
- Computes the **ridge line**, which represents the path of steepest ascent (for maximization) or descent (for minimization) in the objective value.
- Uses **gradient descent** on shadow prices (dual values) to trace the ridge starting from the optimal solution point.
- Ensures reliable tracing by leveraging the convexity properties of LP problems and dual solutions.

---

### **5. Valid Range Analysis**
- For each significant constraint, calculates the **allowable increase and decrease** in the RHS value within which:
  - The LP problem remains feasible.
  - Shadow prices and objective values remain predictable and consistent.
- Provides insights into how far the RHS values can be perturbed without losing optimality.

---

### **6. Visualization**
- Uses **Plotly** to create an interactive 3D surface plot showing the sensitivity of the objective value to RHS adjustments.
  - X-axis and Y-axis represent the RHS values of the two significant constraints.
  - Z-axis represents the objective value (Z).
- Adds visual elements such as:
  - The **ridge line** showing the trajectory of optimality.
  - Markers for **infeasible** regions and **valid ranges**.
  - The original **optimal solution point**.
- Supports animations to illustrate the ridge line computation dynamically.

---

### **7. Dashboard Integration with Dash**
- Integrates with **Dash** to provide an interactive web-based dashboard for sensitivity analysis.
- Features:
  - Dropdown menu to select significant constraints.
  - Sliders to define the percentage range for exploring RHS adjustments.
  - Progress updates during computation.
  - Real-time updates of the 3D sensitivity plot.

---

### **8. Robustness and Flexibility**
- Handles infeasible and unbounded solutions gracefully, logging and marking those points on the visualization.
- Caches computed LP solutions to avoid redundant solver calls, improving performance.
- Designed to work seamlessly in both Jupyter Notebooks (`use_jupyter=True`) and standalone web applications.

---

### **9. Modular Design**
- Provides modular methods for solving LP problems, retrieving constraints, and performing computations (e.g., valid range analysis, ridge line computation).
- Allows easy extension for additional sensitivity analysis features or integration with other tools.

---

### **10. User-Centric Interactivity**
- Empowers users to explore LP sensitivity visually and intuitively, with features like:
  - Dynamic constraint selection.
  - Adjustable RHS ranges for exploration.
  - Clear visual feedback on infeasible and valid regions.

## Practical Key Features

From a **practical perspective**, the `SensitivityAnalyzer` class has significant potential in industries and applications where linear programming (LP) plays a critical role in decision-making. Here's how it meets practical needs and where it fits in the market:

---

### **1. Addressing Sensitivity and Robustness in Optimization**
**Practical Need:**
- Real-world optimization problems are rarely static; input parameters like resource availability, production limits, or demand forecasts often change. Decision-makers need to understand:
  - How robust their solutions are to changes in these parameters.
  - The range of parameter values that maintain feasibility and optimality.

**Market Fit:**
- The `SensitivityAnalyzer` directly meets this need by enabling:
  - Visualization of how changes in constraints affect the objective value.
  - Exploration of allowable ranges for constraint values before solutions become infeasible or suboptimal.
- Industries: Supply chain management, production planning, and resource allocation.

---

### **2. Visualization of Complex Interactions**
**Practical Need:**
- Decision-makers and analysts often struggle to interpret raw numerical output from solvers, especially when interacting parameters (constraints) have complex effects on outcomes.
  
**Market Fit:**
- The 3D sensitivity surface and ridge line visualization provide intuitive insights into how the objective value changes with RHS adjustments.
- Interactive features in Dash allow users to test scenarios in real-time, making it ideal for operational decision support.
- Industries: Logistics, transportation, energy markets, and financial modeling.

---

### **3. Real-Time Scenario Analysis**
**Practical Need:**
- In dynamic environments, decision-makers need tools to quickly evaluate "what-if" scenarios (e.g., "What happens if resource availability changes by ±20%?").

**Market Fit:**
- The ability to compute and visualize sensitivity in real-time aligns with the need for rapid scenario analysis.
- Dash-based interactivity allows for quick adjustments and visualization, making it suitable for:
  - Supply chain control towers.
  - Dynamic production scheduling.
  - Real-time bidding in energy markets.

---

### **4. Decision Validation and Communication**
**Practical Need:**
- Stakeholders (e.g., managers, clients, regulators) often require transparency and validation of optimization decisions.

**Market Fit:**
- The tool's visual approach simplifies the explanation of:
  - Why certain solutions are optimal.
  - How sensitive these solutions are to input changes.
- This transparency builds trust and improves collaboration among teams.
- Industries: Consulting, finance, and regulatory compliance.

---

### **5. Risk Management and Contingency Planning**
**Practical Need:**
- Organizations often need to prepare for disruptions or uncertainties (e.g., supply shortages, demand surges).

**Market Fit:**
- By identifying infeasible regions and valid ranges, the `SensitivityAnalyzer` helps:
  - Anticipate potential risks to feasibility.
  - Develop contingency plans for parameter variations.
- Industries: Disaster recovery planning, risk management, and public sector projects.

---

### **6. Educational and Research Use**
**Practical Need:**
- Educators and researchers require tools to demonstrate and explore the concepts of sensitivity analysis and duality in LP problems.

**Market Fit:**
- The tool provides a hands-on way to:
  - Illustrate LP concepts like shadow prices, sensitivity, and feasible regions.
  - Explore duality and convexity interactively.
- Institutions: Universities, training programs, and research labs.

---

### **Competitive Advantage**
- **Interactivity**: Many LP sensitivity tools focus solely on numerical output, but this tool’s visualization features set it apart.
- **Flexibility**: Works with Pyomo, an open-source modeling tool, making it accessible and adaptable.
- **Ease of Use**: Designed for both technical and non-technical users, bridging the gap between solver outputs and actionable insights.

---

### **Potential Market Segments**
1. **Supply Chain & Logistics**:
   - Route planning, inventory management, resource allocation.
2. **Energy & Utilities**:
   - Power generation planning, grid optimization, and energy bidding.
3. **Financial Services**:
   - Portfolio optimization, risk assessment, and pricing models.
4. **Manufacturing**:
   - Production scheduling, resource planning, and cost minimization.
5. **Consulting Firms**:
   - Operational research services for clients in various industries.
6. **Academia & Education**:
   - Teaching optimization concepts interactively.

---

### **Monetization Opportunities**
- **Open-Source + Freemium Model**:
  - Basic features remain free in an open-source framework, with premium features (e.g., advanced analytics, cloud integration) offered as paid add-ons.
- **Enterprise Licensing**:
  - Provide customized, enterprise-grade sensitivity analysis tools integrated with industry-specific workflows.
- **SaaS Platform**:
  - Offer the tool as a subscription service with cloud-based computation and visualization.
  
---

The `SensitivityAnalyzer` is positioned to fill a critical gap in the market by providing a robust yet accessible tool for sensitivity analysis, blending technical rigor with practical usability.