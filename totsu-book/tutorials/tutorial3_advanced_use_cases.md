# Tutorial 3: Advanced Use Cases

## **1. Introduction**

This tutorial explores advanced applications of the Simplex Solver and SensitivityAnalyzer, focusing on real-world scenarios like blending and network optimization. By combining insights from the first two tutorials, we demonstrate:
- Solving complex LP models.
- Understanding relationships between constraints, objective values, and decision variables.
- Exploring trade-offs and sensitivity in multi-dimensional contexts.

---

## **2. Blending Problem: Balancing Costs and Quality**

The **Blending Problem** optimizes production costs while meeting quality and capacity constraints. This example highlights solving a **minimization problem**.

### **Objective**:
Minimize $Z = 110x_1 + 120x_2 + 130x_3 + 110x_4 + 115x_5 - 150y$.

### **Constraints**:
1. Refining vegetable oil capacity:
   
$$
 x_1 + x_2 \leq 200 
$$

2. Refining non-vegetable oil capacity:
   
$$
 x_3 + x_4 + x_5 \leq 250 
$$

3. Hardness range:
   - Maximum hardness:
     
$$
 8.8x_1 + 6.1x_2 + 2.0x_3 + 4.2x_4 + 5.0x_5 - 6y \leq 0 
$$

   - Minimum hardness:
     
$$
 8.8x_1 + 6.1x_2 + 2.0x_3 + 4.2x_4 + 5.0x_5 - 3y \geq 0 
$$

4. Weight balance:
   
$$
 x_1 + x_2 + x_3 + x_4 + x_5 - y = 0 
$$


### **Steps**:

### **Step 1: Solve the Model**
Solve the blending model using the Simplex Solver:

```python
from blending import create_model
from totsu.core.super_simplex_solver import SuperSimplexSolver

model = create_model()
solver = SuperSimplexSolver()
solution = solver.solve(model)

print("Optimal Objective Value:", solver.get_current_objective_value())
for var in solution:
    print(f"{var} = {solution[var]}")
```

### **Step 2: Analyze Constraints and Variables**
Evaluate the critical constraints and slack variables to understand feasibility and quality requirements.

```python
print("Constraint Violations:", model.hardness_min.body(), model.hardness_max.body())
```

#### **What to Look For:**
- **Minimization Process**: Observe how the tableau iterations reduce the total production cost.
- **Slack Variables**: Identify unused capacities or non-binding constraints.

---

## **3. Minimum Cost Flow: Optimizing Networks**

The **Minimum Cost Flow Problem** models transportation or supply chain optimization. This example demonstrates how **fixed variables** are handled through pre-processing and post-processing in the tableau.

### **Objective**:
Minimize total transportation costs across a network:

$$
 Z = \sum_{(i, j)} c_{ij} x_{ij} 
$$


### **Constraints**:
1. Flow balance at each node:
   
$$
 \sum_{j} x_{ji} - \sum_{j} x_{ij} = b_i, \quad \forall i 
$$

   Where $b_i$ represents supply ($b_i > 0$), demand ($b_i < 0$), or transit ($b_i = 0$).
   
2. Non-negativity:
   
$$
 x_{ij} \geq 0 
$$


### **Steps**:

### **Step 1: Define and Solve the Model**

```python
from minimum_cost_flow import create_model
from totsu.core.super_simplex_solver import SuperSimplexSolver

model = create_model()
solver = SuperSimplexSolver()
solution = solver.solve(model)

print("Optimal Objective Value:", solver.get_current_objective_value())
for var in solution:
    print(f"{var} = {solution[var]}")
```

### **Step 2: Visualize Tableau and Analyze Flows**
Use the TableauVisualizer to understand flow distribution and detect bottlenecks.

```python
from totsu.core.tableau_visualizer import TableauVisualizer

visualizer = TableauVisualizer(model, use_jupyter=True)
visualizer.solve_model()
visualizer.show_tableau_visualization()
```

#### **What to Look For:**
- **Pre-Processed Fixed Variables**: Observe that fixed flow values are pre-computed outside the tableau, simplifying the optimization process.
- **Post-Processed Objective Value**: After optimization, verify that fixed variables are accounted for in the final objective value.
- **Flow Balances**: Ensure that all nodes satisfy the balance constraints.

---

## **4. Advanced Sensitivity Analysis**

Building on Tutorial 2, analyze trade-offs and explore significant constraints in multi-dimensional scenarios.

### **Step 1: Sensitivity Analysis for Blending**

```python
from totsu.core.sensitivity_analyzer import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(model, solver="glpk")
analyzer.solve_primal()
analyzer.show_analyzer()
```

### **Step 2: Insights from Duality**
Evaluate dual variables and reduced costs to make informed decisions:

```python
print("Dual Variables:", analyzer.computation_data['dual_values'])
print("Reduced Costs:", analyzer.computation_data['reduced_costs'])
```

---

## **5. Conclusion**

This tutorial demonstrated:
1. Solving complex LP problems like blending and network optimization.
2. Performing advanced sensitivity analysis to understand trade-offs.
3. Visualizing LP models for deeper insights.

With these tools and techniques, you can tackle real-world optimization challenges confidently!

