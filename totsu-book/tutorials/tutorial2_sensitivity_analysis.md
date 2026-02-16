# Tutorial 2: Sensitivity Analysis with SensitivityAnalyzer

## **1. Introduction**

Sensitivity analysis is a crucial step in understanding the robustness of an optimal solution in linear programming (LP). It helps answer questions like:
- How does the optimal objective value change when a constraint's right-hand side (RHS) is modified?
- Which constraints most significantly impact the objective?

The **Totsu SensitivityAnalyzer** allows users to:
- Explore changes to constraint bounds.
- Visualize objective value variations across constraint ranges.
- Identify significant constraints and their valid ranges.

This tutorial also introduces the **Duality Theorem**, which establishes the relationship between the primal and dual problems in LP. By exploring the dual model of the Product Mix problem, we demonstrate how dual variables (shadow prices) and reduced costs connect to sensitivity analysis.

---

## **2. Problem Setup**

We will analyze the Product Mix problem, a maximization LP that balances resource constraints to achieve optimal profits.

### **Primal Model**

#### **Objective**:
Maximize $Z = 550x_1 + 600x_2 + 350x_3 + 400x_4 + 200x_5$.

#### **Constraints**:
1. Grinding capacity:
   $$ 12x_1 + 20x_2 + 25x_4 + 15x_5 \leq 288 $$
2. Drilling capacity:
   $$ 10x_1 + 8x_2 + 16x_3 \leq 192 $$
3. Manpower capacity:
   $$ 20x_1 + 20x_2 + 20x_3 + 20x_4 + 20x_5 \leq 384 $$

#### **Non-Negativity**:
$$ x_1, x_2, x_3, x_4, x_5 \geq 0 $$

### **Dual Model**
The dual problem translates constraints from the primal model into variables and vice versa:

#### **Objective**:
Minimize $W = 288y_1 + 192y_2 + 384y_3$.

#### **Constraints**:
1. Product 1: $12y_1 + 10y_2 + 20y_3 \geq 550$
2. Product 2: $20y_1 + 8y_2 + 20y_3 \geq 600$
3. Product 3: $16y_2 + 20y_3 \geq 350$
4. Product 4: $25y_1 + 20y_3 \geq 400$
5. Product 5: $15y_1 + 20y_3 \geq 200$

#### **Non-Negativity**:
$$ y_1, y_2, y_3 \geq 0 $$

---

## **3. Duality Theorem**

The **Duality Theorem** states that:
1. The optimal objective values of the primal and dual problems are equal.
2. Dual variables ($y_1, y_2, y_3$) represent shadow prices, indicating how much the objective value would improve per unit increase in the RHS of the corresponding constraint.
3. Reduced costs in the primal problem indicate whether producing additional units of a product is profitable given the dual prices.

---

## **4. Using SensitivityAnalyzer**

### **Step 1: Solve the Primal Model**
First, solve the LP problem and identify the optimal solution.

```python
from product_mix import create_model
from totsu.core.sensitivity_analyzer import SensitivityAnalyzer
from pyomo.environ import SolverFactory

model = create_model()
solver = SolverFactory("glpk")
solver.solve(model)

analyzer = SensitivityAnalyzer(model, solver="glpk")
result = analyzer.solve_primal()
if result:
    print("Primal solution solved successfully.")
else:
    print("Error solving the primal model.")
```

---

### **Step 2: Analyze the Dual Model**
Solve the dual model and interpret dual variables as shadow prices:

```python
from product_mix import create_dual_model

dual_model = create_dual_model()
solver.solve(dual_model)

dual_solution = {
    var.name: var.value for var in dual_model.component_objects(Var, active=True)
}
print("Dual Solution:", dual_solution)
```

#### **Interpretation**:
- $y_1$: Value of increasing grinding capacity.
- $y_2$: Value of increasing drilling capacity.
- $y_3$: Value of increasing manpower capacity.

---

### **Step 3: Visualize Objective Changes**
Visualize how the objective value varies when adjusting the RHS of significant constraints.

```python
analyzer.show_analyzer()
```

This opens an interactive visualization with:
- **3D Surface Plot**: Objective value across the ranges of two selected constraints.
- **Infeasible and Unbounded Regions**: Highlighted areas indicating constraint violations or unbounded solutions.

---

## **5. Advanced Insights**

### **1. Valid Ranges for Constraints**
The SensitivityAnalyzer computes valid ranges for the RHS of significant constraints, ensuring feasibility and optimality.

```python
for constr, valid_range in analyzer.computation_data['valid_ranges'].items():
    print(f"Constraint: {constr}, Valid Range: {valid_range}")
```

### **2. Exploring Reduced Costs**
Reduced costs indicate whether increasing production of a specific product is profitable under current constraints.

```python
for var_name, reduced_cost in analyzer.computation_data['reduced_costs'].items():
    print(f"Product: {var_name}, Reduced Cost: {reduced_cost}")
```

---

## **6. Expected Results**

- **Significant Constraints**:
  - Grinding and Manpower capacities are likely to emerge as critical due to high utilization rates.

- **Valid Ranges**:
  - Example: Manpower capacity may have a tight range due to its binding nature in the primal solution.

- **Dual Variables (Shadow Prices)**:
  - Provide insights into the worth of increasing capacities.

- **Reduced Costs**:
  - Indicate whether producing additional units of certain products would be profitable.

---

## **7. Conclusion**

This tutorial demonstrated:
1. Solving a primal LP model.
2. Exploring duality and interpreting shadow prices.
3. Performing sensitivity analysis using the Totsu SensitivityAnalyzer.
4. Visualizing and interpreting constraint impact and reduced costs.

In the next tutorial, we will explore advanced real-world scenarios to deepen our understanding of LP models.
