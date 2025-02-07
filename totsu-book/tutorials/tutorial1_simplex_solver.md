# Tutorial 1: Introduction to the Simplex Solver

## **1. Introduction**

The **Simplex Method** is a powerful algorithm for solving linear programming (LP) problems. It iteratively improves an objective function, finding an optimal solution while adhering to constraints.

Understanding the **Simplex Tableau** is critical for grasping how the algorithm works. The tableau not only represents the LP problem in a tabular format but also allows systematic exploration of feasible solutions. Through tableau visualization, learners and practitioners can:
- Debug LP models by observing pivot operations.
- Analyze constraint satisfaction.
- Study the traversal of vertices in the feasible region.

---

## **2. Why Does Simplex Work?**

The Simplex algorithm works by systematically improving the value of the objective function while maintaining feasibility. Each **iteration** of the algorithm corresponds to moving from one vertex of the feasible region to another. This guarantees improvement (or optimality) at each step because:

1. **Pivot Column Selection**: A variable is chosen to enter the basis based on its potential to improve the objective value.
2. **Pivot Row Selection**: A variable is chosen to leave the basis to ensure feasibility.

By continuing this process, the algorithm identifies the vertex of the feasible region that optimizes the objective function.

---

## **3. What is a Basic Feasible Solution (BFS)?**

A **Basic Feasible Solution (BFS)** is a cornerstone of the Simplex algorithm. It refers to a solution that:
1. **Satisfies all constraints**: Ensures feasibility.
2. **Has the minimum number of non-zero variables**: Corresponds to the number of constraints.

Each BFS represents a vertex of the feasible region. The Simplex algorithm starts at one BFS and systematically moves to adjacent vertices, improving the objective value.

> **Example**: In the Product Mix problem, a BFS could be $x_1 = 0, x_2 = 0, s_1 = 4, s_2 = 5$, where $s_1$ and $s_2$ are slack variables representing unused resources.

---

## **4. Phases of Simplex**

### **Why Are There Two Phases?**
To solve a linear programming problem using Simplex, it must first be converted into a **standardized form**, which involves:
- All constraints expressed as equalities.
- Non-negative decision variables.

### **Phase 1**
Phase 1 focuses on finding an initial Basic Feasible Solution (BFS) by introducing artificial variables. If a feasible solution is found, the algorithm proceeds to Phase 2.

### **Phase 2**
Phase 2 optimizes the original objective function starting from the feasible solution identified in Phase 1.

---

## **5. Exploring Two Models with TableauVisualizer**

We use the **Product Mix Problem** to explore how the Simplex algorithm handles problems in Phase 1 and Phase 2:

### **Primal Model: Starting with Phase 2**
The primal model is already in standardized form and begins directly with Phase 2.

#### **Steps**:

1. **Define the Primal Model**

```python
from product_mix import create_model
model = create_model()
```

2. **Solve with TableauVisualizer**

```python
from totsu.core.tableau_visualizer import TableauVisualizer

visualizer = TableauVisualizer(model, use_jupyter=True)
visualizer.solve_model()
visualizer.show_tableau_visualization()
```

#### **What to Observe:**
- **Iteration Progress**: Each step represents a move to an adjacent vertex in the feasible region.
- **Optimality**: Observe how pivot operations improve the objective value.

---

### **Dual Model: Requiring Phase 1**
The dual model of the Product Mix Problem requires artificial variables to find a BFS before optimizing the dual objective.

#### **Steps**:

1. **Define the Dual Model**

```python
from product_mix import create_dual_model
dual_model = create_dual_model()
```

2. **Solve Phase 1 with TableauVisualizer**

```python
visualizer_dual = TableauVisualizer(dual_model, use_jupyter=True)
visualizer_dual.solve_model()
visualizer_dual.show_tableau_visualization()
```

#### **What to Observe:**
- **Artificial Variables**: Observe how artificial variables are introduced and minimized to find a feasible solution.
- **Transition to Phase 2**: See how the tableau changes as the algorithm switches to optimizing the dual objective.

---

## **6. Tutorial Workflow**

### **Step 1: Model Definition**
Define the LP model using Pyomo:

```python
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, maximize, NonNegativeReals

def create_model():
    model = ConcreteModel()
    # Decision Variables
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    
    # Objective Function
    model.obj = Objective(expr=3 * model.x1 + 2 * model.x2, sense=maximize)
    
    # Constraints
    model.c1 = Constraint(expr=model.x1 + model.x2 <= 4)
    model.c2 = Constraint(expr=2 * model.x1 + model.x2 <= 5)
    
    return model
```

---

### **Step 2: Solve with the Simplex Solver**
Use the Totsu Simplex Solver to solve the problem:

```python
from totsu.core.super_simplex_solver import SuperSimplexSolver

model = create_model()
solver = SuperSimplexSolver()
solution = solver.solve(model)

# Print results
print(f"Optimal Objective Value: {solver.get_current_objective_value()}")
for var in solution:
    print(f"{var} = {solution[var]}")
```

---

## **7. Expected Output**

- **Primal Model**:
  - Optimal Solution: $x_1 = 1$, $x_2 = 3$
  - Objective Value: $Z = 11$

- **Dual Model**:
  - Artificial variables are eliminated in Phase 1.
  - Optimal Dual Solution: $y_1, y_2, y_3$ representing shadow prices.

---

This concludes the introduction to the Simplex Solver. In the next tutorial, we will explore more advanced problems and visualization techniques.

