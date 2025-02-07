# Introduction to Totsu Project

Welcome to the **Totsu Project**, a powerful suite of tools for exploring, visualizing, and solving linear programming (LP) problems using the Simplex algorithm. Designed for educators, students, and practitioners, Totsu makes complex optimization concepts accessible and interactive.

---

## **Why Totsu?**

Linear programming plays a critical role in solving real-world problems across industries like supply chain management, finance, and production planning. However, understanding the nuances of LP, such as tableau operations, sensitivity analysis, and duality, often feels daunting.

Totsu simplifies this process by offering:
- **Interactive Visualization**: Explore Simplex tableau and sensitivity analysis dynamically.
- **Educational Focus**: Gain insights into LP concepts like duality, pivot operations, and feasibility.
- **Customization**: Tailor models and analyses for specific use cases.
- **Open-Source Accessibility**: Fully transparent and extensible for your needs.

---

## **Key Features**

### 1. **TableauVisualizer**
- **Visualize Simplex Tableau**:
  - Understand how the Simplex method progresses iteration by iteration.
  - Learn about pivot operations for improving objectives and ensuring feasibility.
- **Educational Insights**:
  - Highlight entering and leaving variables.
  - Show constraints and basic variable transitions in real time.

### 2. **SensitivityAnalyzer**
- **Duality and Sensitivity**:
  - Visualize the impact of changing constraints on the objective.
  - Explore ridge and valley behaviors for primal and dual models.
- **Solver Flexibility**:
  - Use different solvers (e.g., SuperSimplexSolver, GLPK).

### 3. **Practical Examples**
- Pre-built models include:
  - **Product Mix**: Maximize profits while balancing resource constraints.
  - **Blending**: Minimize costs while meeting product quality specifications.
  - **Minimum Cost Flow**: Optimize transportation networks.

---

## **Who Should Use Totsu?**

Totsu is for:
- **Educators**: Demonstrate optimization concepts in classrooms with interactive examples.
- **Students**: Learn LP and the Simplex method through hands-on exploration.
- **Practitioners**: Quickly analyze and solve LP problems with sensitivity insights.

---

## **Getting Started**

### Installation

1. **Install via Conda** (Recommended):
   ```bash
   conda create -n totsu-env python=3.9
   conda activate totsu-env
   pip install -r requirements.txt
   ```

2. **Optional: Install JupyterBook for Tutorials**:
   ```bash
   pip install jupyter-book
   ```

### Repository
Explore the [Totsu GitHub Repository](https://github.com/your-repo-url) for examples and documentation.

### Tutorials
Jump into practical tutorials:
1. **Product Mix**: Maximize profits while exploring duality.
2. **Blending**: Minimize costs and meet product constraints.
3. **Minimum Cost Flow**: Optimize transportation networks.

---

## **Totsu: Learning Through Visualization**

Totsu bridges the gap between theory and practice, empowering users to:
- Visualize optimization processes.
- Explore sensitivity and duality insights.
- Solve real-world problems interactively.

Join us on this journey to make linear programming intuitive and engaging!

