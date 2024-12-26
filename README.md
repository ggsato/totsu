# **Totsu (凸): A Visual Approach to Linear Programming**

**Totsu** (凸), meaning "convex" in Japanese, is an open-source project designed to revolutionize how the Simplex method is understood, taught, and applied. It combines the power of efficient linear programming with **interactive visualizations** that demystify the optimization process for students, educators, and practitioners.

With **Totsu**, users can:
- Solve linear programming problems with a modular and extensible Simplex solver.
- Explore optimization concepts interactively through visualization tools like the **TableauVisualizer** and **SensitivityAnalyzer**.
- Gain deep insights into feasible regions, duality, and sensitivity analysis without needing to be an LP expert.

Totsu transforms optimization into an engaging, visual experience, making it ideal for both educational and practical applications.

---

## **Key Features**

### **1. Intuitive Tableau Visualization**
- Step-by-step visualization of the Simplex method, highlighting:
  - **Entering variables** to improve the objective.
  - **Leaving variables** to maintain feasibility.
  - The entire tableau evolution across iterations.
- Perfect for **teaching and learning** the mechanics of the Simplex algorithm interactively.

![Tableau Visualization](resources/tableau_vis.gif)

---

### **2. Interactive Sensitivity Analysis**
- Explore how changes in constraint values impact the objective function with:
  - **3D Objective Value Surface**: Valid ranges for significant dual variables visualized interactively.
  - **Ridge Analysis**: Trace optimal paths dynamically with gradient-guided animations.

![Sensitivity Analysis](resources/sensitivity_analysis.gif)
---

### **3. Seamless Integration with Pyomo**
- Designed to work alongside **Pyomo**, a popular open-source optimization modeling library.
- Easily integrates into workflows involving dynamic constraints or Branch and Bound for integer programming.

---

## **Why Totsu?**

- **For Educators**: Bring optimization to life with interactive tools that simplify teaching the Simplex algorithm and LP concepts.
- **For Students**: Master linear programming by visualizing each step, from tableau updates to sensitivity analysis.
- **For Practitioners**: Gain actionable insights into real-world decision-making problems, such as supply chain optimization or resource allocation.

---

## **Installation**

Totsu can be installed via Conda for robust dependency management.

```bash
conda install -c conda-forge pyomo numpy pytest plotly dash dash-bootstrap-components
pytest
```

For **Jupyter Notebook** support:
```bash
conda install jupyterlab
```

---

## **Quick Start**

### **1. TableauVisualizer**
Launch the TableauVisualizer to explore the Simplex method step by step:
```bash
python3 -m totsu.examples.model_building_imp.examples.chp1_1_product_mix.product_mix_tableau_visualization
```

### **2. SensitivityAnalyzer**
Dive into sensitivity analysis and visualize objective value surfaces:
```bash
python3 -m totsu.examples.model_building_imp.examples.chp1_1_product_mix.product_mix_visualization
```

### **3. Jupyter Support**
Open `demo.ipynb` in **JupyterLab** to run visualizers interactively:
```bash
jupyter lab
```

---

## **Roadmap and Vision**

Totsu aims to make **linear programming** accessible and practical for everyone:
- **Education**: Empower universities and online learning platforms with an interactive tool for teaching LP.
- **Small Businesses**: Simplify decision-making by visualizing sensitivity to constraints.
- **Open Source**: Build a vibrant community around LP visualization and optimization.

---

## **Contributing**
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved.

---

## **License**
Totsu is licensed under the [MIT License](LICENSE), ensuring it remains accessible and open for all.

---