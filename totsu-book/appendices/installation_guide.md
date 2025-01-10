# Installation Guide for Totsu Project

Follow these steps to install and set up the Totsu Project for exploring linear programming problems.

---

## **System Requirements**
- **Operating System**: Linux, macOS, or Windows
- **Python Version**: Python 3.8 or later
- **Dependencies**:
  - `Pyomo` for modeling.
  - `Dash` and `Plotly` for visualization.

---

## **Installation Steps**

### **1. Using Conda (Recommended)**
```bash
# Create a new environment
conda create -n totsu-env python=3.9
conda activate totsu-env

# Install requirements
pip install -r requirements.txt
```

### **2. Installing JupyterBook (Optional)**
```bash
pip install jupyter-book
```

### **3. Installing Optional Solvers**
```bash
conda install -c conda-forge glpk
```

---

## **Verifying the Installation**
Run the following command to verify your setup:
```bash
python -m unittest discover tests
```

You should see all tests passing if the installation is successful.

---

## **Troubleshooting**
- **Issue**: Missing dependency error.
  - **Solution**: Ensure all dependencies in `requirements.txt` are installed.
- **Issue**: Solver not found.
  - **Solution**: Install the required solver using Conda or system package managers.

For further assistance, visit the [Totsu GitHub Issues Page](https://github.com/your-repo-url/issues).
```

---
