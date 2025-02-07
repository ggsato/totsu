# Pokémon Team Selection Optimization

This project provides a **Mathematical Optimization Model** using **Integer Linear Programming (ILP)** with **Pyomo** to optimize Pokémon team selection in **Pokémon GO battles**. The model selects an optimal team of Pokémon types based on **attack, defense, or balanced strategies** against a given set of opponent Pokémon.

## 📌 Features
- **Team Selection Optimization**: Selects a team of Pokémon types based on the given opponent types.
- **Three Optimization Strategies**:
  - **Maximize Attack**: Focuses on choosing Pokémon that deal the most damage to the opponent.
  - **Maximize Defense**: Selects Pokémon that resist attacks from the opponent.
  - **Balanced Approach**: A weighted combination of attack and defense.
- **Support for Dual-Type Pokémon**:
  - **Defense Mechanics**: If a Pokémon has two types, its effectiveness is **combined**.
  - **Attack Mechanics**: Attack effectiveness remains **single-type**.
- **AbstractModel-based Pyomo Implementation**:
  - Allows flexible input of Pokémon type effectiveness data.
  - Uses **Binary Decision Variables** to ensure each type is either selected or not.

## 📂 File Structure
```
📁 pokemon_team_selection/
│── 📝 README.md            # Project documentation (this file)
│── 🏗️ pokemon_team_selection.py  # Main Pyomo optimization model
│── 🔬 test_pokemon_team_selection.py  # Test cases for validation
│── 📊 data/                 # Type effectiveness matrix and other input data (TODO)
```

---

## ⚙️ Installation

### **Prerequisites**
Ensure you have Python installed and set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### **Install Dependencies**
```bash
pip install pyomo pytest
```

For solving ILP problems, install a solver like **GLPK**:
```bash
conda install -c conda-forge glpk  # If using Conda
pip install glpk  # If available via pip
```

---

## 🚀 Usage

### **Basic Usage**
Run the script with arguments specifying:
1. **Objective type**: `attack`, `defense`, or `balanced`
2. **Opponent Pokémon types**: 1 to 3 types (can include dual types)
3. **Optional team size** (default: 3)

```bash
python pokemon_team_selection.py attack Fire Water Grass
```

Example for a dual-type opponent:
```bash
python pokemon_team_selection.py defense "Flying-Electric"
```

Specify a team size (e.g., `4` instead of default `3`):
```bash
python pokemon_team_selection.py balanced Psychic Fairy --team_size 4
```

---

## 📖 Explanation of the Model

The model **optimizes the team selection** using a **binary variable** approach:

### **Decision Variables**
Each Pokémon type $i$ is either **selected (1)** or **not selected (0)**:

$$
x_i \in \{0,1\}, \quad \forall i \in T
$$

where $T$ is the set of all Pokémon types.

### **Objective Functions**
#### **Attack Maximization**

$$
\max \sum_{i \in T} x_i \sum_{j \in S} A_{ij}
$$

where $A_{ij}$ is the **effectiveness** of type $i$ against opponent type $j$.

#### **Defense Maximization**

$$
\min \sum_{j \in S} \sum_{i \in T} A_{ij} \cdot x_i
$$

This minimizes the sum of opponent attacks against the chosen Pokémon.

#### **Balanced Strategy**

$$
\max \lambda \sum_{i \in T} x_i \sum_{j \in S} A_{ij} - (1-\lambda) \sum_{j \in S} \sum_{i \in T} A_{ij} \cdot x_i
$$

where $\lambda$ is a **user-defined balance factor** (default: 0.5).

### **Constraints**
#### **Team Size Constraint**
Ensures exactly $N$ Pokémon types are selected:

$$
\sum_{i \in T} x_i = N
$$

---

## 🔬 Testing
Run the test cases to validate the implementation:
```bash
pytest test_pokemon_team_selection.py
```

---

## 📜 License
This project is **open-source** under the MIT License.

---

## 🤝 Contributing
If you have ideas to improve the model or add new features, feel free to submit a pull request!
