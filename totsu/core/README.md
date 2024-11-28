# Common Questions

## How does totsu compute and store dual variables and reduced costs?

### **1. Computing Dual Variables (Shadow Prices)**

In linear programming, dual variables can be computed using the final simplex tableau. Specifically, the dual variables $y$ are calculated using the relation:

$$
y^T = c_B^T B^{-1}
$$

Where:
- $c_B$ is the vector of objective coefficients for the basic variables.
- $B^{-1}$ is the inverse of the basis matrix $B$ formed by the columns of the constraints corresponding to the basic variables.

Since computing $B^{-1}$ directly can be computationally intensive, we can leverage the fact that in the final simplex tableau, the rows (excluding the last row) contain the information needed to reconstruct $B^{-1}$.

### **2. Computing Reduced Costs**

The reduced cost for a non-basic variable $j$ is given by:

$$
\text{Reduced Cost}_j = c_j - y^T A_j
$$

Where:
- $c_j$ is the objective coefficient of variable $j$.
- $A_j$ is the column of the constraint matrix corresponding to variable $j$.