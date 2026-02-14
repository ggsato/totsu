## 1. High-Level BLC Strategy

1. **Bait (B)**: Lures in certain threats.  
2. **Leader (L)**: Punishes the threats that appear against B.  
3. **Cover (C)**: Protects the Leader from its own major threats, forming a chain B→L→C.

### 1.1. Pairwise Synergy: $\alpha$ and $\beta$

- $\alpha_{(B,L)}$: synergy from how well $L$ handles $B$’s weaknesses or offensive shortfalls.  
- $\beta_{(L,C)}$: synergy from how well $C$ handles $L$’s weaknesses.

### 1.2. Triple-based “Common Threat” Penalty: $\gamma$

Even if the pairwise synergy is high, all three might share a crippling weakness (e.g., triple Fire weakness). To fix that, we define $\gamma_{(B,L,C)}$ measuring something like:

- The **number** of threat types that remain uncovered for all three.  
- Or the **number** of “common threat” types that punish B, L, C in some sense.

We incorporate $\gamma$ as a **penalty** in the objective, preventing the solver from picking a triple that is simultaneously vulnerable to the same type(s).

---

## 2. Splitting Offense & Defense via Sets

Each Pokémon (or dual-type) $p$ is associated with four sets (or similarly computed logic):

1. **OP[p]** (Offensive Power): The set of types $p$ hits strongly.  
2. **DV[p]** (Defensive Vulnerability): The set of types that do big damage to $p$.  
3. **OI[p]** (Offensive Ineffectiveness): The set of types that $p$ can’t hurt effectively.  
4. **DR[p]** (Defensive Resilience): The set of types that do little damage to $p$.

### 2.1. Pairwise Synergy Functions

$$
\alpha_{(B,L)} 
= 
w_1 \,\bigl|\mathrm{DV}[B] \cap \mathrm{OP}[L]\bigr|
\;+\;
w_2 \,\bigl|\mathrm{OI}[B] \cap \mathrm{DR}[L]\bigr|,
$$
$$
\beta_{(L,C)} 
=
w_3 \,\bigl|\mathrm{DV}[L] \cap \mathrm{OP}[C]\bigr|
\;+\;
w_4 \,\bigl|\mathrm{OI}[L] \cap \mathrm{DR}[C]\bigr|.
$$

### 2.2. Triple-based Penalty Function

$$
\gamma_{(B,L,C)} 
= 
\text{(number of common threats / uncovered weaknesses / etc.)}
$$
for example:

- $\mathrm{BL\_threats} = \mathrm{DV}[B]\setminus \mathrm{DR}[L]$  
- $\mathrm{LC\_threats} = \mathrm{DV}[L]\setminus \mathrm{DR}[C]$  
- $\mathrm{CB\_threats} = \mathrm{DV}[C]\setminus \mathrm{DR}[B]$  
- $\mathrm{common\_threats} = \mathrm{BL\_threats}\cap \mathrm{LC\_threats}\cap \mathrm{CB\_threats}$.  
- Then $\gamma_{(B,L,C)} = \lvert\mathrm{common\_threats}\rvert$.

---

## 3. Decision Variables

We pick exactly one Bait, one Leader, one Cover from a set of types $\mathcal{P}$. Each element of $\mathcal{P}$ may be a **single type** $(t, None)$ or a **dual type** $(t_0, t_1)$, if `support_dual_types=True`.

Define binary variables:

$$
x_p^B,\; x_p^L,\; x_p^C \in \{0,1\}\quad \forall p \in \mathcal{P},
$$
where $x_p^B =1$ if $p$ is chosen as Bait, etc. Subject to:

1. Exactly one Bait, Leader, Cover:
   $$
   \sum_{p\in \mathcal{P}} x_p^B=1,\quad
   \sum_{p\in \mathcal{P}} x_p^L=1,\quad
   \sum_{p\in \mathcal{P}} x_p^C=1.
   $$
2. Distinct roles:
   $$
   x_p^B + x_p^L + x_p^C \le 1,\;\forall p\in\mathcal{P}.
   $$

We also define:

- **zBL[b,l]** $\approx xB[b]*xL[l]$ (the product).  
- **zLC[l,c]** $\approx xL[l]*xC[c]$.  
- **zTri[b,l,c]** $\approx xB[b]*xL[l]*xC[c]$.

All these are done with **auxiliary** binary variables and the usual linear “linking constraints”:

$$
zBL[b,l]\le xB[b],\;\; zBL[b,l]\le xL[l],\;\; zBL[b,l]\ge xB[b]+xL[l]-1,
$$
(and similarly for zLC, zTri with 3 constraints + a bigger “-2” on the last inequality).

---

## 4. Objective Function

We incorporate the synergy terms and the triple-based penalty. For instance:

$$
\max\;
\Bigl(
\sum_{(b,l)} \alpha_{(b,l)} \cdot zBL[b,l]
\;+\;
\sum_{(l,c)} \beta_{(l,c)} \cdot zLC[l,c]
\;-\;
\sum_{(b,l,c)} \gamma_{(b,l,c)} \cdot zTri[b,l,c]
\Bigr).
$$

### 4.1. Summarized

- $\alpha_{(b,l)}$ is the B→L synergy, multiplied by zBL[b,l].  
- $\beta_{(l,c)}$ is the L→C synergy, multiplied by zLC[l,c].  
- $\gamma_{(b,l,c)}$ is the triple-based penalty (the number of “common threats”), multiplied by zTri[b,l,c].  

We choose the triple that *maximizes* synergy minus penalty.

---

## 5. Example Final Pseudocode

**Initialization**:

1. Build your sets for each type: $\mathrm{OP}[p], \mathrm{DV}[p], \mathrm{OI}[p], \mathrm{DR}[p]$.  
2. Build param $\alpha_{p,q}$, $\beta_{q,r}$ for each pair, $\gamma_{p,q,r}$ for each triple.  
   - If `support_dual_types=True`, define suitable `alpha_init, beta_init, gamma_init` that handle dimension=2.  

**Model**:

1. `model.P` is a set (dimension=1 or 2).  
2. `model.xB[p], model.xL[p], model.xC[p]` are binary.  
3. `model.zBL[b,l], model.zLC[l,c], model.zTri[b,l,c]` are binary for all relevant pairs/triples.  
4. Link them with the usual constraints so each `z` equals the product.  
5. The objective is the synergy expression, e.g.:

   ```python
   def synergy_expr(m):
       synergy = sum(m.alpha[b,l] * m.zBL[b,l] for b in m.P for l in m.P) \
               + sum(m.beta[l,c]  * m.zLC[l,c] for l in m.P for c in m.P)
       penalty = sum(m.gamma[b,l,c] * m.zTri[b,l,c] for b in m.P for l in m.P for c in m.P)
       return synergy - penalty
   model.obj = Objective(rule=synergy_expr, sense=maximize)
   ```

**Solution**:

- The solver picks exactly one Bait, Leader, Cover, forms the selected pairwise and triplewise synergy, penalizes triple vulnerabilities, etc.  
- You retrieve your chosen B, L, C by finding which `xB[b]`, `xL[l]`, `xC[c]` are =1.
