from pyomo.environ import ConcreteModel, AbstractModel, Set, Param, Var, Objective, Constraint, ConstraintList, SolverFactory, maximize, minimize, Binary, Any, SolverStatus
from ...core.totsu_branch_and_bound_solver import TotsuBranchAndBoundSolver # required to register

POKEMON_TYPES = ["Normal", "Fire", "Water", "Grass", "Electric", "Fighting", "Poison", "Ground", "Flying", 
                 "Psychic", "Bug", "Rock", "Ghost", "Ice", "Dragon", "Dark", "Steel", "Fairy"]
MIXED_POKEMON_TYPES = [(t, None) for t in POKEMON_TYPES] + [(t1, t2) for t1 in POKEMON_TYPES for t2 in POKEMON_TYPES if t1 != t2]
EFFECTIVENESS = {
    "S": 1.6, # Super effective
    "O": 1.0, # Normal effectiveness
    "N": 0.626, # Not very effective
    "I": 0.625*0.625 # Immune
}
# See: Type Effectiveness in Battle
# https://niantic.helpshift.com/hc/en/6-pokemon-go/faq/2132-type-effectiveness-in-battle/
MATCHUP_MATRIX = {
    "Normal":   {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "N", 
                 "Ghost": "I", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Fire":     {"Normal": "O", "Fire": "N", "Water": "N", "Grass": "S", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "S", "Rock": "N", 
                 "Ghost": "O", "Ice": "S", "Dragon": "N", "Dark": "O", "Steel": "S", "Fairy": "O"},
    "Water":    {"Normal": "O", "Fire": "S", "Water": "N", "Grass": "N", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "S", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "S", 
                 "Ghost": "O", "Ice": "O", "Dragon": "N", "Dark": "O", "Steel": "O", "Fairy": "O"},
    "Grass":    {"Normal": "O", "Fire": "N", "Water": "S", "Grass": "N", "Electric": "O", "Fighting": "O", 
                 "Poison": "N", "Ground": "S", "Flying": "N", "Psychic": "O", "Bug": "N", "Rock": "S", 
                 "Ghost": "O", "Ice": "O", "Dragon": "N", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Electric":    {"Normal": "O", "Fire": "O", "Water": "S", "Grass": "N", "Electric": "N", "Fighting": "O", 
                 "Poison": "O", "Ground": "I", "Flying": "S", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "N", "Dark": "O", "Steel": "O", "Fairy": "O"},
    "Fighting":    {"Normal": "S", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "N", "Ground": "O", "Flying": "N", "Psychic": "N", "Bug": "N", "Rock": "S", 
                 "Ghost": "I", "Ice": "S", "Dragon": "O", "Dark": "S", "Steel": "S", "Fairy": "N"},
    "Poison":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "S", "Electric": "O", "Fighting": "O", 
                 "Poison": "N", "Ground": "N", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "N", 
                 "Ghost": "N", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "I", "Fairy": "S"},
    "Ground":    {"Normal": "O", "Fire": "S", "Water": "O", "Grass": "N", "Electric": "S", "Fighting": "O", 
                 "Poison": "S", "Ground": "O", "Flying": "I", "Psychic": "O", "Bug": "N", "Rock": "S", 
                 "Ghost": "O", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "S", "Fairy": "O"},
    "Flying":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "S", "Electric": "N", "Fighting": "S", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "S", "Rock": "N", 
                 "Ghost": "O", "Ice": "O", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Psychic":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "S", 
                 "Poison": "S", "Ground": "O", "Flying": "O", "Psychic": "N", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "O", "Dark": "I", "Steel": "N", "Fairy": "O"},
    "Bug":    {"Normal": "O", "Fire": "N", "Water": "O", "Grass": "S", "Electric": "O", "Fighting": "N", 
                 "Poison": "N", "Ground": "O", "Flying": "N", "Psychic": "S", "Bug": "O", "Rock": "O", 
                 "Ghost": "N", "Ice": "O", "Dragon": "O", "Dark": "S", "Steel": "N", "Fairy": "N"},
    "Rock":    {"Normal": "O", "Fire": "S", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "N", 
                 "Poison": "O", "Ground": "N", "Flying": "S", "Psychic": "O", "Bug": "S", "Rock": "O", 
                 "Ghost": "O", "Ice": "S", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Ghost":    {"Normal": "I", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "S", "Bug": "O", "Rock": "O", 
                 "Ghost": "S", "Ice": "O", "Dragon": "O", "Dark": "N", "Steel": "O", "Fairy": "O"},
    "Ice":    {"Normal": "O", "Fire": "N", "Water": "N", "Grass": "S", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "S", "Flying": "S", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "N", "Dragon": "S", "Dark": "O", "Steel": "N", "Fairy": "O"},
    "Dragon":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "S", "Dark": "O", "Steel": "N", "Fairy": "I"},
    "Dark":    {"Normal": "O", "Fire": "O", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "N", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "S", "Bug": "O", "Rock": "O", 
                 "Ghost": "S", "Ice": "O", "Dragon": "O", "Dark": "N", "Steel": "O", "Fairy": "N"},
    "Steel":    {"Normal": "O", "Fire": "N", "Water": "N", "Grass": "O", "Electric": "N", "Fighting": "O", 
                 "Poison": "O", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "S", 
                 "Ghost": "O", "Ice": "S", "Dragon": "O", "Dark": "O", "Steel": "N", "Fairy": "S"},
    "Fairy":    {"Normal": "O", "Fire": "N", "Water": "O", "Grass": "O", "Electric": "O", "Fighting": "S", 
                 "Poison": "N", "Ground": "O", "Flying": "O", "Psychic": "O", "Bug": "O", "Rock": "O", 
                 "Ghost": "O", "Ice": "O", "Dragon": "S", "Dark": "S", "Steel": "N", "Fairy": "O"},
}
def effect(atk_type, def_type):
    # e.g., from your MATCHUP_MATRIX
    code = MATCHUP_MATRIX[atk_type][def_type]
    return EFFECTIVENESS[code]

def effect_dual(atk, defn0, defn1):
    """Handles both single-type and dual-type defenders."""
    if defn1 is None:
        return EFFECTIVENESS[MATCHUP_MATRIX[atk][defn0]]  # Single type case
    return EFFECTIVENESS[MATCHUP_MATRIX[atk][defn0]] * EFFECTIVENESS[MATCHUP_MATRIX[atk][defn1]]

def build_sets_for_type(t):
    """
    Returns a tuple of (OP[t], DV[t], OI[t], DR[t]) for a single type t.
    - OP[t]: set of types that t hits for >1.0 multiplier
    - DV[t]: set of types that hit t for >1.0 multiplier
    - OI[t]: set of types that t hits for <1.0 multiplier
    - DR[t]: set of types that hit t for <1.0 multiplier
    """
    OP_t   = set()  # Offensive Power
    DV_t   = set()  # Defensive Vulnerability
    OI_t   = set()  # Offensive Ineffectiveness
    DR_t   = set()  # Defensive Resilience

    # We'll loop over all possible types in your POKEMON_TYPES
    for x in POKEMON_TYPES:
        eff_tx = effect(t, x)   # how well t -> x
        eff_xt = effect(x, t)   # how well x -> t
        
        # Offensive sets
        if eff_tx > 1.0:
            OP_t.add(x)
        elif eff_tx < 1.0:
            OI_t.add(x)
        
        # Defensive sets
        if eff_xt > 1.0:
            DV_t.add(x)
        elif eff_xt < 1.0:
            DR_t.add(x)

    return (OP_t, DV_t, OI_t, DR_t)

def build_sets_for_dualtype(t0, t1):
    """
    Returns a tuple of (OP[t], DV[t], OI[t], DR[t]) for a dual type t.
    - OP[t]: set of types that t hits for >1.0 multiplier
    - DV[t]: set of types that hit t for >1.0 multiplier
    - OI[t]: set of types that t hits for <1.0 multiplier
    - DR[t]: set of types that hit t for <1.0 multiplier
    """
    OP_t   = set()  # Offensive Power
    DV_t   = set()  # Defensive Vulnerability
    OI_t   = set()  # Offensive Ineffectiveness
    DR_t   = set()  # Defensive Resilience

    # We'll loop over all possible types in your POKEMON_TYPES
    for x in POKEMON_TYPES:
        eff_tx = effect_dual(t0, x, None)   # how well t -> x
        eff_xt = effect_dual(x, t0, t1)   # how well x -> t
        
        # Offensive sets
        if eff_tx > 1.0:
            OP_t.add(x)
        elif eff_tx < 1.0:
            OI_t.add(x)
        
        # Defensive sets
        if eff_xt > 1.0:
            DV_t.add(x)
        elif eff_xt < 1.0:
            DR_t.add(x)

    return (OP_t, DV_t, OI_t, DR_t)

def build_all_sets(support_dual_types=False):
    # dictionary for all types
    OP = {}
    DV = {}
    OI = {}
    DR = {}

    if not support_dual_types:
        for t in POKEMON_TYPES:
            (OP[t], DV[t], OI[t], DR[t]) = build_sets_for_type(t)
    else:
        # Consider single type and dual types
        for t0, t1 in MIXED_POKEMON_TYPES:
            (OP[t0, t1], DV[t0, t1], OI[t0, t1], DR[t0, t1]) = build_sets_for_dualtype(t0, t1)
    return OP, DV, OI, DR

def explain_choice(B, L, C, OP, DV, OI, DR):
    # W and S are dicts: W[p] = set of types that beat p, S[p] = set of types p can beat
    b_weak_set = DV[B].union(OI[B])
    l_weak_set = DV[L].union(OI[L])
    c_weak_set = DV[C].union(OI[C])
    BL_intersect_counter = b_weak_set.intersection(OP[L].union(DR[L]))
    LC_intersect_counter = l_weak_set.intersection(OP[C].union(OP[C]))
    BC_intersect_counter = b_weak_set.intersection(OP[C].union(OP[C]))

    B_L_C_common_DV = b_weak_set.intersection(l_weak_set).intersection(c_weak_set)

    print(f"\nSelected Bait = {B}, Leader = {L}, Cover = {C}")
    print(f"Bait->Leader synergy: The types that punish {B} but are punished by {L} = {BL_intersect_counter}")
    print(f"Leader->Cover synergy: The types that punish {L} but are punished by {C} = {LC_intersect_counter}")
    print(f"Bait->Cover synergy: The types that punish {B} but are punished by {C} = {BC_intersect_counter}")
    print(f"Common threat: {B_L_C_common_DV}")
    print(f"DV[{B}] = {DV[B]}")
    print(f"DV[{L}] = {DV[L]}")
    print(f"DV[{C}] = {DV[C]}")

class BLCTeamSelection:
    def __init__(self, support_dual_types=False):
        self.support_dual_types = support_dual_types
        self.pokemon_types = None

        self.OP = None
        self.DV = None
        self.OI = None
        self.DR = None

        self.model = None

    def build(self, pokemon_types):
        self.pokemon_types = self.parse_pokemon_types(pokemon_types) if self.support_dual_types else pokemon_types
        self.OP, self.DV, self.OI, self.DR = build_all_sets(self.support_dual_types)
        self.model = self.create_model()

    @staticmethod
    def parse_pokemon_types(pokemon_types):
        """Parse opponent Pokémon types (supports dual types)."""
        parsed_types = []
        for opt in pokemon_types:
            types = opt.split("-")
            if len(types) == 1 and types[0] in POKEMON_TYPES:
                parsed_types.append((types[0], None))
            elif len(types) == 2 and all(t in POKEMON_TYPES for t in types):
                parsed_types.append((types[0], types[1]))
            else:
                raise ValueError(f"Invalid type: {opt}")
        return parsed_types

    def synergy_alpha(self, B, L):
        """
        synergy_alpha(B,L):
        Uses the union of B's weaknesses (DV[B] ∪ OI[B]) and 
        the union of L's strengths (OP[L] ∪ DR[L]),
        returning |(DV[B] ∪ OI[B]) ∩ (OP[L] ∪ DR[L])|.
        """

        b_weak_set = self.DV[B].union(self.OI[B])
        l_strong_set = self.OP[L].union(self.DR[L])

        return len(b_weak_set.intersection(l_strong_set))

    def synergy_beta(self, L, C):
        """
        synergy_beta(L,C):
        Uses the union of L's weaknesses (DV[L] ∪ OI[L]) and 
        the union of C's strengths (OP[C] ∪ DR[C]),
        returning |(DV[L] ∪ OI[L]) ∩ (OP[C] ∪ DR[C])|.
        """

        l_weak_set = self.DV[L].union(self.OI[L])
        c_strong_set = self.OP[C].union(self.DR[C])

        return len(l_weak_set.intersection(c_strong_set))

    def synergy_gamma(self, B, C):
        """
        synergy_gamma(B,C):
        Uses the union of B's weaknesses (DV[B] ∪ OI[B]) and 
        the union of C's strengths (OP[C] ∪ DR[C]),
        returning |(DV[B] ∪ OI[B]) ∩ (OP[C] ∪ DR[C])|.
        """

        b_weak_set = self.DV[B].union(self.OI[B])
        c_strong_set = self.OP[C].union(self.DR[C])

        return len(b_weak_set.intersection(c_strong_set))

    def penalty(self, B, L, C):
        """
        penalty(B,L,C) = 
        """
        
        b_weak_set = self.DV[B].union(self.OI[B])
        l_weak_set = self.DV[L].union(self.OI[L])
        c_weak_set = self.DV[C].union(self.OI[C])
        
        common_threats = b_weak_set.intersection(l_weak_set).intersection(c_weak_set)

        return len(common_threats)

    def create_model(self):
        # Build the Pyomo model
        model = ConcreteModel("BLC_Model")

        # Choose among P
        if not self.support_dual_types:
            model.P = Set(initialize=self.pokemon_types)
        else:
            model.P = Set(
                dimen=2,
                initialize = self.pokemon_types
            )

        # Define alpha, beta as Param
        if not self.support_dual_types:
            def alpha_init(m, p, q):
                return self.synergy_alpha(p, q)

            def beta_init(m, q, r):
                return self.synergy_beta(q, r)

            def gamma_init(m, p, r):
                return self.synergy_gamma(p, r)

            def penalty_init(m, p, q, r):
                return self.penalty(p, q, r)
        else:
            def alpha_init(m, p0, p1, q0, q1):
                return self.synergy_alpha((p0, p1), (q0, q1))

            def beta_init(m, q0, q1, r0, r1):
                return self.synergy_beta((q0, q1), (r0, r1))

            def gamma_init(m, p0, p1, r0, r1):
                return self.synergy_gamma((p0, p1), (r0, r1))

            def penalty_init(m, p0, p1, q0, q1, r0, r1):
                return self.penalty((p0, p1), (q0, q1), (r0, r1))

        model.alpha = Param(model.P, model.P, initialize=alpha_init, default=0)
        model.beta  = Param(model.P, model.P, initialize=beta_init, default=0)
        model.gamma  = Param(model.P, model.P, initialize=gamma_init, default=0)
        model.penalty = Param(model.P, model.P, model.P, initialize=penalty_init, default=0)

        # Binary variables: xB, xL, xC
        model.xB = Var(model.P, within=Binary)  # Bait
        model.xL = Var(model.P, within=Binary)  # Leader
        model.xC = Var(model.P, within=Binary)  # Cover

        # "zBL[b,l]" = xB[b]*xL[l]
        model.zBL = Var(model.P, model.P, within=Binary) 
        # "zLC[l,c]" = xL[l]*xC[c]
        model.zLC = Var(model.P, model.P, within=Binary)
        # "zBC[b,c]" = xB[b]*xC[c]
        model.zBC = Var(model.P, model.P, within=Binary)
        # zTri[b,l,c] = xB[b]*xL[l]*xC[c]
        model.zTri = Var(model.P, model.P, model.P, within=Binary)

        # ========== Linking Constraints ==========

        model.link_zBL = ConstraintList()
        for b in model.P:
            for l in model.P:
                # zBL[b,l] <= xB[b]
                model.link_zBL.add(model.zBL[b,l] <= model.xB[b])
                # zBL[b,l] <= xL[l]
                model.link_zBL.add(model.zBL[b,l] <= model.xL[l])
                # zBL[b,l] >= xB[b] + xL[l] - 1
                model.link_zBL.add(model.zBL[b,l] >= model.xB[b] + model.xL[l] - 1)

        model.link_zLC = ConstraintList()
        for l in model.P:
            for c in model.P:
                # zLC[l,c] <= xL[l]
                model.link_zLC.add(model.zLC[l,c] <= model.xL[l])
                # zLC[l,c] <= xC[c]
                model.link_zLC.add(model.zLC[l,c] <= model.xC[c])
                # zLC[l,c] >= xL[l] + xC[c] - 1
                model.link_zLC.add(model.zLC[l,c] >= model.xL[l] + model.xC[c] - 1)

        model.link_zBC = ConstraintList()
        for b in model.P:
            for c in model.P:
                # zBC[b,c] <= xB[b]
                model.link_zBC.add(model.zBC[b,c] <= model.xB[b])
                # zBC[b,c] <= xC[c]
                model.link_zBC.add(model.zBC[b,c] <= model.xC[c])
                # zBC[b,c] >= xB[l] + xC[c] - 1
                model.link_zBC.add(model.zBC[b,c] >= model.xB[b] + model.xC[c] - 1)

        # Linking constraints so zTri[b,l,c] = 1 ⇔ xB[b]=1 & xL[l]=1 & xC[c]=1
        model.link_zTri = ConstraintList()
        for b in model.P:
            for l in model.P:
                for c in model.P:
                    # zTri[b,l,c] <= xB[b]
                    model.link_zTri.add(model.zTri[b,l,c] <= model.xB[b])
                    # zTri[b,l,c] <= xL[l]
                    model.link_zTri.add(model.zTri[b,l,c] <= model.xL[l])
                    # zTri[b,l,c] <= xC[c]
                    model.link_zTri.add(model.zTri[b,l,c] <= model.xC[c])
                    # zTri[b,l,c] >= xB[b] + xL[l] + xC[c] - 2
                    model.link_zTri.add(
                        model.zTri[b,l,c] >= model.xB[b] + model.xL[l] + model.xC[c] - 2
                    )

        # ========== Exactly 1 Bait/Leader/Cover constraints ==========

        model.one_bait   = Constraint(expr=sum(model.xB[p] for p in model.P) == 1)
        model.one_leader = Constraint(expr=sum(model.xL[p] for p in model.P) == 1)
        model.one_cover  = Constraint(expr=sum(model.xC[p] for p in model.P) == 1)

        if not self.support_dual_types:
            def distinct_role_rule(m, p):
                return m.xB[p] + m.xL[p] + m.xC[p] <= 1
        else:
            def distinct_role_rule(m, p0, p1):
                return m.xB[(p0, p1)] + m.xL[(p0, p1)] + m.xC[(p0, p1)] <= 1
        model.distinct_role = Constraint(model.P, rule=distinct_role_rule)

        # ========== No shared weakness between L and C ==============
        if not self.support_dual_types:
            def no_shared_weakness_lc(m, q, r):
                return len(self.DV[q].intersection(self.DV[r])) * m.zLC[q, r] <= 0
        else:
            def no_shared_weakness_lc(m, q0, q1, r0, r1):
                return len(self.DV[(q0, q1)].intersection(self.DV[(r0, r1)])) * m.zLC[(q0, q1), (r0, r1)] <= 0
        model.no_shared_weakness_lc = Constraint(model.P, model.P, rule=no_shared_weakness_lc)

        # ========== No shared weakness between B and C ==============
        if not self.support_dual_types:
            def no_shared_weakness_bc(m, p, r):
                return len(self.DV[p].intersection(self.DV[r])) * m.zBC[p, r] <= 0
        else:
            def no_shared_weakness_bc(m, p0, p1, r0, r1):
                return len(self.DV[(p0, p1)].intersection(self.DV[(r0, r1)])) * m.zBC[(p0, p1), (r0, r1)] <= 0
        model.no_shared_weakness_bc = Constraint(model.P, model.P, rule=no_shared_weakness_bc)

        # ========== Objective: sum of alpha and beta with auxiliary variables ==========

        def synergy_expr(m):
            synergy = (
                sum(m.alpha[b,l] * m.zBL[b,l] for b in m.P for l in m.P)
                +
                sum(m.beta[l,c] * m.zLC[l,c] for l in m.P for c in m.P)
                +
                sum(m.gamma[b,c] * m.zBC[b,c] for b in m.P for c in m.P)
            )
            return synergy
        model.obj = Objective(rule=synergy_expr, sense=maximize)

        return model

    def solve(self, solver_name, B=None, L=None, C=None):
        solver = SolverFactory(solver_name)
        self.fix_BLC(B, L, C)
        results = solver.solve(self.model, tee=True)

        print("Status:", results.solver.status)
        print("Termination:", results.solver.termination_condition)

        if results.solver.status != SolverStatus.ok:
            print("Failed to solve the problem")
            return

        # Extract solution
        chosen_bait   = [p for p in self.model.P if self.model.xB[p].value == 1][0]
        chosen_leader = [p for p in self.model.P if self.model.xL[p].value == 1][0]
        chosen_cover  = [p for p in self.model.P if self.model.xC[p].value == 1][0]

        print("Solution:")
        print("  Bait  =", chosen_bait)
        print("  Leader=", chosen_leader)
        print("  Cover =", chosen_cover)

        explain_choice(chosen_bait, chosen_leader, chosen_cover, self.OP, self.DV, self.OI, self.DR)

    def fix_BLC(self, B, L, C):
        if B is not None:
            if not self.support_dual_types:
                print(f"fixing xB as {B}")
                self.model.xB[B].fix(1)
            else:
                B_dual = self.parse_pokemon_types([B])[0]
                print(f"fixing xB as {B_dual}")
                self.model.xB[B_dual[0], B_dual[1]].fix(1)
        if L is not None:
            if not self.support_dual_types:
                print(f"fixing xL as {L}")
                self.model.xL[L].fix(1)
            else:
                L_dual = self.parse_pokemon_types([L])[0]
                print(f"fixing xL as {L_dual}")
                self.model.xL[L_dual[0], L_dual[1]].fix(1)
        if C is not None:
            if not self.support_dual_types:
                print(f"fixing xC as {C}")
                self.model.xC[C].fix(1)
            else:
                C_dual = self.parse_pokemon_types([C])[0]
                print(f"fixing xC as {C_dual}")
                self.model.xC[C_dual[0], C_dual[1]].fix(1)

    def solve_by_loops(self, B=None, L=None, C=None):
        from heapq import heappush, heappop
        class BattleTeam:
            def __init__(self, B, L, C, obj):
                self.B = B
                self.L = L
                self.C = C
                self.obj = obj
            def __lt__(self, other):
                return -self.obj < -other.obj

        b_list = [self.parse_pokemon_types([B])[0]] if B is not None else self.model.P
        l_list = [self.parse_pokemon_types([L])[0]] if L is not None else self.model.P
        c_list = [self.parse_pokemon_types([C])[0]] if C is not None else self.model.P
        heap = []
        for b in b_list:
            for l in l_list:
                if l == b:
                    continue
                for c in c_list:
                    if c == b or c == l:
                        continue
                    # no shared weakness between l and c
                    if len(self.DV[l].intersection(self.DV[c])) > 0 or len(self.DV[b].intersection(self.DV[c])) > 0:
                        continue
                    # calculate alpha, beta, penalty
                    alpha = self.synergy_alpha(b, l)
                    beta = self.synergy_beta(l, c)
                    gamma = self.synergy_gamma(b, c)
                    # objective value
                    obj = alpha + beta + gamma
                    heappush(heap, BattleTeam(b, l, c, obj))
        # finally, retrieve the 1st item
        best_obj = 0
        best_teams = []
        team = heappop(heap)
        while best_obj <= team.obj:
            best_obj = team.obj
            best_teams.append(team)
            team = heappop(heap)
        for team in best_teams:
            print("============")
            print(f"BEST TEAM[obj={team.obj}]: Bait={team.B}, Leader={team.L}, Cover={team.C}")
            print("============")
            explain_choice(team.B, team.L, team.C, self.OP, self.DV, self.OI, self.DR)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A program to show the BLC team selection for pokemon battle league")
    parser.add_argument("pokemon_types", nargs="+", help="Types of pokemons. At least 3 types required.")
    parser.add_argument("--solver", type=str, default="cbc", help="The solver name")
    parser.add_argument("--use_dual", action='store_true', help="If set, both of single and dual types are considered")
    parser.add_argument("--B", type=str, help="(Optional) If specified a particular pokemon type as B, B is fixed")
    parser.add_argument("--L", type=str, help="(Optional) If specified a particular pokemon type as L, L is fixed")
    parser.add_argument("--C", type=str, help="(Optional) If specified a particular pokemon type as C, C is fixed")
    args = parser.parse_args()

    print(f"chosen pokemon types: {args.pokemon_types}")
    if len(args.pokemon_types) < 3:
        print("At least 3 pokemon types are required")
        exit(1)

    # 1) build all sets
    # 
    # OffensePower (OP): the set of types r hits super-effectively.
    # DefenseResilience (DR): the set of types that do not do much damage to r.
    # DV[B] ∩ OP[L]: “r to handle the types that threaten q on offense,”
    # 
    # OffenseIneffectiveness (OI): the set that threatens q.
    # DefenseVulnerability (DV): the set discovered by scanning MATCHUP_MATRIX[t][q] == 'S'
    # OI[B] ∩ DR[L]: “r resists the types q can’t hurt,”
    OP, DV, OI, DR = build_all_sets()

    blc = BLCTeamSelection(support_dual_types=args.use_dual)

    blc.build(args.pokemon_types)

    if args.solver == "loops":

        # solve by loops
        blc.solve_by_loops(args.B, args.L, args.C)

    else:

        blc.solve(args.solver, args.B, args.L, args.C)