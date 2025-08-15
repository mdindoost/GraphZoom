# coarsen_two_triangles_demo.py
# ---------------------------------------------------------------
# Two triangles with a bridge (2--3). We:
#  1) Build fine graph and its Laplacians.
#  2) Coarsen to 2 supernodes (left triangle vs right triangle).
#  3) Show two coarse constructions:
#       (A) Naive aggregation (sum): L_c = P^T L P; L_sym_c from A_c
#       (B) Fitted normalized operator (Petrov–Galerkin on S = D^{-1/2} A D^{-1/2}):
#             S_c = (P^T P)^{-1} P^T S P, then L_sym_c = I - S_c
#  4) Prolong coarse Fiedler vector back to fine and compare with fine Fiedler vector.
# ---------------------------------------------------------------

import numpy as np
import networkx as nx

# ---------- utilities ----------
def eigh_sorted(M):
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]

def normalized_laplacian(A):
    d = A.sum(axis=1)
    D = np.diag(d)
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(d)
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    D_is = np.diag(inv_sqrt)
    L_sym = np.eye(A.shape[0]) - D_is @ A @ D_is
    return L_sym, D, D_is

def normalized_adjacency(A):
    """S = D^{-1/2} A D^{-1/2}"""
    d = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(d)
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    D_is = np.diag(inv_sqrt)
    return D_is @ A @ D_is

def restrict_signal(x_fine, P):
    """Average within clusters: x_coarse = (P^T P)^{-1} P^T x"""
    PtP_inv = np.linalg.inv(P.T @ P)
    return PtP_inv @ (P.T @ x_fine)

def prolong_signal(x_coarse, P):
    """Piecewise-constant unpool: x_fine = P x_coarse"""
    return P @ x_coarse

def cos_abs(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(abs(a.dot(b)))

# ---------- 1) build fine graph ----------
# Left triangle: 0-1-2-0
# Right triangle: 3-4-5-3
# Bridge: (2,3)
edges = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(2,3)]
n = 6
A = np.zeros((n, n))
for u, v in edges:
    A[u, v] = 1.0
    A[v, u] = 1.0

print("="*80)
print("Fine graph: two triangles with a bridge (2--3)")
print("="*80)
print("Adjacency A:\n", A)

L_sym, D, D_is = normalized_laplacian(A)
L = D - A

print("\nUnnormalized Laplacian L:\n", L)
print("\nSymmetric normalized Laplacian L_sym:\n", np.round(L_sym, 4))

# Fine eigenpairs
evals_L, evecs_L = eigh_sorted(L)
evals_Lsym, evecs_Lsym = eigh_sorted(L_sym)
print("\nEigenvalues(L):", np.round(evals_L, 6))
print("Eigenvalues(L_sym):", np.round(evals_Lsym, 6))
print("Fiedler value of L (λ2):", float(evals_L[1]))
print("Fiedler value of L_sym (λ2):", float(evals_Lsym[1]))
fiedler_fine = evecs_Lsym[:, 1]  # second smallest of L_sym
print("Fiedler vector (fine, L_sym) ≈\n", np.round(fiedler_fine, 6))

# ---------- 2) coarsen to two supernodes ----------
# Cluster 0: {0,1,2}; Cluster 1: {3,4,5}
P = np.zeros((n, 2))
P[0,0]=1; P[1,0]=1; P[2,0]=1
P[3,1]=1; P[4,1]=1; P[5,1]=1

print("\nAssignment P (fine→coarse):\n", P)

# ---------- 3A) Naive coarse graph (sum aggregation) ----------
# Coarse adjacency and Laplacians by direct sums
A_c = P.T @ A @ P
D_c = np.diag(A_c.sum(axis=1))
with np.errstate(divide='ignore'):
    inv_sqrt_c = 1.0 / np.sqrt(np.diag(D_c))
inv_sqrt_c[np.isinf(inv_sqrt_c)] = 0.0
D_is_c = np.diag(inv_sqrt_c)
L_c = P.T @ (D - A) @ P                # = P^T L P
L_sym_c_direct = np.eye(2) - D_is_c @ A_c @ D_is_c

print("\n[Naive coarse] A_c = P^T A P:\n", A_c)
print("[Naive coarse] L_c = P^T L P:\n", L_c)
print("[Naive coarse] L_sym_c (direct):\n", np.round(L_sym_c_direct, 6))

evals_Lc, evecs_Lc = eigh_sorted(L_c)
evals_Lsym_c_direct, evecs_Lsym_c_direct = eigh_sorted(L_sym_c_direct)
print("Eigenvalues(L_c):", np.round(evals_Lc, 6))
print("Eigenvalues(L_sym_c_direct):", np.round(evals_Lsym_c_direct, 6))

# ---------- 3B) Fitted coarse operator on normalized adjacency ----------
# We fit S_c that best matches S in least squares: S_c = (P^T P)^{-1} P^T S P
S = normalized_adjacency(A)                # fine normalized adjacency
PtP_inv = np.linalg.inv(P.T @ P)
S_c = PtP_inv @ (P.T @ S @ P)              # Petrov–Galerkin fit
L_sym_c_fit = np.eye(2) - S_c

print("\n[Fitted coarse] S_c = (P^T P)^-1 P^T S P:\n", np.round(S_c, 6))
print("[Fitted coarse] L_sym_c_fit = I - S_c:\n", np.round(L_sym_c_fit, 6))

evals_Lsym_c_fit, evecs_Lsym_c_fit = eigh_sorted(L_sym_c_fit)
print("Eigenvalues(L_sym_c_fit):", np.round(evals_Lsym_c_fit, 6))

# ---------- 4) Prolong coarse Fiedler vector & compare ----------
# Use the FITTED coarse operator (this preserves message passing better).
fiedler_coarse = evecs_Lsym_c_fit[:, 1]  # second smallest
h_prolong = prolong_signal(fiedler_coarse, P)

print("\nCoarse Fiedler (L_sym_c_fit):", np.round(fiedler_coarse, 6))
print("Prolonged to fine: P @ fiedler_coarse ≈\n", np.round(h_prolong, 6))

similarity = cos_abs(h_prolong, fiedler_fine)
print("\nCosine similarity |< P·v_c , v_fine >| =", similarity)
print("(~1.0 means the coarse Fiedler, unpooled, matches the fine Fiedler shape.)")

# ---------- 5) What about 'reconstructing' the fine graph? ----------
print("\nNOTE:")
print("  From only the 2-node coarse graph, you cannot uniquely reconstruct all fine edges.")
print("  What you CAN do is 'prolongate' signals: y_fine = P y_coarse (piecewise constant).")
print("  To recover edges/weights, you need extra info (the original A or a refinement model).")
print("  In multilevel GNNs, we typically:")
print("    - train on the coarse graph,")
print("    - then unpool activations with P (and optionally refine with local diffusion).")
