import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def build_path_graph(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return sp.csr_matrix(A)

def compute_propagation(A, method="mean"):
    degrees = np.array(A.sum(axis=1)).flatten()
    if method == "mean":
        D_inv = sp.diags(1.0 / degrees)
        return D_inv @ A
    elif method == "gcn":
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        return D_inv_sqrt @ A @ D_inv_sqrt
    else:
        raise ValueError("Unknown method")

def manual_Q(n, clusters):
    Q = np.zeros((len(clusters), n))
    for i, cluster in enumerate(clusters):
        for node in cluster:
            Q[i, node] = 1.0
    Q = Q / np.sqrt(Q.sum(axis=1, keepdims=True))
    return sp.csr_matrix(Q)

def run_experiment(name, A, S, Q, x, power_iters=1):
    Q_plus = Q.transpose().tocsr()
    x_orig = x.copy()

    # Run power iterations: S^k x
    for _ in range(power_iters):
        x = S @ x
    Sx = x

    x_c = Q @ x_orig
    S_c_mp = Q @ S @ Q_plus
    for _ in range(power_iters):
        x_c = S_c_mp @ x_c
    Sx_mp = Q_plus @ x_c

    x_c = Q @ x_orig
    A_c = Q @ A @ Q_plus
    deg_c = np.array(A_c.sum(axis=1)).flatten()
    D_inv_c = sp.diags(1.0 / deg_c)
    S_c_naive = D_inv_c @ A_c
    for _ in range(power_iters):
        x_c = S_c_naive @ x_c
    Sx_naive = Q_plus @ x_c

    print(f"\n=== {name} with Power Iteration k={power_iters} ===")
    print("Original S^k x:       ", np.round(Sx.flatten(), 3))
    print("MP-aware S_c^k x:     ", np.round(Sx_mp.flatten(), 3))
    print("Naive S_c^k x:        ", np.round(Sx_naive.flatten(), 3))

    err_mp = np.linalg.norm(Sx - Sx_mp)
    err_naive = np.linalg.norm(Sx - Sx_naive)
    print(f"Error (MP-aware):   {err_mp:.6f}")
    print(f"Error (Naive):      {err_naive:.6f}")

def generate_signals(A, n):
    signals = {}
    signals['piecewise'] = np.array([0, 0, 0, 1, 1, 1, 2, 2]).reshape(-1, 1)
    signals['linear'] = np.linspace(0, 1, n).reshape(-1, 1)
    L = sp.csgraph.laplacian(A, normed=True)
    eigvals, eigvecs = eigsh(L, k=2, which='SM')
    signals['spectral'] = eigvecs[:, [1]]  # skip constant eigenvector
    return signals

def main():
    n = 8
    A = build_path_graph(n)
    clusters = [[0, 1, 2], [3, 4, 5], [6, 7]]
    Q = manual_Q(n, clusters)
    signals = generate_signals(A, n)

    for sig_name, x in signals.items():
        print(f"\nInput signal ({sig_name}):", x.flatten())
        for method in ["mean", "gcn"]:
            S = compute_propagation(A, method=method)
            for k in [1, 2, 3]:
                run_experiment(f"{method.upper()} Aggregation [{sig_name}]", A, S, Q, x, power_iters=k)

if __name__ == "__main__":
    main()
