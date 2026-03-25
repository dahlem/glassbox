#!/usr/bin/env python3
"""Numerical validation of the nonlocal gauge-Hodge operator framework.

Implements the key experiments needed to support the theoretical claims in
"Nonlocal Gauge-Hodge Operators for Attention Geometry."

Experiments:
  1. H_alpha spectrum:  eigenvalues(H) = sigma(M_asym)^2
  2. Hodge convergence: exact vs matrix-free at increasing n
  3. Low-rank compression: spectral decay of M_asym, error vs rank R
  4. Subsampling convergence: Hodge features stabilize as n grows
  5. Perturbation stability: Lipschitz continuity of Hodge features
  6. C_rms vs C_exact: characterize the relationship

Usage:
    uv run benchmarks/bench_theory.py
    uv run benchmarks/bench_theory.py --experiments 1 2 3 --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import torch

from glassbox.hodge import (
    _compute_G_materialized,
    _compute_routing_features_materialized,
    _estimate_curl_materialized,
    _sample_triangles,
    compute_G_matrix_free,
    compute_routing_features,
    estimate_curl_matrix_free,
)
from glassbox.svd import (
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    lanczos,
    matvec_Masym_blocked,
    randomized_svd,
)

EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Helpers (inline exact Hodge from test infrastructure)
# ---------------------------------------------------------------------------


def _make_M(L, D, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    scale = 1.0 / math.sqrt(D)
    A = torch.softmax(Q @ K.T * scale, dim=-1)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    return Q, K, scale, A, M, d_k_inv_sqrt


def _build_B1(n):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)
    B1 = torch.zeros(n, m, dtype=torch.float64)
    for e_idx, (i, j) in enumerate(edges):
        B1[i, e_idx] = -1
        B1[j, e_idx] = +1
    return B1, edges


def _build_B2(n, edges):
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    triangles = list(combinations(range(n), 3))
    m = len(edges)
    t = len(triangles)
    B2 = torch.zeros(m, t, dtype=torch.float64)
    for tri_idx, (i, j, k) in enumerate(triangles):
        B2[edge_to_idx[(i, j)], tri_idx] = +1
        B2[edge_to_idx[(j, k)], tri_idx] = +1
        B2[edge_to_idx[(i, k)], tri_idx] = -1
    return B2


def _hodge_decompose(f, B1, B2):
    rcond = 1e-10
    L0 = B1 @ B1.T
    L0_pinv = torch.linalg.pinv(L0, rcond=rcond)
    phi = L0_pinv @ B1 @ f
    f_grad = B1.T @ phi
    if B2.shape[1] > 0:
        L2 = B2.T @ B2
        L2_pinv = torch.linalg.pinv(L2, rcond=rcond)
        psi = L2_pinv @ B2.T @ f
        f_curl = B2 @ psi
    else:
        f_curl = torch.zeros_like(f)
    return f_grad, f_curl


def _matrix_to_edge_flow(M, edges):
    f = torch.zeros(len(edges), dtype=M.dtype)
    for e_idx, (i, j) in enumerate(edges):
        f[e_idx] = M[i, j] - M[j, i]
    return f


def _exact_hodge_coefficients(M):
    n = M.shape[0]
    M_f64 = M.to(torch.float64)
    B1, edges = _build_B1(n)
    B2 = _build_B2(n, edges)
    f = _matrix_to_edge_flow(M_f64, edges)
    f_grad, f_curl = _hodge_decompose(f, B1, B2)
    M_fro = torch.linalg.norm(M_f64, "fro")
    sqrt2 = math.sqrt(2.0)
    G = (torch.linalg.norm(f) / (sqrt2 * M_fro)).item()
    C = (torch.linalg.norm(f_curl) / (sqrt2 * M_fro)).item()
    Gamma = (torch.linalg.norm(f_grad) / (sqrt2 * M_fro)).item()
    return G, C, Gamma, f, f_grad, f_curl


# ---------------------------------------------------------------------------
# Irreversibility operator H_alpha = M_asym^T @ M_asym = -M_asym^2
# ---------------------------------------------------------------------------


def matvec_H_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size=256):
    """H_alpha @ v = -M_asym @ (M_asym @ v) = M_asym^T @ (M_asym @ v)."""
    w = matvec_Masym_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size)
    return -matvec_Masym_blocked(Q, K, w, d_k_inv_sqrt, scale, block_size)


# ===========================================================================
# Experiment 1: H_alpha Spectrum
# ===========================================================================


def experiment_1_H_spectrum(n_values=None, D=8, n_seeds=5):
    """Verify eigenvalues(H_alpha) = sigma(M_asym)^2."""
    if n_values is None:
        n_values = [8, 12, 16, 20]
    print("\n" + "=" * 70)
    print("Experiment 1: H_alpha Spectrum Validation")
    print("  Theorem: eigenvalues(H) = sigma(M_asym)^2")
    print("=" * 70)

    results = []
    for n in n_values:
        errors = []
        for seed in range(n_seeds):
            Q, K, scale, A, M, d_k_inv_sqrt = _make_M(n, D, seed=seed)

            # Ground truth: eigenvalues of M_asym^T M_asym
            M_asym = (M - M.T) / 2.0
            sigma_asym = torch.linalg.svdvals(M_asym.to(torch.float64))
            eig_ref = sigma_asym[:min(6, len(sigma_asym))].square()

            # Self-adjointness check
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            v = torch.randn(n)
            w = torch.randn(n)
            Hv = matvec_H_blocked(Q, K, v, d_k_mf, scale, block_size=n)
            Hw = matvec_H_blocked(Q, K, w, d_k_mf, scale, block_size=n)
            sa_err = abs(Hv.dot(w).item() - v.dot(Hw).item())

            # Positive semidefiniteness check
            psd = Hv.dot(v).item()

            # Matrix-free eigenvalues via Lanczos on H
            k = min(6, n - 1)
            op = lambda x: matvec_H_blocked(Q, K, x, d_k_mf, scale, block_size=n)
            evals, _ = lanczos(op, n, k, max(2 * k + 2, 20), "cpu")
            evals_sorted, _ = torch.sort(evals, descending=True)
            evals_top = evals_sorted[:len(eig_ref)]

            # Compare
            max_err = torch.max(torch.abs(eig_ref[:len(evals_top)] - evals_top.to(torch.float64))).item()
            errors.append(max_err)

        mean_err = sum(errors) / len(errors)
        row = {"n": n, "mean_max_eig_error": mean_err, "sa_check": sa_err, "psd_check": psd >= -1e-8}
        results.append(row)
        status = "PASS" if mean_err < 0.05 else "FAIL"
        print(f"  n={n:>3}: max_eig_error={mean_err:.6f}  self_adj={sa_err:.2e}  psd={psd >= -1e-8}  [{status}]")

    return {"experiment": "H_spectrum", "results": results}


# ===========================================================================
# Experiment 2: Hodge Convergence (Exact vs Matrix-Free)
# ===========================================================================


def experiment_2_hodge_convergence(n_values=None, D=4, n_seeds=10):
    """Convergence of matrix-free G, C to exact Hodge values at increasing n."""
    if n_values is None:
        n_values = [5, 8, 10, 12, 16, 20, 25, 30]
    print("\n" + "=" * 70)
    print("Experiment 2: Hodge Convergence (Exact vs Matrix-Free)")
    print("  Corollary 5.3: G_n, C_n → exact values")
    print("=" * 70)

    results = []
    for n in n_values:
        G_errs, C_corr, pyth_residuals = [], [], []
        for seed in range(n_seeds):
            Q, K, scale, A, M, d_k_inv_sqrt = _make_M(n, D, seed=seed)

            # Exact Hodge
            G_ex, C_ex, Gamma_ex, _, _, _ = _exact_hodge_coefficients(M)

            # Matrix-free
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            lse = compute_logsumexp_blocked(Q, K, scale)
            f = compute_routing_features(Q, K, d_k_mf, scale, lse, rank=2, min_samples=200)

            G_errs.append(abs(f["G"] - G_ex))
            C_corr.append((f["C"], C_ex))
            pyth_residuals.append(abs(f["G"] ** 2 - f["Gamma"] ** 2 - f["C"] ** 2))

        mean_G_err = sum(G_errs) / len(G_errs)
        mean_pyth = sum(pyth_residuals) / len(pyth_residuals)
        # Rank correlation for C
        c_mf = [x[0] for x in C_corr]
        c_ex = [x[1] for x in C_corr]
        row = {
            "n": n,
            "mean_G_error": mean_G_err,
            "mean_pyth_residual": mean_pyth,
            "C_mf_mean": sum(c_mf) / len(c_mf),
            "C_exact_mean": sum(c_ex) / len(c_ex),
        }
        results.append(row)
        print(
            f"  n={n:>3}: |G_err|={mean_G_err:.6f}  "
            f"C_mf={row['C_mf_mean']:.4f}  C_exact={row['C_exact_mean']:.4f}  "
            f"pyth={mean_pyth:.2e}"
        )

    return {"experiment": "hodge_convergence", "results": results}


# ===========================================================================
# Experiment 3: Low-Rank Compression of M_asym
# ===========================================================================


def experiment_3_low_rank(n_values=None, D=8, max_rank=20, n_seeds=5):
    """Spectral decay of M_asym and compression error vs rank R."""
    if n_values is None:
        n_values = [32, 64, 128]
    print("\n" + "=" * 70)
    print("Experiment 3: Low-Rank Compression (Theorem 6.1)")
    print("  Spectral decay of M_asym and error vs rank R")
    print("=" * 70)

    results = []
    for n in n_values:
        for seed in range(n_seeds):
            Q, K, scale, A, M, d_k_inv_sqrt = _make_M(n, D, seed=seed)
            M_asym = (M - M.T) / 2.0
            sigma = torch.linalg.svdvals(M_asym.to(torch.float64))
            k = min(max_rank, len(sigma))
            sv_list = sigma[:k].tolist()

            # Normalized decay: sigma_i / sigma_1
            if sv_list[0] > 0:
                decay = [s / sv_list[0] for s in sv_list]
            else:
                decay = [0.0] * k

            # Relative energy captured at each rank R
            total_energy = sigma.square().sum().item()
            energy_at_R = []
            cumsum = 0.0
            for i in range(k):
                cumsum += sigma[i].item() ** 2
                energy_at_R.append(cumsum / (total_energy + EPSILON))

            row = {
                "n": n,
                "seed": seed,
                "singular_values": sv_list[:10],  # top 10
                "normalized_decay": decay[:10],
                "energy_captured": energy_at_R[:10],
            }
            results.append(row)

        # Print summary for this n
        avg_decay = [0.0] * min(10, k)
        for r in results[-n_seeds:]:
            for i, d in enumerate(r["normalized_decay"][:10]):
                avg_decay[i] += d / n_seeds
        decay_str = "  ".join(f"R{i + 1}:{avg_decay[i]:.3f}" for i in range(min(5, len(avg_decay))))
        energy_str = ""
        avg_energy = [0.0] * min(10, k)
        for r in results[-n_seeds:]:
            for i, e in enumerate(r["energy_captured"][:10]):
                avg_energy[i] += e / n_seeds
        energy_str = "  ".join(f"R{i + 1}:{avg_energy[i]:.1%}" for i in [0, 1, 3, 9] if i < len(avg_energy))
        print(f"  n={n:>3}: decay: {decay_str}")
        print(f"         energy: {energy_str}")

    return {"experiment": "low_rank", "results": results}


# ===========================================================================
# Experiment 4: Subsampling Convergence (Theorem 5.2)
# ===========================================================================


def experiment_4_subsampling(N=128, D=8, n_sub_values=None, n_subsamples=20, n_seeds=3):
    """Hodge features at subsampled n converge to full-sequence values."""
    if n_sub_values is None:
        n_sub_values = [16, 24, 32, 48, 64, 96, N]
    print("\n" + "=" * 70)
    print(f"Experiment 4: Subsampling Convergence (N={N})")
    print("  Theorem 5.2: features at subsampled n → full-sequence values")
    print("=" * 70)

    results = []
    for seed in range(n_seeds):
        Q_full, K_full, scale, _, _, _ = _make_M(N, D, seed=seed)

        # Full-sequence features
        _, d_k_full = compute_dk_blocked(Q_full, K_full, scale)
        lse_full = compute_logsumexp_blocked(Q_full, K_full, scale)
        f_full = compute_routing_features(
            Q_full, K_full, d_k_full, scale, lse_full, rank=4, min_samples=200,
        )

        for n in n_sub_values:
            if n >= N:
                G_vals = [f_full["G"]]
                C_vals = [f_full["C"]]
            else:
                G_vals, C_vals = [], []
                for _ in range(n_subsamples):
                    idx = torch.randperm(N)[:n].sort().values
                    Q_sub = Q_full[idx]
                    K_sub = K_full[idx]
                    _, d_k_sub = compute_dk_blocked(Q_sub, K_sub, scale)
                    lse_sub = compute_logsumexp_blocked(Q_sub, K_sub, scale)
                    f_sub = compute_routing_features(
                        Q_sub, K_sub, d_k_sub, scale, lse_sub, rank=2, min_samples=200,
                    )
                    G_vals.append(f_sub["G"])
                    C_vals.append(f_sub["C"])

            import statistics as st

            row = {
                "N": N,
                "n": n,
                "seed": seed,
                "G_mean": st.mean(G_vals),
                "G_std": st.stdev(G_vals) if len(G_vals) > 1 else 0.0,
                "C_mean": st.mean(C_vals),
                "C_std": st.stdev(C_vals) if len(C_vals) > 1 else 0.0,
                "G_full": f_full["G"],
                "C_full": f_full["C"],
            }
            results.append(row)

    # Print summary (averaged over seeds)
    for n in n_sub_values:
        rows = [r for r in results if r["n"] == n]
        G_err = sum(abs(r["G_mean"] - r["G_full"]) for r in rows) / len(rows)
        G_std = sum(r["G_std"] for r in rows) / len(rows)
        C_err = sum(abs(r["C_mean"] - r["C_full"]) for r in rows) / len(rows)
        C_std = sum(r["C_std"] for r in rows) / len(rows)
        print(f"  n={n:>4}: G_err={G_err:.4f} (std={G_std:.4f})  C_err={C_err:.4f} (std={C_std:.4f})")

    return {"experiment": "subsampling", "results": results}


# ===========================================================================
# Experiment 5: Perturbation Stability
# ===========================================================================


def experiment_5_perturbation(n=32, D=8, n_seeds=3):
    """Lipschitz continuity of Hodge features under Q perturbation."""
    epsilons = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0]
    print("\n" + "=" * 70)
    print("Experiment 5: Perturbation Stability")
    print("  Lipschitz continuity of G, C, Gamma under Q perturbation")
    print("=" * 70)

    results = []
    for seed in range(n_seeds):
        Q, K, scale, A, M, d_k_inv_sqrt = _make_M(n, D, seed=seed)
        torch.manual_seed(seed + 1000)
        dQ = torch.randn_like(Q)
        dQ = dQ / torch.linalg.norm(dQ, "fro")  # unit perturbation

        # Baseline features
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        lse = compute_logsumexp_blocked(Q, K, scale)
        f0 = compute_routing_features(Q, K, d_k_mf, scale, lse, rank=2, min_samples=200)

        for eps in epsilons:
            Q_pert = Q + eps * dQ
            _, d_k_p = compute_dk_blocked(Q_pert, K, scale)
            lse_p = compute_logsumexp_blocked(Q_pert, K, scale)
            f_p = compute_routing_features(Q_pert, K, d_k_p, scale, lse_p, rank=2, min_samples=200)

            dG = abs(f_p["G"] - f0["G"])
            dC = abs(f_p["C"] - f0["C"])
            dGamma = abs(f_p["Gamma"] - f0["Gamma"])
            pyth = abs(f_p["G"] ** 2 - f_p["Gamma"] ** 2 - f_p["C"] ** 2)

            row = {
                "seed": seed,
                "epsilon": eps,
                "dG": dG,
                "dC": dC,
                "dGamma": dGamma,
                "Lipschitz_G": dG / eps if eps > 0 else 0,
                "pyth_residual": pyth,
            }
            results.append(row)

    # Print summary
    for eps in epsilons:
        rows = [r for r in results if r["epsilon"] == eps]
        dG = sum(r["dG"] for r in rows) / len(rows)
        lip = sum(r["Lipschitz_G"] for r in rows) / len(rows)
        pyth = sum(r["pyth_residual"] for r in rows) / len(rows)
        print(f"  eps={eps:.0e}: dG={dG:.6f}  Lip(G)={lip:.4f}  pyth={pyth:.2e}")

    return {"experiment": "perturbation", "results": results}


# ===========================================================================
# Experiment 6: C_rms vs C_exact Relationship
# ===========================================================================


def experiment_6_curl_relationship(n_values=None, D=4, n_seeds=20):
    """Characterize the relationship between C_rms and C_exact."""
    if n_values is None:
        n_values = [5, 6, 7, 8, 10, 12, 16, 20]
    print("\n" + "=" * 70)
    print("Experiment 6: C_rms vs C_exact Relationship")
    print("  Key for paper: quantify the estimator-to-exact mapping")
    print("=" * 70)

    results = []
    for n in n_values:
        pairs = []
        for seed in range(n_seeds):
            Q, K, scale, A, M, d_k_inv_sqrt = _make_M(n, D, seed=seed)
            G_ex, C_ex, Gamma_ex, _, _, _ = _exact_hodge_coefficients(M)
            C_rms = _estimate_curl_materialized(M, target_cv=0.01, seed=42)
            pairs.append((C_rms, C_ex, G_ex))

        c_rms_vals = [p[0] for p in pairs]
        c_ex_vals = [p[1] for p in pairs]

        # Ratio C_rms / C_exact
        ratios = [r / (e + EPSILON) for r, e in zip(c_rms_vals, c_ex_vals)]
        mean_ratio = sum(ratios) / len(ratios)

        # Rank correlation
        def _rank_corr(x, y):
            n_pts = len(x)
            if n_pts < 3:
                return float("nan")
            rx = torch.argsort(torch.argsort(torch.tensor(x, dtype=torch.float64))).float()
            ry = torch.argsort(torch.argsort(torch.tensor(y, dtype=torch.float64))).float()
            return torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()

        rank_corr = _rank_corr(c_rms_vals, c_ex_vals)

        row = {
            "n": n,
            "mean_C_rms": sum(c_rms_vals) / len(c_rms_vals),
            "mean_C_exact": sum(c_ex_vals) / len(c_ex_vals),
            "mean_ratio": mean_ratio,
            "rank_correlation": rank_corr,
        }
        results.append(row)
        print(
            f"  n={n:>3}: C_rms={row['mean_C_rms']:.4f}  "
            f"C_exact={row['mean_C_exact']:.4f}  "
            f"ratio={mean_ratio:.3f}  "
            f"rank_corr={rank_corr:.3f}"
        )

    return {"experiment": "curl_relationship", "results": results}


# ===========================================================================
# Main
# ===========================================================================


ALL_EXPERIMENTS = {
    1: ("H_alpha Spectrum", experiment_1_H_spectrum),
    2: ("Hodge Convergence", experiment_2_hodge_convergence),
    3: ("Low-Rank Compression", experiment_3_low_rank),
    4: ("Subsampling Convergence", experiment_4_subsampling),
    5: ("Perturbation Stability", experiment_5_perturbation),
    6: ("C_rms vs C_exact", experiment_6_curl_relationship),
}


def main():
    parser = argparse.ArgumentParser(description="Numerical validation of gauge-Hodge operator framework")
    parser.add_argument(
        "--experiments",
        type=int,
        nargs="+",
        default=list(ALL_EXPERIMENTS.keys()),
        help=f"Experiments to run (default: all). Available: {list(ALL_EXPERIMENTS.keys())}",
    )
    parser.add_argument("--json", type=str, default=None, help="Write JSON results to file")
    args = parser.parse_args()

    print("Nonlocal Gauge-Hodge Operator Framework — Numerical Validation")
    print("=" * 70)

    all_results = []
    for exp_id in args.experiments:
        if exp_id not in ALL_EXPERIMENTS:
            print(f"Unknown experiment {exp_id}, skipping")
            continue
        name, fn = ALL_EXPERIMENTS[exp_id]
        result = fn()
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Experiments run: {len(all_results)}")
    for r in all_results:
        print(f"  - {r['experiment']}: {len(r['results'])} data points")

    if args.json:
        Path(args.json).write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
