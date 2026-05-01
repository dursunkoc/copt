"""
Dantzig-Wolfe master LP (DW-MP) via column generation — COPL without network (a_uv is None).
Matches thesis sec:dw in camp-opt/chapters/03_copl.tex.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from docplex.mp.model import Model

from experiment import Case, Parameters, Solution, SolutionResult
from mip_core import MipCore


def _column_bitkey(x: np.ndarray) -> Tuple[int, ...]:
    """Flatten (C,H,D) binary schedule for duplicate detection."""
    return tuple(np.asarray(x, dtype=np.int8).flatten().tolist())


def _schedule_coeffs(
    x: np.ndarray, rp_c: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Per-tez r_bar and channel-day loads a_hd from schedule x shape (C,H,D)."""
    xc = np.asarray(x, dtype=float)
    r_bar = float(np.sum(xc * rp_c[:, np.newaxis, np.newaxis]))
    a_hd = np.sum(xc, axis=0)
    return r_bar, a_hd


def _uses_rh(PMS: Parameters) -> bool:
    """Same branch as mip_core.start_sub_model for rolling-horizon constraints."""
    return PMS.s_cuhd is not None


@dataclass
class DwColumn:
    x: np.ndarray  # (C, H, D) binary
    r_bar: float
    a_hd: np.ndarray  # (H, D)


def _empty_schedule(C: int, H: int, D: int) -> np.ndarray:
    return np.zeros((C, H, D), dtype=np.int8)


def _add_column_if_new(columns_per_u: List[List[DwColumn]], u: int, col: DwColumn) -> bool:
    key = _column_bitkey(col.x)
    for existing in columns_per_u[u]:
        if _column_bitkey(existing.x) == key:
            return False
    columns_per_u[u].append(col)
    return True


def _build_pricing_model(
    core: MipCore,
    PMS: Parameters,
    C: int,
    H: int,
    D: int,
    I: int,
    u: int,
    mu: np.ndarray,
) -> Tuple[Model, Dict[Tuple[int, int, int, int], Any]]:
    """PS(u): max sum (rp_c[c] - mu[h,d]) x[c,u,h,d] subject to (2u)-(11u) + single-channel."""
    mdl = Model(name=f"DW-Pricing-u={u}")
    sub_u = [u]
    X = {
        (c, uu, h, d): mdl.binary_var(f"x_{c}_{uu}_{h}_{d}")
        for c in range(C)
        for uu in sub_u
        for h in range(H)
        for d in range(D)
    }

    core.mip_eligibility_sub(mdl, X, PMS, C, sub_u, H, D)

    # Thesis (3u): at most one channel per campaign per day
    for c in range(C):
        for d in range(D):
            mdl.add_constraint(
                mdl.sum(X[(c, u, hh, d)] for hh in range(H)) <= 1,
                ctname=f"single_ch_c{c}_d{d}",
            )

    if _uses_rh(PMS):
        for f_d in range(1, D + 1):
            core.mip_weekly_communication_rh_sub(mdl, X, PMS, C, sub_u, H, D, f_d)
            core.mip_campaign_communication_rh_sub(mdl, X, PMS, C, sub_u, H, D, f_d)
            core.mip_weekly_quota_rh_sub(mdl, X, PMS, C, sub_u, H, D, I, f_d)
    else:
        core.mip_weekly_communication_sub(mdl, X, PMS, C, sub_u, H, D)
        core.mip_campaign_communication_sub(mdl, X, PMS, C, sub_u, H, D)
        core.mip_weekly_quota_sub(mdl, X, PMS, C, sub_u, H, D, I)

    core.mip_daily_communication_sub(mdl, X, PMS, C, sub_u, H, D)
    core.mip_daily_quota_sub(mdl, X, PMS, C, sub_u, H, D, I)

    obj = mdl.sum(
        X[(c, u, h, d)] * (float(PMS.rp_c[c]) - float(mu[h, d]))
        for c in range(C)
        for h in range(H)
        for d in range(D)
    )
    mdl.maximize(obj)
    return mdl, X


def _extract_pricing_schedule(
    X: Dict[Tuple[int, int, int, int], Any], C: int, u: int, H: int, D: int, sol: Any
) -> np.ndarray:
    out = np.zeros((C, H, D), dtype=np.int8)
    if sol is None:
        return out
    for c in range(C):
        for h in range(H):
            for d in range(D):
                v = X[(c, u, h, d)].solution_value
                out[c, h, d] = 1 if v is not None and v > 0.5 else 0
    return out


def _solve_rmp(
    PMS: Parameters,
    U: int,
    H: int,
    D: int,
    columns_per_u: List[List[DwColumn]],
) -> Tuple[
    Optional[float],
    np.ndarray,
    np.ndarray,
    Optional[Dict[Tuple[int, int], Any]],
    Optional[str],
]:
    """
    Build and solve restricted master LP.
    Returns (z, mu, nu, lam_vars, fail_msg). lam_vars maps (u,j) -> docplex variable.
    """
    mdl = Model(name="DW-RMP")
    lam: Dict[Tuple[int, int], Any] = {}
    for u in range(U):
        for j in range(len(columns_per_u[u])):
            lam[u, j] = mdl.continuous_var(lb=0.0, ub=1.0, name=f"lam_{u}_{j}")

    cap_cons: Dict[Tuple[int, int], Any] = {}
    for h in range(H):
        for d in range(D):
            expr = mdl.sum(
                columns_per_u[u][j].a_hd[h, d] * lam[u, j]
                for u in range(U)
                for j in range(len(columns_per_u[u]))
            )
            cap_cons[h, d] = mdl.add_constraint(
                expr <= float(PMS.t_hd[h, d]), ctname=f"cap_h{h}_d{d}"
            )

    conv_cons: Dict[int, Any] = {}
    for u in range(U):
        conv_cons[u] = mdl.add_constraint(
            mdl.sum(lam[u, j] for j in range(len(columns_per_u[u]))) == 1,
            ctname=f"conv_u{u}",
        )

    obj = mdl.sum(
        columns_per_u[u][j].r_bar * lam[u, j]
        for u in range(U)
        for j in range(len(columns_per_u[u]))
    )
    mdl.maximize(obj)
    sol = mdl.solve(log_output=False)
    if sol is None:
        return None, np.zeros((H, D)), np.zeros(U), None, "RMP solve returned None"

    mu = np.zeros((H, D))
    for h in range(H):
        for d in range(D):
            dv = cap_cons[h, d].dual_value
            mu[h, d] = float(dv) if dv is not None else 0.0

    nu = np.zeros(U)
    for u in range(U):
        dv = conv_cons[u].dual_value
        nu[u] = float(dv) if dv is not None else 0.0

    z = float(mdl.objective_value)
    return z, mu, nu, lam, None


def round_dw_primal(
    columns_per_u: List[List[DwColumn]],
    lam_solution: np.ndarray,
    PMS: Parameters,
    C: int,
    U: int,
    H: int,
    D: int,
) -> Tuple[np.ndarray, float]:
    """
    Tez sec:dw-primal: pick argmax_j lambda_uj, repair channel overload by dropping lowest rp_c.
    lam_solution shape (U, max_cols) with zeros for nonexistent columns.
    """
    rp = PMS.rp_c
    X_cuhd = np.zeros((C, U, H, D), dtype=np.int8)
    for u in range(U):
        ncol = len(columns_per_u[u])
        if ncol == 0:
            continue
        j_star = int(np.argmax(lam_solution[u, :ncol]))
        X_cuhd[:, u, :, :] = columns_per_u[u][j_star].x

    def channel_load(hh: int, dd: int) -> int:
        return int(X_cuhd[:, :, hh, dd].sum())

    tol = 1e-9
    for h in range(H):
        for d in range(D):
            cap = float(PMS.t_hd[h, d])
            while channel_load(h, d) > cap + tol:
                candidates: List[Tuple[int, int, int]] = []
                for uu in range(U):
                    for c in range(C):
                        if X_cuhd[c, uu, h, d] == 1:
                            candidates.append((int(rp[c]), c, uu))
                if not candidates:
                    break
                candidates.sort(key=lambda t: t[0])
                _, c_drop, u_drop = candidates[0]
                X_cuhd[c_drop, u_drop, h, d] = 0

    primal_val = float(np.matmul(rp, X_cuhd.sum(axis=(1, 2, 3))))
    return X_cuhd, primal_val


class DwLpSolution(Solution, MipCore):
    """Column generation for DW-MP; SolutionResult.value = Z^DW."""

    def __init__(
        self,
        eps: float = 1e-5,
        max_iters: int = 5000,
    ):
        super().__init__("DW-LP")
        self.eps = eps
        self.max_iters = max_iters
        self.last_columns_per_u: Optional[List[List[DwColumn]]] = None
        self.last_lam_matrix: Optional[np.ndarray] = None

    def runPh(self, case: Case, Xp_cuhd=None) -> Tuple[None, SolutionResult]:
        start_time = time()
        if case.arguments.get("a_uv") is not None:
            raise ValueError(
                "Dantzig-Wolfe formulation in thesis (sec:dw) does not cover network COPL; "
                "omit a_uv / use Parameters with a_uv=None."
            )

        C = case.arguments["C"]
        U = case.arguments["U"]
        H = case.arguments["H"]
        D = case.arguments["D"]
        I = case.arguments["I"]

        PMS: Parameters = super().generate_parameters(case, Xp_cuhd)
        if PMS.a_uv is not None:
            raise ValueError("DwLpSolution requires PMS.a_uv is None (no network effect).")

        columns_per_u: List[List[DwColumn]] = []
        for _u in range(U):
            x0 = _empty_schedule(C, H, D)
            r_bar, a_hd = _schedule_coeffs(x0, PMS.rp_c)
            col = DwColumn(x=x0.copy(), r_bar=r_bar, a_hd=a_hd.copy())
            columns_per_u.append([col])

        it = 0
        z_dw: Optional[float] = None
        last_lam_matrix: Optional[np.ndarray] = None

        while it < self.max_iters:
            it += 1
            z, mu, nu, lam_vars, err = _solve_rmp(PMS, U, H, D, columns_per_u)
            if err or z is None or lam_vars is None:
                end_time = time()
                info = f"RMP_fail:{err}"
                return (
                    None,
                    SolutionResult(case, 0.0, round(end_time - start_time, 4), info),
                )

            z_dw = z
            max_cols = max(len(columns_per_u[u]) for u in range(U))
            last_lam_matrix = np.zeros((U, max_cols))
            for u in range(U):
                for j in range(len(columns_per_u[u])):
                    last_lam_matrix[u, j] = float(lam_vars[u, j].solution_value)

            added = 0
            for u in range(U):
                pmdl, x_vars = _build_pricing_model(self, PMS, C, H, D, I, u, mu)
                psol = pmdl.solve(log_output=False)
                if psol is None:
                    continue
                v_star = float(pmdl.objective_value)
                if v_star > nu[u] + self.eps:
                    x_new = _extract_pricing_schedule(x_vars, C, u, H, D, psol)
                    r_bar, a_hd = _schedule_coeffs(x_new, PMS.rp_c)
                    col = DwColumn(x=x_new, r_bar=r_bar, a_hd=a_hd)
                    if _add_column_if_new(columns_per_u, u, col):
                        added += 1

            if added == 0:
                break

        end_time = time()
        duration = round(end_time - start_time, 4)
        total_cols = sum(len(columns_per_u[u]) for u in range(U))
        info_parts = [
            f"iters={it}",
            f"total_columns={total_cols}",
            f"eps={self.eps}",
        ]
        if last_lam_matrix is not None:
            info_parts.append(f"lam_shape={last_lam_matrix.shape}")
        info = ";".join(info_parts)

        val_out = float(z_dw) if z_dw is not None else 0.0
        self.last_columns_per_u = columns_per_u
        self.last_lam_matrix = last_lam_matrix
        return (None, SolutionResult(case, val_out, duration, info))


if __name__ == "__main__":
    tiny = Case({"C": 2, "U": 4, "H": 2, "D": 2, "I": 2, "P": 2, "id": -1})
    _sol = DwLpSolution(eps=1e-5, max_iters=100)
    _, sr = _sol.runPh(tiny, None)
    print(sr)
    assert sr.value >= 0
    assert "iters=" in (sr.info or "")
