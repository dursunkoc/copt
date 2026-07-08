"""
Dantzig-Wolfe master LP (DW-MP) via column generation — COPL without network (a_uv is None).
Matches thesis sec:dw in camp-opt/chapters/03_copl.tex.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from docplex.mp.model import Model

from experiment import Case, Experiment, Parameters, Solution, SolutionResult
from mip_core import MipCore

# Pricing PS(u) problems are independent; batching reduces Python/solver overhead.
DEFAULT_PRICING_BATCH_SIZE = 500

# Shared objective with other experiment.Solution heuristics (objective_fn_no_net).
_objective_holder = Solution("__dw_obj__")


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
    sub_U: List[int],
    mu: np.ndarray,
) -> Tuple[Model, Dict[Tuple[int, int, int, int], Any]]:
    """
    Block-diagonal pricing model for customers in sub_U (same as separate PS(u) summed).
    Constraints: mip_partial_model_constraints_sub per batch; no channel coupling.
    """
    ids = "_".join(str(u) for u in sub_U[:3])
    if len(sub_U) > 3:
        ids += f"_.._{sub_U[-1]}"
    mdl = Model(name=f"DW-Pricing-batch[{len(sub_U)}]_u={ids}")
    X = {
        (c, uu, h, d): mdl.binary_var(f"X_c:{c}_u:{uu}_h:{h}_d:{d}")
        for c in range(C)
        for uu in sub_U
        for h in range(H)
        for d in range(D)
    }

    core.mip_partial_model_constraints_sub(mdl, X, PMS, C, sub_U, H, D, I)

    obj = mdl.sum(
        X[(c, u, h, d)] * (float(PMS.rp_c[c]) - float(mu[h, d]))
        for u in sub_U
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


def _pricing_batches(U: int, batch_size: int) -> List[List[int]]:
    """Indices [0..U-1] split into chunks of at most batch_size."""
    bs = max(1, int(batch_size))
    return [list(range(i, min(i + bs, U))) for i in range(0, U, bs)]


def _run_batch_pricing(
    sub_U: List[int],
    mu: np.ndarray,
    PMS: Parameters,
    C: int,
    H: int,
    D: int,
    I: int,
) -> List[Tuple[int, np.ndarray, float]]:
    """Solve pricing for all u in sub_U in one MIP (block-separable)."""
    if not sub_U:
        return []
    core = MipCore()
    mdl, X = _build_pricing_model(core, PMS, C, H, D, I, sub_U, mu)
    psol = mdl.solve(log_output=False)
    rp = PMS.rp_c.astype(float).reshape(C, 1, 1)
    mu3 = mu.astype(float).reshape(1, H, D)
    coeff = rp - mu3
    if psol is None:
        z = np.zeros((C, H, D), dtype=np.int8)
        return [(u, z.copy(), float("-inf")) for u in sub_U]
    out: List[Tuple[int, np.ndarray, float]] = []
    for u in sub_U:
        x_new = _extract_pricing_schedule(X, C, u, H, D, psol)
        v_star = float(np.sum(x_new.astype(float) * coeff))
        out.append((u, x_new, v_star))
    return out


def _run_single_pricing(
    u: int,
    mu: np.ndarray,
    PMS: Parameters,
    C: int,
    H: int,
    D: int,
    I: int,
) -> Tuple[int, np.ndarray, float]:
    """Solve PS(u); thin wrapper around batch solver."""
    return _run_batch_pricing([u], mu, PMS, C, H, D, I)[0]


# Populated in child processes via ProcessPool initializer (avoids pickling PMS once per task).
_dw_worker_ctx: Optional[Tuple[Parameters, int, int, int, int]] = None


def _dw_pricing_process_init(ctx: Tuple[Parameters, int, int, int, int]) -> None:
    global _dw_worker_ctx
    _dw_worker_ctx = ctx


def _dw_pricing_process_worker(
    batch_mu: Tuple[List[int], np.ndarray],
) -> List[Tuple[int, np.ndarray, float]]:
    assert _dw_worker_ctx is not None
    PMS, C, H, D, I = _dw_worker_ctx
    batch, mu = batch_mu
    return _run_batch_pricing(batch, mu, PMS, C, H, D, I)


def _pricing_tasks_thread(
    args: Tuple[List[int], np.ndarray, Parameters, int, int, int, int],
) -> List[Tuple[int, np.ndarray, float]]:
    batch, mu, PMS, C, H, D, I = args
    return _run_batch_pricing(batch, mu, PMS, C, H, D, I)


def _merge_pricing_columns(
    results: List[Tuple[int, np.ndarray, float]],
    nu: np.ndarray,
    eps: float,
    columns_per_u: List[List[DwColumn]],
    rp_c: np.ndarray,
) -> int:
    added_local = 0
    for u, x_new, v_star in results:
        if v_star > nu[u] + eps:
            r_bar, a_hd = _schedule_coeffs(x_new, rp_c)
            col = DwColumn(x=x_new, r_bar=r_bar, a_hd=a_hd)
            if _add_column_if_new(columns_per_u, u, col):
                added_local += 1
    return added_local


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

    primal_val = float(_objective_holder.objective_fn_no_net(rp, X_cuhd))
    return X_cuhd, primal_val


class DwLpSolution(Solution, MipCore):
    """Column generation for DW-MP; SolutionResult.value = Z^DW."""

    def __init__(
        self,
        eps: float = 1e-5,
        max_iters: int = 5000,
        *,
        parallel_pricing: bool = True,
        max_workers: Optional[int] = None,
        parallel_backend: Literal["process", "thread"] = "process",
        pricing_batch_size: int = DEFAULT_PRICING_BATCH_SIZE,
    ):
        super().__init__("DW-LP")
        self.eps = eps
        self.max_iters = max_iters
        self.parallel_pricing = parallel_pricing
        self.max_workers = max_workers
        self.parallel_backend = parallel_backend
        self.pricing_batch_size = max(1, int(pricing_batch_size))
        self.last_columns_per_u: Optional[List[List[DwColumn]]] = None
        self.last_lam_matrix: Optional[np.ndarray] = None

    def _effective_workers(self, num_batches: int) -> int:
        if not self.parallel_pricing:
            return 1
        if self.max_workers is not None:
            w = self.max_workers
        else:
            w = os.cpu_count() or 4
        return max(1, min(int(w), int(num_batches)))

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

        batches = _pricing_batches(U, self.pricing_batch_size)
        max_w = self._effective_workers(len(batches))
        ctx = (PMS, C, H, D, I)

        it = 0
        z_dw: Optional[float] = None
        last_lam_matrix: Optional[np.ndarray] = None
        cg_rmp_err: Optional[str] = None

        def column_generation_loop() -> None:
            nonlocal it, z_dw, last_lam_matrix, cg_rmp_err
            while it < self.max_iters:
                it += 1
                z, mu, nu, lam_vars, err = _solve_rmp(PMS, U, H, D, columns_per_u)
                if err or z is None or lam_vars is None:
                    cg_rmp_err = err or "unknown"
                    return

                z_dw = z
                max_cols = max(len(columns_per_u[u]) for u in range(U))
                last_lam_matrix = np.zeros((U, max_cols))
                for u in range(U):
                    for j in range(len(columns_per_u[u])):
                        last_lam_matrix[u, j] = float(lam_vars[u, j].solution_value)

                if max_w <= 1:
                    results: List[Tuple[int, np.ndarray, float]] = []
                    for b in batches:
                        results.extend(
                            _run_batch_pricing(b, mu, PMS, C, H, D, I)
                        )
                elif self.parallel_backend == "thread":
                    tasks = [(b, mu, PMS, C, H, D, I) for b in batches]
                    nested = list(ex_threads.map(_pricing_tasks_thread, tasks))
                    results = [row for sub in nested for row in sub]
                else:
                    batch_mu_pairs = [(b, mu) for b in batches]
                    nested = list(
                        ex_process.map(_dw_pricing_process_worker, batch_mu_pairs)
                    )
                    results = [row for sub in nested for row in sub]

                added = _merge_pricing_columns(
                    results, nu, self.eps, columns_per_u, PMS.rp_c
                )
                if added == 0:
                    break

        if max_w <= 1:
            column_generation_loop()
        elif self.parallel_backend == "thread":
            with ThreadPoolExecutor(max_workers=max_w) as ex_threads:
                column_generation_loop()
        else:
            with ProcessPoolExecutor(
                max_workers=max_w,
                initializer=_dw_pricing_process_init,
                initargs=(ctx,),
            ) as ex_process:
                column_generation_loop()

        if z_dw is None:
            end_time = time()
            info = f"RMP_fail:{cg_rmp_err or 'unknown'}"
            return (
                None,
                SolutionResult(case, 0.0, round(end_time - start_time, 4), info),
            )

        end_time = time()
        duration = round(end_time - start_time, 4)
        total_cols = sum(len(columns_per_u[u]) for u in range(U))
        info_parts = [
            f"iters={it}",
            f"total_columns={total_cols}",
            f"eps={self.eps}",
            f"pricing_batch_size={self.pricing_batch_size}",
            f"pricing_batches={len(batches)}",
            f"pricing_workers={max_w}",
            f"pricing_backend={self.parallel_backend if max_w > 1 else 'serial'}",
        ]
        info_parts.append(f"lam_shape={last_lam_matrix.shape}")
        info = ";".join(info_parts)

        val_out = float(z_dw)
        self.last_columns_per_u = columns_per_u
        self.last_lam_matrix = last_lam_matrix
        sr = SolutionResult(case, val_out, duration, info)
        with open(
            f'result_dw_{datetime.now().strftime("%d-%m-%Y %H_%M_%S")}.txt',
            "w",
            encoding="utf-8",
        ) as f:
            f.write(repr(sr))
        return (None, sr)


if __name__ == "__main__":
    from cases import cases

    expr = Experiment(cases)
    solutions = expr.run_cases_with(
        DwLpSolution(eps=1e-5, max_iters=5000, parallel_backend="process"),
        ph=False,
    )
    for solution in solutions:
        print(solution)
