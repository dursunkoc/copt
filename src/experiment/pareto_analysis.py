"""
ϵ-kısıt (toplam doğrudan mesaj üst sınırı) ile Pareto / trade-off analizi.

mip_solution_with_network_applied.MipSolutionWithNetwork ile aynı model kurulumunu
kullanır; tek model örneği üzerinde bütçe kısıtının RHS değerini güncelleyerek
yeniden çözer (modeli her iterasyonda sıfırdan kurmaz).
"""
from __future__ import annotations

from time import time
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from experiment import Case
from mip_solution_with_network_applied import MipSolutionWithNetwork
from network_generator import gen_network


def _total_direct_messages_expr(mdl, X, C: int, U: int, H: int, D: int):
    return mdl.sum(
        X[(c, u, h, d)]
        for c in range(C)
        for u in range(U)
        for h in range(H)
        for d in range(D)
    )


def _build_epsilon_caps(M0: int, step_fraction: float) -> List[int]:
    """M0 referansına göre azalan tamsayı bütçe üst sınırları (yinelemelerde tekrar yok)."""
    if M0 <= 0 or step_fraction <= 0:
        return []
    seen = set()
    caps: List[int] = []
    k = 1
    while True:
        cap = int(np.floor(M0 * (1.0 - k * step_fraction)))
        if cap < 0:
            break
        if cap not in seen:
            seen.add(cap)
            caps.append(cap)
        if cap == 0:
            break
        k += 1
    return caps


def run_pareto_analysis(
    mip_solver: MipSolutionWithNetwork,
    case: Case,
    step_fraction: float = 0.1,
    Xp_cuhd=None,
    csv_path: str = "pareto_results.csv",
    png_path: str = "pareto_curve.png",
) -> pd.DataFrame:
    """
    1) Bütçe kısıtı olmadan çöz; taban doğrudan mesaj sayısı M0 ve amaç Z0.
    2) sum X_cuhd <= ϵ kısıtını (tek kısıt) ekleyip RHS ile ϵ grid'ini tarar.

    Dönüş: her satırda izin verilen üst sınır, gerçekleşen toplam X, amaç, çözüm süresi.
    """
    C = case.arguments["C"]
    U = case.arguments["U"]
    H = case.arguments["H"]
    D = case.arguments["D"]
    I = case.arguments["I"]

    nw_start = time()
    a_uv, _G = gen_network(
        seed=mip_solver.seed,
        p=mip_solver.p,
        n=U,
        m=mip_solver.m,
        drop_prob=mip_solver.drop_prob,
        net_type=mip_solver.net_type,
    )
    nw_duration = time() - nw_start

    PMS = mip_solver.generate_parameters(case, Xp_cuhd, a_uv=a_uv)
    mdl, X = mip_solver.start_model(True, PMS, C, U, H, D, I)

    rows = []
    total_x_expr = _total_direct_messages_expr(mdl, X, C, U, H, D)
    budget_ct = None

    def solve_and_record(allowed_cap: Optional[float], iteration_label: str):
        print(f"[Pareto] cozum basliyor | iterasyon={iteration_label} | eps={allowed_cap}")
        t0 = time()
        result = mdl.solve(log_output=False, time_limit=60)
        elapsed = time() - t0
        if result is not None and hasattr(result, "solve_details"):
            status = getattr(result.solve_details, "status", str(result.solve_details))
        elif result is not None:
            status = str(result.solve_status)
        else:
            status = "no_solution"
        if result is not None and result.objective_value is not None:
            obj = float(result.objective_value)
        else:
            obj = np.nan
        X_arr = mip_solver.create_var_for_greedy(result, C, D, H, U)
        actual_msgs = int(X_arr.sum()) if X_arr is not None else np.nan
        rows.append(
            {
                "iterasyon": iteration_label,
                "izin_verilen_butce_ust_sinir": allowed_cap
                if allowed_cap is not None
                else np.nan,
                "gonderilen_gercek_mesaj_sayisi": actual_msgs,
                "amac_fonksiyonu": obj,
                "cozum_suresi_saniye": round(elapsed, 6),
                "cozum_durumu": str(status),
                "ag_kurulum_saniye": round(nw_duration, 6),
            }
        )
        print(
            "[Pareto] cozum bitti | "
            f"iterasyon={iteration_label} | "
            f"eps={allowed_cap} | "
            f"mesaj={actual_msgs} | "
            f"amac={obj} | "
            f"sure={round(elapsed, 6)}s | "
            f"durum={status}"
        )
        return result

    # Taban: doğrudan mesaj üst sınırı yok
    base_result = solve_and_record(None, "taban_kisitsiz")
    if base_result is None:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        return df

    X_base = mip_solver.create_var_for_greedy(base_result, C, D, H, U)
    M0 = int(X_base.sum())

    epsilon_caps = _build_epsilon_caps(M0, step_fraction)
    for eps in tqdm(epsilon_caps, desc="Pareto eps iterasyonlari", unit="eps"):
        if budget_ct is None:
            budget_ct = mdl.add_constraint(total_x_expr <= eps, ctname="pareto_direct_msg_cap")
        else:
            budget_ct.rhs = eps
        res = solve_and_record(float(eps), f"epsilon_<={eps}")
        if res is None:
            break

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    plot_df = df.dropna(subset=["amac_fonksiyonu", "gonderilen_gercek_mesaj_sayisi"])
    plot_df = plot_df[np.isfinite(plot_df["amac_fonksiyonu"])]
    if plt is not None and len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            plot_df["gonderilen_gercek_mesaj_sayisi"],
            plot_df["amac_fonksiyonu"],
            marker="o",
            linewidth=1.5,
        )
        ax.set_xlabel("Total Direct Messages Sent (Fatigue)")
        ax.set_ylabel("Total Network-Aware Influence (Objective Value)")
        ax.set_title("Pareto Frontier: Influence vs. Direct Messaging")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

    return df


if __name__ == "__main__":
    # Hızlı duman testi — tez üretiminde cases.py’deki gerçek vaka ile değiştirin.
    demo_case = Case({"C": 10, "U": 3000, "H": 1, "D": 7, "I": 2, "P": 3, "id": -1})
    solver = MipSolutionWithNetwork(seed=142, net_type="erdos", m=None, p=0.05, drop_prob=0.0)
    out = run_pareto_analysis(
        solver,
        demo_case,
        step_fraction=0.1,
        csv_path="pareto_results.csv",
        png_path="pareto_curve.png",
    )
    print(out)
