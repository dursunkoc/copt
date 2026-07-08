# Pareto Instance 8 — teze eklenecek içerik (uygulama notu)

**Durum:** Ortam şu an *plan modunda* olduğu için `05_copln.tex` / `main.tex` doğrudan düzenlenemedi. Aşağıdaki metni elle yapıştırın veya **Agent modunda** “uygula” deyin.

**Yapıldı (terminal):** `pareto_curve_2000.png` dosyası `copt/pareto_curve_2000.png` konumundan `copt/camp-opt/pareto_curve_2000.png` konumuna kopyalandı.

---

## 1) `camp-opt/main.tex`

`epsfig` paketinin yanına `graphicx` ekleyin:

```latex
\usepackage{amsmath,amssymb,latexsym,float,epsfig,graphicx,subfigure}
```

---

## 2) `camp-opt/chapters/05_copln.tex`

`\subsection{Impact of Incorporating Network Effects in Campaign Planning}\label{s:test_evaluation_network}` satırının **hemen üstüne** aşağıdaki bloğu ekleyin.

```latex
\subsection{Supplementary trade-off analysis via an $\epsilon$-constraint on direct messaging (Instance~8)}\label{s:epsilon-tradeoff-inst8}

The preceding discussion evaluates solution methods and network utilization using fixed communication rules encoded in the COP-LN feasibility system. From a managerial perspective, practitioners may additionally wish to quantify how tightening an aggregate cap on \emph{direct} outbound contacts affects the network-aware objective in~\eqref{mathmodel_obj_neteff}, holding the remaining COP-LN structure unchanged. Such a cap can be interpreted as a stylized representation of fatigue governance that limits total direct touches across the planning horizon, beyond the customer-local frequency limits already enforced in~\eqref{mathmodel_eligibility}--\eqref{mathmodel_integrity}.

To provide a compact illustration, we report a supplementary $\epsilon$-constraint experiment on Instance~8 in Table~\ref{table:tbl_test_instances-net}, i.e., $|C|=10$, $|U|=2000$, and $|E|=56$ in the sales-enriched network. We first solve the baseline COP-LN without an explicit aggregate cap on $\sum_{c,h,d,u} X_{cuhd}$. We then re-solve the same model subject to
\begin{equation}
\sum_{c\in C}\sum_{u\in U}\sum_{h\in H}\sum_{d\in D} X_{cuhd} \;\le\; \epsilon,
\label{eq:pareto_eps_cap}
\end{equation}
for a decreasing sequence of integer caps $\epsilon$ obtained by successive reductions relative to the baseline direct-message volume (an $\epsilon$-constraint parametrization in the sense of classical multi-objective scalarization).

Table~\ref{tab:pareto_inst8_eps} summarizes the realized direct-message totals and objective values recorded for this Instance~8 run. Relative to the baseline incumbent $(360,\,92039)$, tightening the cap to $\epsilon=324$ is associated with a modest deterioration of the objective to $90415$ (approximately $1.8\%$ below the baseline), whereas a more aggressive cap $\epsilon=288$ yields $74649$ (approximately $18.9\%$ below the baseline). These patterns are consistent with diminishing returns to additional direct messaging when indirect awareness is already shaped by~\eqref{mathmodel_w/n_netcoverage}: once the cap forces a substantial reduction in direct activations, the objective declines more rapidly because both direct coverage and neighbor-mediated coverage become harder to sustain simultaneously.

\begin{table}[t]
\centering
\caption{Illustrative $\epsilon$-constraint sweep for Instance~8: realized direct message volume and COP-LN objective value~\eqref{mathmodel_obj_neteff}. Values are solver outputs under the stated time limit; except for the trivial $\epsilon=0$ case, incumbent solutions may not be proven optimal.}
\label{tab:pareto_inst8_eps}
\footnotesize
\begin{tabular}{@{}lrr@{}}
\toprule
Setting ($\epsilon$ cap) & Direct messages & Objective \\
\midrule
Baseline (no aggregate cap) & 360 & 92039 \\
324 & 324 & 90415 \\
288 & 288 & 74649 \\
251 & 251 & 72300 \\
216 & 216 & 70885 \\
180 & 180 & 58345 \\
143 & 143 & 49770 \\
107 & 107 & 34533 \\
71 & 71 & 30906 \\
35 & 35 & 15102 \\
0 & 0 & 0 \\
\bottomrule
\end{tabular}
\end{table}

\begin{remark}[Interpretation under solver time limits]
For the reported Instance~8 sweep, CPLEX terminated at the prescribed time limit for all nontrivial caps in Table~\ref{tab:pareto_inst8_eps}, returning a feasible incumbent solution. The $\epsilon=0$ restriction yields the trivial optimal value $0$. Consequently, the plotted profile should be interpreted as an empirical \emph{incumbent} trade-off curve rather than a provably Pareto-optimal frontier of the binary COP-LN.
\end{remark}

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{pareto_curve_2000.png}
\caption{Empirical trade-off between realized direct messaging volume and the COP-LN objective value~\eqref{mathmodel_obj_neteff} for Instance~8 under the aggregate cap~\eqref{eq:pareto_eps_cap}. Points correspond to incumbent solutions obtained within the solver time limit (Remark after Table~\ref{tab:pareto_inst8_eps}).}
\label{fig:pareto_inst8_eps}
\end{figure}
```

---

## 3) `camp-opt/chapters/06_conclusion.tex` (öneri)

`\section{Managerial Insights}` içinde, uygun bir yere (ör. `\subsection{Communication governance...}` sonrası) kısa köprü paragrafı:

```latex
\subsection{Aggregate caps on direct messaging}
Chapter~\ref{ch:copln} includes a supplementary $\epsilon$-constraint analysis on Instance~8 that traces how an aggregate upper bound on direct assignments interacts with the network-aware objective~\eqref{mathmodel_obj_neteff} (\S\ref{s:epsilon-tradeoff-inst8}). The reported profile should be read as an incumbent trade-off curve under a fixed solver time limit rather than a provably Pareto-optimal frontier; nevertheless, it provides a disciplined way to communicate how aggressive fatigue-style caps can erode achievable influence when neighbor-mediated awareness is endogenous to the plan.
```

---

## Derleme

`pdflatex` ile `pareto_curve_2000.png` dosyasının `main.tex` ile aynı dizinde (`camp-opt/`) olduğundan emin olun.
