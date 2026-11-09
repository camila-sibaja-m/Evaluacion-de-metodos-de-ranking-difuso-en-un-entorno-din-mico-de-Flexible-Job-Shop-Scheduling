#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esquema dinámico Make-to-Order para el FFJSP difuso con SimPy (Ecs. 6-7)
y robustez Monte Carlo (Ec. 8). Corresponde a las Secciones 4.3.2 y 4.4-4.5.

Diseño:
  1. Llegadas: proceso de Poisson por trabajo con lambda0 = rho * mu_hat * m (Sección 4.5).
     mu_hat = 1 / tiempo medio E_1/2, promediado sobre todas las operaciones y máquinas.
     El patrón de llegadas es fijo dentro de cada (instancia, rho): se comparte entre β.
  2. Monte Carlo: duraciones reales ~ Triangular(a1, a2, a3) con numpy. No se afirma
     equivalencia con la transformación pignística. Las K réplicas se muestrean una vez
     por (instancia, rho) y se reutilizan para los tres β (números aleatorios comunes).
  3. Ruteo (Ec. 6): al quedar lista una operación, se asigna a la máquina elegible con
     menor E_β. Despacho (Ec. 7): al liberarse una máquina, atiende de su cola la operación
     con menor E_β, mediante PriorityResource (prioridad = E_β(p_ijm)).
     La decisión solo usa E_β; la duración real se revela al iniciar la operación.
  4. SimPy real: un proceso por trabajo y un PriorityResource por máquina.
  5. Los archivos Instancia*.txt se leen en solo lectura y nunca se modifican.

Requisitos:  Python 3.8+,  numpy,  simpy,  matplotlib
    pip install numpy simpy matplotlib

Uso:
    python cels_dinamico.py                     # K=1000, rho ∈ {0.60, 0.80, 0.95}
    python cels_dinamico.py --K 100             # prueba rápida
    python cels_dinamico.py --rho 0.8           # un solo rho
    python cels_dinamico.py --no-plots          # solo tablas

Salidas (carpeta 'resultados_dinamicos' junto al script, o --outdir):
    tabla_principal.*              (.csv, .md, .png)  β=1/2 por instancia y rho
    tabla_ranking.*                (.csv, .md, .png)  ε y P(C>C_proy) para β=0, 1/2, 1
    ranking_dinamico_rhoXX.png     ε medio por instancia y β (una figura por rho)
    montecarlo_dinamico_rhoXX.png  histogramas de C_exec por instancia y β (una figura por rho)
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
import simpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-9
BETAS = (0.0, 0.5, 1.0)
RHOS = (0.60, 0.80, 0.95)
COLORS_BETA = {0.0: '#d9d9d9', 0.5: '#7f7f7f', 1.0: '#000000'}   # estilo Fig. 2 del paper


def beta_label(b):
    return {0.0: '0', 0.5: '1/2', 1.0: '1'}[b]


def fmt(x, d=4):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def rho_tag(rho):
    return f"rho{int(round(rho * 100))}"


# =============================================================================
# 1. Números difusos triangulares (TFN), vectorizados
# =============================================================================
def defuzzify_array(d, beta):
    """
    E_β sobre TFN con forma (..., 3): E_β = β·(a1+a2)/2 + (1−β)·(a2+a3)/2.
    β=0 -> tiempos altos (pesimista); β=1 -> tiempos bajos (optimista).
    """
    return beta * (d[..., 0] + d[..., 1]) / 2.0 + (1.0 - beta) * (d[..., 1] + d[..., 2]) / 2.0


def _parse_tfn(token):
    vals = [float(x) for x in token.split(',')]
    if len(vals) != 3:
        raise ValueError(f"TFN inválido: '{token}'")
    return vals


# =============================================================================
# 2. Instancia FFJSP (lectura en solo lectura)
# =============================================================================
class FfJSPInstance:
    def __init__(self, name, job_ops, durations):
        self.name = name
        self.job_ops = job_ops                       # job_ops[j] = ids de operación en orden de etapa
        self.n_jobs = len(job_ops)
        self.n_ops = sum(len(ops) for ops in job_ops)
        self.n_machines = durations.shape[1]
        self.durations = durations                   # np.ndarray (n_ops, M, 3)
        self.mu_hat = 1.0 / float(defuzzify_array(durations, 0.5).mean())

    @classmethod
    def from_file(cls, path):
        """Lee filas 'Job j  etapa  celda_M1 ... celda_Mk' con celdas 'a,b,c'."""
        pattern = re.compile(r'^\s*Job\s+(\d+)\s+(\d+)\s+(.*)$')
        rows, n_machines = {}, None
        with open(path, encoding='utf-8') as f:            # solo lectura
            for line in f:
                m = pattern.match(line)
                if not m:
                    continue
                job = int(m.group(1)) - 1
                cells = [_parse_tfn(tok) for tok in m.group(3).split()]
                if n_machines is None:
                    n_machines = len(cells)
                elif len(cells) != n_machines:
                    raise ValueError(f"{path}: número de máquinas inconsistente en Job {job + 1}")
                rows.setdefault(job, []).append(cells)
        if not rows:
            raise ValueError(f"{path}: no se encontraron filas de datos")
        n_jobs = max(rows) + 1
        job_ops, flat, op = [], [], 0
        for j in range(n_jobs):
            ids = []
            for cells in rows.get(j, []):
                ids.append(op)
                flat.append(cells)
                op += 1
            job_ops.append(ids)
        name = os.path.splitext(os.path.basename(path))[0]
        return cls(name, job_ops, np.array(flat, dtype=float))


# =============================================================================
# 3. Muestreo Monte Carlo: Triangular(a1, a2, a3) con numpy
# =============================================================================
def sample_triangular(durations, K, rng):
    """K realizaciones de duraciones reales, forma (K, n_ops, M)."""
    a1 = durations[..., 0]
    a3 = np.maximum(durations[..., 2], a1 + 1e-9)
    a2 = np.clip(durations[..., 1], a1, a3)
    return rng.triangular(a1, a2, a3, size=(K,) + a1.shape)


# =============================================================================
# 4. Simulación SimPy: ruteo (Ec. 6) y despacho (Ec. 7) separados en el tiempo
# =============================================================================
def simulate(inst, dec, arrivals, S=None):
    """
    dec:      (n_ops, M) con E_β. Es la única información que usan las decisiones.
    arrivals: (n_jobs,) instantes de llegada de cada trabajo.
    S:        (n_ops, M) duraciones reales. Si es None, se ejecuta con E_β (escenario base).

    Retorna (C_max, tiempo total ocupado de máquinas).
    """
    env = simpy.Environment()
    machines = [simpy.PriorityResource(env, capacity=1) for _ in range(inst.n_machines)]
    completion = np.zeros(inst.n_jobs)
    busy = [0.0]

    def job_proc(j):
        yield env.timeout(float(arrivals[j]))
        for o in inst.job_ops[j]:
            # Ec. 6 (ruteo): al quedar lista la operación, elige la máquina con menor E_β.
            m = int(np.argmin(dec[o]))
            # Ec. 7 (despacho): entra a la cola de m con prioridad E_β(p_om).
            with machines[m].request(priority=float(dec[o, m])) as req:
                yield req
                dur = float(dec[o, m]) if S is None else float(S[o, m])
                busy[0] += dur
                yield env.timeout(dur)
        completion[j] = env.now

    for j in range(inst.n_jobs):
        env.process(job_proc(j))
    env.run()
    return float(completion.max()), busy[0]


def evaluate(inst, beta, arrivals, S_all):
    """Corre el baseline C_proy y las K réplicas C_exec para un β dado."""
    dec = defuzzify_array(inst.durations, beta)
    c_proy, busy_proy = simulate(inst, dec, arrivals)
    rho_real = busy_proy / (inst.n_machines * c_proy)
    C = np.array([simulate(inst, dec, arrivals, S_all[k])[0] for k in range(S_all.shape[0])])
    eps = float(np.mean(np.abs(C - c_proy) / c_proy))                   # Ec. 8
    late = float(np.mean(C > c_proy + EPS))
    return dict(c_proy=c_proy, C=C, c_exec_mean=float(C.mean()),
                c_exec_std=float(C.std()), eps=eps, late=late, rho_real=rho_real)


# =============================================================================
# 5. Tablas y gráficas
# =============================================================================
def save_table(headers, rows, base, title):
    with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write(f"**{title}**\n\n")
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '---|' * len(headers) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(c) for c in r) + ' |\n')
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(headers)), 0.9 + 0.45 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title(title, fontweight='bold')
    fig.savefig(base + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ranking(path, rho, labels, eps_by_beta):
    """
    Gráfica de ranking: ε dinámico medio por instancia y β (estilo Fig. 2 del paper).
    eps_by_beta[b] = lista de ε por instancia, en el mismo orden que labels.
    """
    x = np.arange(len(labels))
    w = 0.8 / len(BETAS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, b in enumerate(BETAS):
        ax.bar(x + (k - 1) * w, eps_by_beta[b], w, label=f"β={beta_label(b)}",
               color=COLORS_BETA[b], edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ε dinámico medio")
    ax.set_title(f"Robustez dinámica (MTO) según método de ranking · ρ objetivo = {rho:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_montecarlo(path, rho, nums, results):
    """Cuadrícula: una fila por instancia, una columna por β."""
    nrows, ncols = len(nums), len(BETAS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows), squeeze=False)
    for i, num in enumerate(nums):
        for k, b in enumerate(BETAS):
            ax = axes[i][k]
            r = results[num][rho][b]
            ax.hist(r['C'], bins=40, color='#9ecae1', edgecolor='k')
            ax.axvline(r['c_proy'], color='red', ls='--', label=f"C_proy={r['c_proy']:.2f}")
            ax.axvline(r['C'].mean(), color='k', ls=':', label=f"media={r['C'].mean():.2f}")
            ax.set_title(f"Inst {num} · β={beta_label(b)}  (ε={r['eps']:.4f})", fontsize=9)
            ax.set_xlabel("Cmax ejecutado")
            ax.set_ylabel("frecuencia")
            ax.legend(fontsize=7)
    fig.suptitle(f"Monte Carlo: Cmax ejecutado por despacho dinámico (MTO) · ρ objetivo = {rho:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 6. Experimento principal
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Esquema dinámico MTO con SimPy para FFJSP difuso")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', default=here)
    parser.add_argument('--outdir', default=os.path.join(here, 'resultados_dinamicos'))
    parser.add_argument('--K', type=int, default=1000, help="réplicas Monte Carlo")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rho', type=float, nargs='+', default=list(RHOS),
                        help="utilizaciones objetivo para calibrar lambda0")
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "Instancia*.txt")),
                   key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
    if not files:
        print(f"No se encontraron 'Instancia*.txt' en: {args.data_dir}")
        return
    os.makedirs(args.outdir, exist_ok=True)

    rhos = tuple(args.rho)
    results, insts, lambdas = {}, {}, {}

    for idx, path in enumerate(files):
        inst = FfJSPInstance.from_file(path)
        num = int(re.search(r'(\d+)', inst.name).group(1))
        insts[num] = inst
        results[num] = {}

        print("=" * 78)
        print(f"Instancia {num} | jobs={inst.n_jobs} | ops={inst.n_ops} | M={inst.n_machines} "
              f"| mu_hat={inst.mu_hat:.5f}")

        for ri, rho in enumerate(rhos):
            lam = rho * inst.mu_hat * inst.n_machines                  # lambda0 = rho·mu_hat·m
            lambdas[(num, rho)] = lam

            rng_arr = np.random.default_rng(args.seed + 1000 * idx + ri)
            arrivals = np.cumsum(rng_arr.exponential(1.0 / lam, size=inst.n_jobs))

            rng_exec = np.random.default_rng(args.seed + 7919 * idx + 101 * ri + 1)
            S_all = sample_triangular(inst.durations, args.K, rng_exec)

            results[num][rho] = {}
            for b in BETAS:
                r = evaluate(inst, b, arrivals, S_all)
                results[num][rho][b] = r
                print(f"  ρ_obj={rho:.2f} β={beta_label(b):>3} | λ0={lam:.5f} | "
                      f"ρ_real={r['rho_real']:.3f} | C_proy={r['c_proy']:8.3f} | "
                      f"C_exec={r['c_exec_mean']:8.3f}±{r['c_exec_std']:.3f} | "
                      f"ε={r['eps']:.5f} | P(C>C_proy)={r['late']:.3f}")

    nums = sorted(results)
    labels = [f"Inst {n}" for n in nums]

    # ---- tabla principal (β = 1/2, por instancia y rho) ----
    headers = ["Instancia", "ρ objetivo", "λ0", "ρ real", "C_proy",
               "C_exec media", "C_exec desv", "ε dinámico", "P(C>C_proy)"]
    rows = []
    for n in nums:
        for rho in rhos:
            r = results[n][rho][0.5]
            rows.append([f"Instancia {n}", f"{rho:.2f}", fmt(lambdas[(n, rho)], 5),
                         fmt(r['rho_real'], 3), fmt(r['c_proy'], 2),
                         fmt(r['c_exec_mean'], 2), fmt(r['c_exec_std'], 2),
                         fmt(r['eps'], 4), fmt(r['late'], 3)])
    save_table(headers, rows, os.path.join(args.outdir, 'tabla_principal'),
               "Esquema dinámico MTO (β=1/2): C_proy, C_exec y ε dinámico")

    # ---- tabla de ranking de β (formato Fig. 2) ----
    headers = ["Instancia", "ρ objetivo"]
    for b in BETAS:
        headers += [f"C_proy β={beta_label(b)}", f"ε β={beta_label(b)}",
                    f"P(C>C_proy) β={beta_label(b)}"]
    headers += ["Mejor β (ε)", "Brecha máx−mín (pp)"]
    rows, spreads = [], []
    eps_grid = {rho: {b: [] for b in BETAS} for rho in rhos}
    for n in nums:
        for rho in rhos:
            r = [f"Instancia {n}", f"{rho:.2f}"]
            eps_vals = {}
            for b in BETAS:
                s = results[n][rho][b]
                r += [fmt(s['c_proy'], 2), fmt(s['eps'], 4), fmt(s['late'], 3)]
                eps_vals[b] = s['eps']
                eps_grid[rho][b].append(s['eps'])
            best = min(BETAS, key=lambda b: eps_vals[b])
            spread = (max(eps_vals.values()) - min(eps_vals.values())) * 100.0
            spreads.append(spread)
            r += [beta_label(best), fmt(spread, 3)]
            rows.append(r)
    save_table(headers, rows, os.path.join(args.outdir, 'tabla_ranking'),
               "Ranking de β en esquema dinámico MTO (formato Fig. 2)")

    # ---- consola: resumen ----
    print("\n" + " | ".join(headers))
    for r in rows:
        print(" | ".join(r))
    print("\nε dinámico medio entre instancias, por rho y β:")
    for rho in rhos:
        vals = "  ".join(f"β={beta_label(b)}: {np.mean(eps_grid[rho][b]):.5f}" for b in BETAS)
        print(f"  ρ={rho:.2f} -> {vals}")
    print(f"\nBrecha media entre métodos: {np.mean(spreads):.3f} pp")
    print("Si la brecha tiende a cero, la ventaja de β=1/2 colapsa en MTO.")

    # ---- gráficas ----
    if not args.no_plots:
        for rho in rhos:
            eps_rho = {b: [results[n][rho][b]['eps'] for n in nums] for b in BETAS}
            plot_ranking(os.path.join(args.outdir, f'ranking_dinamico_{rho_tag(rho)}.png'),
                         rho, labels, eps_rho)
            plot_montecarlo(os.path.join(args.outdir, f'montecarlo_dinamico_{rho_tag(rho)}.png'),
                            rho, nums, results)

    print(f"\nTablas y gráficas guardadas en: {args.outdir}")


if __name__ == '__main__':
    main()