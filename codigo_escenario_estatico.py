#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CELS - Coevolutionary Local Search para el Fuzzy Flexible Job Shop (FFJSP).
Implementación basada en Palacios, González-Rodríguez, Vela & Puente (2015).

Requisitos:  Python 3.8+,  numpy,  matplotlib
    pip install numpy matplotlib

Uso:
    python cels_ffjsp.py                       # tabla principal + ranking de beta + determinista
    python cels_ffjsp.py --runs 3              # promedio de 3 corridas (tarda ~3 veces más)
    python cels_ffjsp.py --ablation            # añade análisis de componentes (Tabla 1)
    python cels_ffjsp.py --no-ranking          # solo la ejecución principal
    python cels_ffjsp.py --gen 20 --K 200      # prueba rápida

Salidas (carpeta 'resultados' junto al script, o --outdir):
    tabla_principal.*, tabla_ranking.*, tabla_determinista.*, tabla_ablacion.*
    (.csv, .md, .png)
    convergencia.png, montecarlo.png, comparacion_paper.png,
    ranking.png (Fig. 2), determinista.png (Fig. 4), ablacion.png
"""

import argparse
import csv
import glob
import os
import random
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-9
TOL = 1e-6
INF = float('inf')
NAN = float('nan')

# ---------------------------------------------------------------------------
# Valores de referencia del paper (Palacios et al., 2015), transcritos de las
# Tablas 1 y 2. Clave = número de instancia (Instancia 1 ... 5).
# ---------------------------------------------------------------------------
PAPER_LB = {1: 28.50, 2: 45.00, 3: 43.50, 4: 33.50, 5: 37.50}
PAPER_CELS_BEST = {1: 28.50, 2: 45.25, 3: 43.50, 4: 34.25, 5: 53.25}
PAPER_CELS_AVG = {1: 28.53, 2: 45.25, 3: 44.18, 4: 35.08, 5: 55.07}
PAPER_TABLE1 = {
    'RP':   {1: (62.48, 83.53), 2: (81.98, 107.39), 3: (90.88, 114.11), 4: (76.25, 95.22), 5: (110.18, 133.92)},
    'HP':   {1: (32.30, 37.18), 2: (47.80, 54.53), 3: (50.23, 56.95), 4: (39.25, 44.85), 5: (63.33, 69.58)},
    'CCEA': {1: (30.25, 31.15), 2: (45.75, 46.60), 3: (47.00, 47.63), 4: (37.25, 38.28), 5: (58.50, 60.43)},
    'CELS': {1: (28.50, 28.53), 2: (45.25, 45.25), 3: (43.50, 44.18), 4: (34.25, 35.08), 5: (53.25, 55.07)},
}
# Variantes de la Tabla 1: (tipo de inicialización, búsqueda local activa)
VARIANTS = {'RP': ('rp', False), 'HP': ('hp', False), 'CCEA': ('mixed', False), 'CELS': ('mixed', True)}
BETA_LABEL = {0.0: '0', 0.5: '1/2', 1.0: '1'}
COLORS_BETA = {0.0: '#d9d9d9', 0.5: '#7f7f7f', 1.0: '#000000'}     # estilo Fig. 2 del paper


# =============================================================================
# 1. Números difusos triangulares (TFN)
# =============================================================================
class TFN:
    """Operaciones sobre TFN A = (a1, a2, a3) representados como tuplas."""

    ZERO = (0.0, 0.0, 0.0)

    @staticmethod
    def add(A, B):
        return (A[0] + B[0], A[1] + B[1], A[2] + B[2])

    @staticmethod
    def max_interval(A, B):
        """Máximo por intervalo (aproximación de Fortemps, 1997)."""
        return (max(A[0], B[0]), max(A[1], B[1]), max(A[2], B[2]))

    @staticmethod
    def defuzzify(A, beta):
        """
        Ecuación (7) del paper: E_beta = beta * E_inf(X) + (1-beta) * E_sup(X),
        con E_inf = (a1+a2)/2 y E_sup = (a2+a3)/2.
        Equivale a (beta/2)*a1 + (1/2)*a2 + ((1-beta)/2)*a3.
        beta=0 -> tiempos altos (pesimista); beta=1 -> tiempos bajos (optimista).
        """
        return beta * (A[0] + A[1]) / 2.0 + (1.0 - beta) * (A[1] + A[2]) / 2.0


def _parse_tfn(token):
    vals = [float(x) for x in token.split(',')]
    if len(vals) != 3:
        raise ValueError(f"TFN inválido: '{token}'")
    return vals


# =============================================================================
# 2. Instancia FFJSP y parser de archivos .txt
# =============================================================================
class FfJSPInstance:
    def __init__(self, name, job_ops, durations):
        self.name = name
        self.job_ops = job_ops                      # job_ops[j] = ids de operación en orden de etapa
        self.n_jobs = len(job_ops)
        self.n_ops = sum(len(ops) for ops in job_ops)
        self.n_machines = durations.shape[1]
        self.durations = durations                  # np.ndarray (n_ops, M, 3)
        self.dur = [[tuple(c) for c in row] for row in durations.tolist()]
        self.job_pred = [-1] * self.n_ops
        for ops in job_ops:
            for k in range(1, len(ops)):
                self.job_pred[ops[k]] = ops[k - 1]

    @classmethod
    def from_file(cls, path):
        """Lee filas 'Job j  etapa  celda_M1 ... celda_Mk' con celdas 'a,b,c'."""
        pattern = re.compile(r'^\s*Job\s+(\d+)\s+(\d+)\s+(.*)$')
        rows, n_machines = {}, None
        with open(path, encoding='utf-8') as f:
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
# 3. Decodificador (motor de inserción) y rutas críticas
# =============================================================================
class Schedule:
    __slots__ = ('start_d', 'end_d', 'start_f', 'end_f',
                 'cmax_f', 'cmax_d', 'machine_seq', 'op_pos')


def decode(inst, alpha, pi, beta, dur=None):
    """
    Recorre la permutación con repetición pi y coloca cada operación en el
    intervalo de inactividad más temprano de su máquina (alpha), respetando la
    precedencia del job. Tiempos TFN; las decisiones se toman con E_beta.
    """
    if dur is None:
        dur = inst.dur
    n, M = inst.n_ops, inst.n_machines
    busy = [[] for _ in range(M)]                   # huecos: [s_d, e_d, s_f, e_f, op]
    job_ready_f = [TFN.ZERO] * inst.n_jobs
    job_ready_d = [0.0] * inst.n_jobs
    next_stage = [0] * inst.n_jobs
    start_f, end_f = [TFN.ZERO] * n, [TFN.ZERO] * n
    start_d, end_d = [0.0] * n, [0.0] * n
    op_pos = [0] * n

    for t, j in enumerate(pi):
        o = inst.job_ops[j][next_stage[j]]
        next_stage[j] += 1
        m = alpha[o]
        p = dur[o][m]
        pd = TFN.defuzzify(p, beta)
        R_f, R_d = job_ready_f[j], job_ready_d[j]

        slots = busy[m]
        prev_d, prev_f, idx = 0.0, TFN.ZERO, len(slots)
        for k, slot in enumerate(slots):
            if max(prev_d, R_d) + pd <= slot[0] + EPS:
                idx = k
                break
            prev_d, prev_f = slot[1], slot[3]

        s_f = TFN.max_interval(R_f, prev_f)
        e_f = TFN.add(s_f, p)
        s_d = TFN.defuzzify(s_f, beta)
        e_d = TFN.defuzzify(e_f, beta)
        slots.insert(idx, [s_d, e_d, s_f, e_f, o])

        start_f[o], end_f[o] = s_f, e_f
        start_d[o], end_d[o] = s_d, e_d
        job_ready_f[j], job_ready_d[j] = e_f, e_d
        op_pos[o] = t

    sch = Schedule()
    sch.start_f, sch.end_f = start_f, end_f
    sch.start_d, sch.end_d = start_d, end_d
    sch.cmax_f = (max(e[0] for e in end_f), max(e[1] for e in end_f), max(e[2] for e in end_f))
    sch.cmax_d = TFN.defuzzify(sch.cmax_f, beta)
    sch.machine_seq = [[s[4] for s in busy[m]] for m in range(M)]
    sch.op_pos = op_pos
    return sch


def critical_path(inst, sch):
    """Ruta crítica (lista de operaciones) con preferencia por arcos de máquina."""
    n = inst.n_ops
    mach_pred = [-1] * n
    for seq in sch.machine_seq:
        for k in range(1, len(seq)):
            mach_pred[seq[k]] = seq[k - 1]

    cur = max(range(n), key=lambda o: sch.end_d[o])
    path = [cur]
    while sch.start_d[cur] > EPS:
        s = sch.start_d[cur]
        mp, jp = mach_pred[cur], inst.job_pred[cur]
        if mp >= 0 and abs(sch.end_d[mp] - s) < TOL:
            nxt = mp
        elif jp >= 0 and abs(sch.end_d[jp] - s) < TOL:
            nxt = jp
        else:
            break
        path.append(nxt)
        cur = nxt
    path.reverse()
    return path, mach_pred


def critical_blocks(path, mach_pred):
    """Bloques críticos: subsecuencias máximas consecutivas en la misma máquina."""
    blocks, cur = [], [path[0]]
    for a, b in zip(path, path[1:]):
        if mach_pred[b] == a:
            cur.append(b)
        else:
            blocks.append(cur)
            cur = [b]
    blocks.append(cur)
    return [b for b in blocks if len(b) >= 2]


# =============================================================================
# 4. Individuos de las dos poblaciones
# =============================================================================
class IndividualPM:
    """P^M: asignación de máquinas alpha (una máquina por operación)."""
    __slots__ = ('alpha', 'fitness')

    def __init__(self, alpha, fitness=INF):
        self.alpha, self.fitness = alpha, fitness


class IndividualPT:
    """P^T: permutación con repetición de IDs de job (Bierwirth, 1995)."""
    __slots__ = ('pi', 'fitness')

    def __init__(self, pi, fitness=INF):
        self.pi, self.fitness = pi, fitness


# =============================================================================
# 5. Solver CELS
# =============================================================================
class CELSSolver:
    def __init__(self, inst, beta=0.5, pop_size=50, generations=100,
                 pc_m=0.90, pm_m=0.05, pc_t=0.90, pm_t=0.05,
                 init_mode='mixed', use_ls=True,
                 ls_iter=2, ls_neighbors=10, seed=None, verbose=True):
        self.inst = inst
        self.beta = beta
        self.N = pop_size
        self.G = generations
        self.pc_m, self.pm_m = pc_m, pm_m
        self.pc_t, self.pm_t = pc_t, pm_t
        self.init_mode = init_mode                  # 'hp', 'rp' o 'mixed'
        self.use_ls = use_ls
        self.ls_iter = ls_iter
        self.ls_neighbors = ls_neighbors
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.PM, self.PT = [], []
        self.best_alpha, self.best_pi = None, None
        self.best_sch, self.best_fit = None, INF
        self.history = []

    def _update_best(self, alpha, pi, sch):
        if sch.cmax_d < self.best_fit - EPS:
            self.best_fit = sch.cmax_d
            self.best_alpha, self.best_pi = list(alpha), list(pi)
            self.best_sch = sch

    def _tournament(self, pop):
        a, b = self.rng.sample(pop, 2)
        return a if a.fitness <= b.fitness else b

    # ----- inicialización ---------------------------------------------------
    def _heuristic_solution(self):
        """Inserción heurística: en cada paso escoge el par (job, máquina) de menor ECT (con ruido)."""
        inst, beta, M = self.inst, self.beta, self.inst.n_machines
        next_op = [0] * inst.n_jobs
        job_ready = [0.0] * inst.n_jobs
        mach_ready = [0.0] * M
        alpha, pi = [0] * inst.n_ops, []
        for _ in range(inst.n_ops):
            best_key, choice = None, None
            for j in range(inst.n_jobs):
                if next_op[j] >= len(inst.job_ops[j]):
                    continue
                o = inst.job_ops[j][next_op[j]]
                for m in range(M):
                    ect = max(job_ready[j], mach_ready[m]) + TFN.defuzzify(inst.dur[o][m], beta)
                    key = ect * (1.0 + 0.15 * self.rng.random())
                    if best_key is None or key < best_key:
                        best_key, choice = key, (j, o, m, ect)
            j, o, m, ect = choice
            alpha[o] = m
            pi.append(j)
            next_op[j] += 1
            job_ready[j] = ect
            mach_ready[m] = ect
        return alpha, pi

    def _random_solution(self):
        inst = self.inst
        alpha = [self.rng.randrange(inst.n_machines) for _ in range(inst.n_ops)]
        pi = [j for j, ops in enumerate(inst.job_ops) for _ in ops]
        self.rng.shuffle(pi)
        return alpha, pi

    def _init_populations(self):
        N = self.N
        for i in range(N):
            if self.init_mode == 'hp' or (self.init_mode == 'mixed' and i < N // 2):
                alpha, pi = self._heuristic_solution()
            else:
                alpha, pi = self._random_solution()
            self.PM.append(IndividualPM(alpha))
            self.PT.append(IndividualPT(pi))

        a0, p0 = self.PM[0].alpha, self.PT[0].pi
        self._update_best(a0, p0, decode(self.inst, a0, p0, self.beta))
        for i in range(N):
            a, f = self._evaluate_M(self.PM[i].alpha, i)
            self.PM[i].alpha, self.PM[i].fitness = a, f
        for i in range(N):
            p, f = self._evaluate_T(self.PT[i].pi, i)
            self.PT[i].pi, self.PT[i].fitness = p, f

    # ----- evaluación cooperativa con 3 socios ------------------------------
    def _evaluate_M(self, alpha, pos):
        """Combina alpha con 3 socios de P^T (Best, aleatorio, posicional); BL sobre el mejor."""
        j = self.rng.choice(self.PT).pi
        partners = (self.best_pi, j, self.PT[pos].pi)
        best_sch, best_pi = None, None
        for pi in partners:
            sch = decode(self.inst, alpha, pi, self.beta)
            if best_sch is None or sch.cmax_d < best_sch.cmax_d:
                best_sch, best_pi = sch, pi
        if self.use_ls:
            na, np_, nsch = self.local_search(alpha, best_pi, best_sch)
        else:
            na, np_, nsch = list(alpha), best_pi, best_sch
        self._update_best(na, np_, nsch)
        return na, nsch.cmax_d

    def _evaluate_T(self, pi, pos):
        j = self.rng.choice(self.PM).alpha
        partners = (self.best_alpha, j, self.PM[pos].alpha)
        best_sch, best_alpha = None, None
        for alpha in partners:
            sch = decode(self.inst, alpha, pi, self.beta)
            if best_sch is None or sch.cmax_d < best_sch.cmax_d:
                best_sch, best_alpha = sch, alpha
        if self.use_ls:
            na, np_, nsch = self.local_search(best_alpha, pi, best_sch)
        else:
            na, np_, nsch = best_alpha, list(pi), best_sch
        self._update_best(na, np_, nsch)
        return np_, nsch.cmax_d

    # ----- búsqueda local (hill climbing sobre rutas críticas) --------------
    def local_search(self, alpha, pi, sch):
        """
        P^M: reasigna operaciones críticas a otras máquinas elegibles.
        P^T: invierte el primer y el último arco de cada bloque crítico.
        Muestra de movimientos; solo se aceptan mejoras del makespan E_beta.
        """
        cur_a, cur_p = list(alpha), list(pi)
        M = self.inst.n_machines
        for _ in range(self.ls_iter):
            path, mach_pred = critical_path(self.inst, sch)
            moves = [('M', o, m) for o in path for m in range(M) if m != cur_a[o]]
            for block in critical_blocks(path, mach_pred):
                moves.append(('T', block[0], block[1]))
                if len(block) > 2:
                    moves.append(('T', block[-2], block[-1]))
            if len(moves) > self.ls_neighbors:
                moves = self.rng.sample(moves, self.ls_neighbors)

            improved = False
            for kind, x, y in moves:
                if kind == 'M':
                    na, np_ = list(cur_a), cur_p
                    na[x] = y
                else:
                    pa, pb = sch.op_pos[x], sch.op_pos[y]
                    if cur_p[pa] == cur_p[pb]:
                        continue
                    na, np_ = cur_a, list(cur_p)
                    np_[pa], np_[pb] = np_[pb], np_[pa]
                ns = decode(self.inst, na, np_, self.beta)
                if ns.cmax_d < sch.cmax_d - EPS:
                    cur_a, cur_p, sch = na, np_, ns
                    improved = True
                    break
            if not improved:
                break
        return cur_a, cur_p, sch

    # ----- operadores -------------------------------------------------------
    def _cross_M(self, a, b):
        c = self.rng.randint(1, len(a) - 1)
        return a[:c] + b[c:], b[:c] + a[c:]

    def _cross_T(self, p1, p2):
        """JOX (Ono et al., 1996)."""
        jobs = {j for j in range(self.inst.n_jobs) if self.rng.random() < 0.5}

        def child(a, b):
            fill = iter([j for j in b if j not in jobs])
            return [j if j in jobs else next(fill) for j in a]

        return child(p1, p2), child(p2, p1)

    def _mut_M(self, alpha):
        a = list(alpha)
        for o in range(len(a)):
            if self.rng.random() < self.pm_m:
                a[o] = self.rng.randrange(self.inst.n_machines)
        return a

    def _mut_T(self, pi):
        p = list(pi)
        if self.rng.random() < self.pm_t:
            tok = p.pop(self.rng.randrange(len(p)))
            p.insert(self.rng.randrange(len(p) + 1), tok)
        return p

    # ----- generación -------------------------------------------------------
    def _evolve(self, kind):
        pop = self.PM if kind == 'M' else self.PT
        pc = self.pc_m if kind == 'M' else self.pc_t
        mutate = self._mut_M if kind == 'M' else self._mut_T
        cross = self._cross_M if kind == 'M' else self._cross_T

        quads, children = [], []
        for _ in range(self.N // 2):
            p1, p2 = self._tournament(pop), self._tournament(pop)
            g1 = p1.alpha if kind == 'M' else p1.pi
            g2 = p2.alpha if kind == 'M' else p2.pi
            c1, c2 = cross(g1, g2) if self.rng.random() < pc else (list(g1), list(g2))
            quads.append((p1, p2))
            children.append(mutate(c1))
            children.append(mutate(c2))

        child_objs = []
        for k, g in enumerate(children):
            if kind == 'M':
                a, f = self._evaluate_M(g, k)
                child_objs.append(IndividualPM(a, f))
            else:
                p, f = self._evaluate_T(g, k)
                child_objs.append(IndividualPT(p, f))

        new_pop = []
        for q, (p1, p2) in enumerate(quads):
            quartet = sorted([p1, p2, child_objs[2 * q], child_objs[2 * q + 1]], key=lambda x: x.fitness)
            new_pop.extend(quartet[:2])

        worst = max(range(len(new_pop)), key=lambda i: new_pop[i].fitness)
        if kind == 'M':
            new_pop[worst] = IndividualPM(list(self.best_alpha), self.best_fit)
            self.PM = new_pop
        else:
            new_pop[worst] = IndividualPT(list(self.best_pi), self.best_fit)
            self.PT = new_pop

    def run(self):
        self._init_populations()
        for g in range(1, self.G + 1):
            self._evolve('M')
            self._evolve('T')
            self.history.append(self.best_fit)
            if self.verbose and (g == 1 or g % 10 == 0 or g == self.G):
                print(f"   Gen {g:3d}/{self.G} | mejor E(Cmax) = {self.best_fit:.3f}")
        return self.best_alpha, self.best_pi, self.best_sch


# =============================================================================
# 6. Monte Carlo con transformación pignística
# =============================================================================
class MonteCarloEvaluator:
    """
    Transformación pignística (Dubois, Prade & Sandri, 1993): alpha ~ U(0,1) y
    x ~ U(corte alpha), con corte alpha = [a1 + alpha(a2-a1), a3 - alpha(a3-a2)].
    """

    def __init__(self, inst, beta=0.5, K=1000, seed=None):
        self.inst, self.beta, self.K = inst, beta, K
        self.rng = np.random.default_rng(seed)

    def sample_pignistic(self):
        d = self.inst.durations
        shape = (self.K,) + d.shape[:2]
        a = self.rng.random(shape)
        v = self.rng.random(shape)
        lo = d[..., 0] + a * (d[..., 1] - d[..., 0])
        hi = d[..., 2] - a * (d[..., 2] - d[..., 1])
        return lo + v * (hi - lo)

    def evaluate(self, alpha, sch, E_cmax):
        """
        Ejecución en planta del horario predictivo: se conservan la asignación alpha y la
        SECUENCIA de cada máquina del horario predictivo; solo cambian las duraciones.
        Eq. (10): eps = (1/K) * sum_k |C_k - E_beta(C_pred)| / E_beta(C_pred).
        """
        S = self.sample_pignistic()
        order = sorted(range(self.inst.n_ops), key=lambda o: (sch.start_d[o], sch.end_d[o]))
        C = np.empty(self.K)
        for k in range(self.K):
            C[k] = execute_fixed_sequence(self.inst, alpha, sch.machine_seq, order, S[k])
        eps_bar = float(np.mean(np.abs(C - E_cmax) / E_cmax))
        return C, eps_bar


def execute_fixed_sequence(inst, alpha, machine_seq, order, dur_crisp):
    """
    Calcula el makespan real de un escenario con las secuencias por máquina FIJAS.
    order: operaciones en orden de inicio del horario predictivo (respeta precedencias).
    dur_crisp: array (n_ops, M) con duraciones realizadas.
    """
    n = inst.n_ops
    mach_pred = [-1] * n
    for seq in machine_seq:
        for k in range(1, len(seq)):
            mach_pred[seq[k]] = seq[k - 1]
    end = [0.0] * n
    for o in order:
        start = 0.0
        jp, mp = inst.job_pred[o], mach_pred[o]
        if jp >= 0:
            start = max(start, end[jp])
        if mp >= 0:
            start = max(start, end[mp])
        end[o] = start + float(dur_crisp[o, alpha[o]])
    return max(end)


# =============================================================================
# 7. Experimentos, tablas y gráficas
# =============================================================================
def run_variant(inst, inst_idx, variant, beta, runs, args, label="", eval_inst=None):
    """
    Ejecuta CELS (o una variante) 'runs' veces. El horario se optimiza sobre 'inst';
    el error epsilon se mide con Monte Carlo sobre 'eval_inst' (por defecto, el mismo inst).
    """
    init, use_ls = VARIANTS[variant]
    eval_inst = inst if eval_inst is None else eval_inst
    results = []
    for r in range(runs):
        seed = args.seed + inst_idx + 101 * r
        solver = CELSSolver(inst, beta=beta, pop_size=args.pop, generations=args.gen,
                            init_mode=init, use_ls=use_ls, ls_iter=args.ls_iter,
                            ls_neighbors=args.ls_neighbors, seed=seed, verbose=False)
        t0 = time.time()
        alpha, pi, sch = solver.run()
        elapsed = time.time() - t0
        mc = MonteCarloEvaluator(eval_inst, beta=beta, K=args.K, seed=seed)
        C, eps = mc.evaluate(alpha, sch, sch.cmax_d)
        late = float(np.mean(C > sch.cmax_d))                  # P(C_k > E_beta): horario retrasado
        mad = float(np.mean(np.abs(C - sch.cmax_d)))           # desviación absoluta media
        results.append(dict(E=sch.cmax_d, eps=eps, late=late, mad=mad,
                            time=elapsed, C=C, history=solver.history))
        print(f"   {label}{variant} corrida {r + 1}/{runs}: E={sch.cmax_d:.3f}  "
              f"eps={eps:.4f}  t={elapsed:.1f}s")
    return results


def summarize(results):
    best = min(results, key=lambda x: x['E'])
    return dict(E_best=best['E'], E_avg=float(np.mean([r['E'] for r in results])),
                eps_best=best['eps'], eps_avg=float(np.mean([r['eps'] for r in results])),
                time_avg=float(np.mean([r['time'] for r in results])),
                late_avg=float(np.mean([r['late'] for r in results])),
                mad_avg=float(np.mean([r['mad'] for r in results])),
                best_run=best)


def fmt(x, d=4):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def mre(value, lb):
    return NAN if lb is None or np.isnan(lb) else (value - lb) / lb * 100.0


def save_table(headers, rows, base, title):
    """Guarda la tabla en CSV, Markdown y PNG."""
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
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(headers)), 0.9 + 0.45 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title(title, fontweight='bold')
    fig.savefig(base + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def grouped_bars(ax, labels, series, title, ylabel, colors=None, edge='k'):
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(series))
    for k, (name, vals) in enumerate(series.items()):
        color = colors[k] if colors else None
        ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, w, label=name, color=color, edgecolor=edge)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()


def _grid(n):
    cols = 3
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 3.5 * rows_), squeeze=False)
    return fig, list(axes.flat)


def plot_convergence(main, path):
    nums = sorted(main)
    fig, axes = _grid(len(nums))
    for ax, num in zip(axes, nums):
        h = main[num]['s']['best_run']['history']
        ax.plot(range(1, len(h) + 1), h, marker='.')
        if PAPER_CELS_BEST.get(num):
            ax.axhline(PAPER_CELS_BEST[num], color='green', ls='--', label='CELS paper (mejor)')
            ax.legend()
        ax.set_title(f"Instancia {num}")
        ax.set_xlabel("Generación")
        ax.set_ylabel("Mejor E(Cmax)")
    for ax in axes[len(nums):]:
        ax.axis('off')
    fig.suptitle("Convergencia de CELS (mejor corrida)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_montecarlo(main, path):
    nums = sorted(main)
    fig, axes = _grid(len(nums))
    for ax, num in zip(axes, nums):
        br = main[num]['s']['best_run']
        C, E = br['C'], br['E']
        ax.hist(C, bins=40, color='#9ecae1', edgecolor='k')
        ax.axvline(E, color='red', ls='--', label=f"E(Cmax)={E:.2f}")
        ax.axvline(C.mean(), color='k', ls=':', label=f"media={C.mean():.2f}")
        ax.set_title(f"Instancia {num}  (eps={br['eps']:.4f})")
        ax.set_xlabel("Cmax simulado")
        ax.set_ylabel("frecuencia")
        ax.legend(fontsize=8)
    for ax in axes[len(nums):]:
        ax.axis('off')
    fig.suptitle("Monte Carlo K escenarios pignísticos (mejor corrida)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CELS para FFJSP (Palacios et al., 2015)")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', default=here)
    parser.add_argument('--outdir', default=os.path.join(here, 'resultados'))
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--gen', type=int, default=100)
    parser.add_argument('--K', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1, help="corridas independientes por configuración")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ls-iter', type=int, default=2)
    parser.add_argument('--ls-neighbors', type=int, default=10)
    parser.add_argument('--ablation', action='store_true', help="RP, HP, CCEA y CELS (Tabla 1)")
    parser.add_argument('--no-ranking', action='store_true',
                        help="omite el ranking de beta (Fig. 2) y el esquema determinista (Fig. 4)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "Instancia*.txt")),
                   key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
    if not files:
        print(f"No se encontraron 'Instancia*.txt' en: {args.data_dir}")
        return
    os.makedirs(args.outdir, exist_ok=True)

    instances = {}
    for idx, path in enumerate(files):
        inst = FfJSPInstance.from_file(path)
        num = int(re.search(r'(\d+)', inst.name).group(1))
        instances[num] = (idx, inst)
    nums = sorted(instances)
    labels = [f"Inst {n}" for n in nums]

    # ---- ejecución principal (beta = 0.5) ----
    main_res = {}
    for num in nums:
        idx, inst = instances[num]
        print("=" * 72)
        print(f"Instancia {num} ({inst.name}) | jobs={inst.n_jobs} | ops={inst.n_ops} | M={inst.n_machines}")
        res = run_variant(inst, idx, 'CELS', args.beta, args.runs, args)
        s = summarize(res)
        main_res[num] = dict(inst=inst, res=res, s=s)
        print(f"  -> E best={s['E_best']:.3f}  E avg={s['E_avg']:.3f}  "
              f"eps best={s['eps_best']:.4f}  eps avg={s['eps_avg']:.4f}  t={s['time_avg']:.1f}s")

    headers = ["Instancia", "LB", "E best", "E avg", "MRE best (%)", "MRE avg (%)",
               "eps best", "eps avg", "t prom (s)", "Paper best", "Paper avg"]
    rows = []
    for num in nums:
        s = main_res[num]['s']
        lb = PAPER_LB.get(num, NAN)
        rows.append([f"Instancia {num}", fmt(lb, 2), fmt(s['E_best'], 2), fmt(s['E_avg'], 2),
                     fmt(mre(s['E_best'], lb), 2), fmt(mre(s['E_avg'], lb), 2),
                     fmt(s['eps_best'], 4), fmt(s['eps_avg'], 4), fmt(s['time_avg'], 1),
                     fmt(PAPER_CELS_BEST.get(num, NAN), 2), fmt(PAPER_CELS_AVG.get(num, NAN), 2)])
    save_table(headers, rows, os.path.join(args.outdir, 'tabla_principal'),
               "Resultados CELS (formato Tabla 2)")
    print("\n" + " | ".join(headers))
    for r in rows:
        print(" | ".join(r))

    plot_convergence(main_res, os.path.join(args.outdir, 'convergencia.png'))
    plot_montecarlo(main_res, os.path.join(args.outdir, 'montecarlo.png'))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    grouped_bars(ax, labels, {
        'LB': [PAPER_LB.get(n, NAN) for n in nums],
        'CELS paper (avg)': [PAPER_CELS_AVG.get(n, NAN) for n in nums],
        'Nuestro CELS (avg)': [main_res[n]['s']['E_avg'] for n in nums],
    }, "E(Cmax): comparación con el paper", "E(Cmax)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'comparacion_paper.png'), dpi=150)
    plt.close(fig)

    # ---- ablación (Tabla 1) ----
    if args.ablation:
        print("\n== Análisis de componentes (Tabla 1) ==")
        abl = {'CELS': {n: main_res[n]['s'] for n in nums}}
        for v in ['RP', 'HP', 'CCEA']:
            abl[v] = {n: summarize(run_variant(instances[n][1], instances[n][0], v, args.beta,
                                               args.runs, args, label="[abl] ")) for n in nums}
        h = ["Instancia"]
        for v in ['RP', 'HP', 'CCEA', 'CELS']:
            h += [f"{v} best(avg)", f"Paper {v}"]
        arows = []
        for n in nums:
            r = [f"Instancia {n}"]
            for v in ['RP', 'HP', 'CCEA', 'CELS']:
                s = abl[v][n]
                r.append(f"{fmt(s['E_best'], 2)} ({fmt(s['E_avg'], 2)})")
                r.append(f"{fmt(PAPER_TABLE1[v][n][0], 2)} ({fmt(PAPER_TABLE1[v][n][1], 2)})")
            arows.append(r)
        save_table(h, arows, os.path.join(args.outdir, 'tabla_ablacion'),
                   "Componentes del algoritmo (formato Tabla 1)")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        grouped_bars(axes[0], labels, {v: [abl[v][n]['E_avg'] for n in nums] for v in ['RP', 'HP', 'CCEA', 'CELS']},
                     "Nuestra implementación (avg)", "E(Cmax)")
        grouped_bars(axes[1], labels, {v: [PAPER_TABLE1[v][n][1] for n in nums] for v in ['RP', 'HP', 'CCEA', 'CELS']},
                     "Paper (avg)", "E(Cmax)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'ablacion.png'), dpi=150)
        plt.close(fig)

    if not args.no_ranking:
        # ---- ranking de beta: error epsilon para beta = 0, 1/2 y 1 (Fig. 2) ----
        print("\n== Ranking de beta (Fig. 2) ==")
        betas = [0.0, 0.5, 1.0]
        rank = {0.5: {n: main_res[n]['s'] for n in nums}}        # beta = 1/2 ya calculado
        for b in [0.0, 1.0]:
            rank[b] = {n: summarize(run_variant(instances[n][1], instances[n][0], 'CELS', b,
                                                args.runs, args, label=f"[b={b}] ")) for n in nums}

        # eps con denominador común: E de beta=1/2 para los tres horarios de la misma instancia
        h = ["Instancia", "E beta=0", "eps beta=0", "P(C>E) beta=0", "eps comun beta=0",
             "E beta=1/2", "eps beta=1/2", "P(C>E) beta=1/2", "eps comun beta=1/2",
             "E beta=1", "eps beta=1", "P(C>E) beta=1", "eps comun beta=1", "beta elegido (eps)"]
        rrows = []
        for n in nums:
            e_ref = rank[0.5][n]['E_avg']
            elegido = min(betas, key=lambda b: rank[b][n]['eps_avg'])
            r = [f"Instancia {n}"]
            for b in betas:
                s = rank[b][n]
                r += [fmt(s['E_avg'], 2), fmt(s['eps_avg'], 4), fmt(s['late_avg'], 3),
                      fmt(s['mad_avg'] / e_ref, 4)]
            r.append(BETA_LABEL[elegido])
            rrows.append(r)
        save_table(h, rrows, os.path.join(args.outdir, 'tabla_ranking'),
                   "Ranking de beta: E y error epsilon por instancia (formato Fig. 2)")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        grouped_bars(ax, labels, {f"beta={BETA_LABEL[b]}": [rank[b][n]['eps_avg'] for n in nums] for b in betas},
                     "Media de eps según método de ranking", "eps medio",
                     colors=[COLORS_BETA[b] for b in betas])
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'ranking.png'), dpi=150)
        plt.close(fig)

        # ---- determinista vs difuso (Fig. 4) ----
        print("\n== Horario determinista vs difuso (Fig. 4) ==")
        det = {}
        for n in nums:
            idx, inst = instances[n]
            # Determinista: se optimiza con la duración modal a2 (sin incertidumbre)
            crisp = FfJSPInstance(inst.name, inst.job_ops,
                                  np.repeat(inst.durations[:, :, 1:2], 3, axis=2))
            det[n] = summarize(run_variant(crisp, idx, 'CELS', 0.5, args.runs, args,
                                           label="[det] ", eval_inst=inst))
        h = ["Instancia", "E difuso beta=1/2", "eps difuso", "E determinista", "eps determinista"]
        drows = [[f"Instancia {n}", fmt(rank[0.5][n]['E_avg'], 2), fmt(rank[0.5][n]['eps_avg'], 4),
                  fmt(det[n]['E_avg'], 2), fmt(det[n]['eps_avg'], 4)] for n in nums]
        save_table(h, drows, os.path.join(args.outdir, 'tabla_determinista'),
                   "Horario difuso (beta=1/2) frente a determinista (formato Fig. 4)")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        grouped_bars(ax, labels, {
            'beta=1/2 (difuso)': [rank[0.5][n]['eps_avg'] for n in nums],
            'determinista': [det[n]['eps_avg'] for n in nums],
        }, "Error eps: horario difuso frente a determinista", "eps medio",
            colors=['#000000', '#a6a6a6'])
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'determinista.png'), dpi=150)
        plt.close(fig)

    print(f"\nTablas y gráficas guardadas en: {args.outdir}")


if __name__ == '__main__':
    main()