# -*- coding: utf-8 -*-
"""诊断：交错环形布局各尺寸/塔位 效率与 60MW 可行性"""
import numpy as np, pandas as pd, time, sys
from layout import ring_layout
from common import (solar_position, TOWER_H, ETA_REF, mirror_normal, eta_cos,
                    eta_at, D_MONTHS, TIMES)
from shadow import compute_eta_sb
from truncation import compute_eta_trunc


def comp(C, tower_xy, w, h, K=4):
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    cos = at = sb = tr = 0.0
    for D in D_MONTHS:
        for ST in TIMES:
            s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
            n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
            cos += (eta_cos(s, r_hat) * A).sum() / (A * N)
            at += (eta_at(dHR) * A).sum() / (A * N)
            sb += (compute_eta_sb(C, tower_xy, s, w, h, K=K) * A).sum() / (A * N)
            tr += (compute_eta_trunc(C, tower_xy, s, w, h, K=K, Nr=2, Nphi=8)
                   * A).sum() / (A * N)
    n = 60.0
    return cos / n, at / n, sb / n, tr / n


if __name__ == '__main__':
    cases = []
    for tower in [(0, 0), (0, -120), (0, 80), (0, -60)]:
        for (w, h) in [(8, 8), (7, 7), (6, 6), (6, 5), (6, 4)]:
            for (dr, da) in [(1.0, 1.2), (1.3, 1.4)]:
                cases.append((tower, w, h, dr, da))
    print(f"共 {len(cases)} 个布局诊断（K=4）", flush=True)
    t_all = time.time()
    for tower, w, h, dr, da in cases:
        h_inst = max(4.0, h / 2 + 1.0)
        C = ring_layout(tower, w, h, h_inst, dr, da)
        if len(C) < 100:
            continue
        t0 = time.time()
        c, a, sb, tr = comp(C, tower, w, h, K=4)
        eta = c * a * sb * tr * ETA_REF
        A = len(C) * w * h
        E = 0.968 * eta * A / 1000.0   # MW 近似
        print(f"塔{tower} {w}x{h} dr{dr}da{da} N={len(C):4d} A={A:7.0f} "
              f"cos={c:.4f} at={a:.4f} sb={sb:.4f} tr={tr:.4f} "
              f"η={eta:.4f} E≈{E:.1f}MW [{time.time()-t0:.0f}s]", flush=True)
    print(f"总耗时 {time.time()-t_all:.0f}s")
