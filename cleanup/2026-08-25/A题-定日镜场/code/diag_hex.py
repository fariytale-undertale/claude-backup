# -*- coding: utf-8 -*-
"""六角布局效率评估（K=2 快速，60时点）"""
import numpy as np, time
from layout import hex_layout
from common import (solar_position, TOWER_H, ETA_REF, mirror_normal, eta_cos,
                    eta_at, D_MONTHS, TIMES)
from shadow import compute_eta_sb
from truncation import compute_eta_trunc


def eval_layout(C, tower_xy, w, h, K=2):
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    c = at = sb = tr = 0.0
    for D in D_MONTHS:
        for ST in TIMES:
            s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
            n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
            c += (eta_cos(s, r_hat) * A).sum() / (A * N)
            at += (eta_at(dHR) * A).sum() / (A * N)
            sb += (compute_eta_sb(C, tower_xy, s, w, h, K=K) * A).sum() / (A * N)
            tr += (compute_eta_trunc(C, tower_xy, s, w, h, K=K, Nr=2, Nphi=8)
                   * A).sum() / (A * N)
    n60 = 60.0
    return c / n60, at / n60, sb / n60, tr / n60


if __name__ == '__main__':
    cases = [
        ((0, 0), 6, 6, 1.0), ((0, -60), 6, 6, 1.0), ((0, -100), 6, 6, 1.0),
        ((0, 0), 7, 7, 1.0), ((0, -60), 7, 7, 1.0),
        ((0, 0), 8, 8, 1.0), ((0, 0), 6, 6, 1.15), ((0, 0), 6, 6, 1.3),
        ((0, -80), 6, 6, 1.0), ((0, -120), 6, 6, 1.0),
    ]
    for tower, w, h, df in cases:
        h_inst = max(4.0, h / 2 + 1.0)
        C = hex_layout(tower, w, h, h_inst, df)
        if len(C) < 100:
            continue
        t0 = time.time()
        c, at, sb, tr = eval_layout(C, tower, w, h, K=2)
        eta = c * at * sb * tr * ETA_REF
        A = len(C) * w * h
        E = 0.968 * eta * A / 1000.0
        print(f"六角 塔{tower} {w}x{h} df={df} N={len(C):4d} A={A:7.0f} "
              f"cos={c:.4f} at={at:.4f} sb={sb:.4f} tr={tr:.4f} "
              f"η={eta:.4f} E≈{E:.1f}MW [{time.time()-t0:.0f}s]", flush=True)
