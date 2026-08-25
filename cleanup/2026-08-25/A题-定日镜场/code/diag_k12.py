# -*- coding: utf-8 -*-
"""K=12 精判据：问题2/3 布局的真实效率与 60MW 可达性诊断。"""
import numpy as np, time
from layout import hex_layout
from p3_grad import hex_zone_grad
from common import solar_position, TOWER_H, ETA_REF, mirror_normal, D_MONTHS, TIMES
from shadow import compute_eta_sb
from truncation import compute_eta_trunc


def eval_k(C, w_arr, h_arr, tower_xy, K):
    N = len(C)
    A = w_arr * h_arr
    Atot = A.sum()
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    eta_sum = 0.0
    e_sum = 0.0
    for D in D_MONTHS:
        for ST in TIMES:
            s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
            d = 1.366 * (0.3498 + 0.5784 * np.exp(-0.2757 / np.maximum(np.sin(alpha), 1e-6)))
            n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
            cos = np.sqrt((1.0 + s @ r_hat.T) / 2.0)
            at = 0.99321 - 0.0001176 * dHR + 1.97e-8 * dHR ** 2
            sb = compute_eta_sb(C, tower_xy, s, w_arr, h_arr, K=K)
            tr = compute_eta_trunc(C, tower_xy, s, w_arr, h_arr, K=K, Nr=2, Nphi=8)
            eta = cos * at * sb * tr * ETA_REF
            eta_sum += (eta * A).sum() / Atot
            e_sum += d * (eta * A).sum()
    n60 = 60.0
    return eta_sum / n60, e_sum / n60 / 1000.0


if __name__ == '__main__':
    tower = (0.0, 0.0)
    # 问题2：7x7 均匀六角 df=1.0
    C2 = hex_layout(tower, 7, 7, 4.5, 1.0)
    w2 = np.full(len(C2), 7.0); h2 = np.full(len(C2), 7.0)
    t0 = time.time()
    eta2, E2 = eval_k(C2, w2, h2, tower, 12)
    print(f"问题2 布局 7x7六角 N={len(C2)}: K=12 η={eta2:.5f} E={E2:.2f}MW "
          f"(K=4: η=0.4665 E=61.2) 耗时{time.time()-t0:.0f}s", flush=True)
    # 问题3：分区 7x6/7x7
    C3, w3, h3 = hex_zone_grad(tower, 120.0, 7.0, 6.0, 7.0, 7.0)
    t0 = time.time()
    eta3, E3 = eval_k(C3, w3, h3, tower, 12)
    print(f"问题3 布局 分区 N={len(C3)}: K=12 η={eta3:.5f} E={E3:.2f}MW "
          f"(K=4: η=0.4675 E=60.8) 耗时{time.time()-t0:.0f}s", flush=True)
