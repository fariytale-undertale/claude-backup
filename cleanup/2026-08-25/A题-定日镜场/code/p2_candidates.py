# -*- coding: utf-8 -*-
"""问题2候选布局 K=4 精确评估 + 60MW 删镜。"""
import numpy as np, time, json, os
from layout import hex_layout
from p2_search import full_objective
from common import (D_MONTHS, TIMES, TOWER_H, ETA_REF, solar_position, dni,
                    mirror_normal, eta_cos, eta_at)
from shadow import compute_eta_sb
from truncation import compute_eta_trunc

OUT = r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/output'
TIMES60 = [(D, ST) for D in D_MONTHS for ST in TIMES]


def per_mirror_contrib(C, tower_xy, w, h, K=4):
    """每镜年均功率贡献(kW)。"""
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    contrib = np.zeros(N)
    for D, ST in TIMES60:
        s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
        d = dni(alpha)
        n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
        cos = eta_cos(s, r_hat)
        at = eta_at(dHR)
        sb = compute_eta_sb(C, tower_xy, s, w, h, K=K)
        tr = compute_eta_trunc(C, tower_xy, s, w, h, K=K, Nr=2, Nphi=8)
        eta = cos * at * sb * tr * ETA_REF
        contrib += d * A * eta / len(TIMES60)
    return contrib


if __name__ == '__main__':
    candidates = [
        ((0, 0), 7, 7, 1.0), ((0, -100), 7, 7, 1.0), ((0, -120), 7, 7, 1.0),
        ((0, -100), 8, 8, 1.0), ((0, -140), 7, 7, 1.0),
        ((0, -100), 7, 7, 1.04),
    ]
    results = []
    for tower, w, h, df in candidates:
        h_inst = max(4.0, h / 2 + 1.0)
        C = hex_layout(tower, w, h, h_inst, df)
        if len(C) < 100:
            continue
        t0 = time.time()
        eta, E = full_objective(C, tower, w, h, K=4)
        A = len(C) * w * h
        r = dict(xt=tower[0], yt=tower[1], w=w, h=h, h_inst=h_inst, df=df,
                 N=len(C), A=A, eta=eta, E=E)
        print(f"候选 塔{tower} {w}x{h} df={df} N={len(C)} A={A:.0f} "
              f"η={eta:.5f} E={E:.2f}MW 耗时{time.time()-t0:.0f}s", flush=True)
        results.append(r)
    # 60MW 删镜
    print("\n=== 60MW 最小面积删镜 ===", flush=True)
    for r in results:
        if r['E'] >= 60.0:
            C = hex_layout((r['xt'], r['yt']), r['w'], r['h'], r['h_inst'], r['df'])
            contrib = per_mirror_contrib(C, (r['xt'], r['yt']), r['w'], r['h'], K=4)
            order = np.argsort(-contrib)
            cum = np.cumsum(contrib[order])
            n_keep = min(int(np.searchsorted(cum, 60000.0)) + 1, len(C))
            A_sel = n_keep * r['w'] * r['h']
            E_sel = cum[n_keep - 1]
            unit = E_sel / A_sel
            r['n_keep'] = n_keep
            r['A_sel'] = A_sel
            r['E_sel'] = E_sel
            r['unit_power'] = unit
            print(f"塔({r['xt']},{r['yt']}) {r['w']}x{r['h']} df={r['df']}: "
                  f"保留{n_keep}/{len(C)}镜 A_sel={A_sel:.0f} E_sel={E_sel:.0f}kW "
                  f"单位面积={unit:.4f} kW/m2", flush=True)
    with open(os.path.join(OUT, 'problem2_candidates.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print("\n完成", flush=True)
