# -*- coding: utf-8 -*-
"""问题2优化：径向渐变间距布局（内密外疏，模仿附件布局模式）+ K=4 验证附件复现。"""
import numpy as np, pandas as pd, time
from common import (solar_position, TOWER_H, ETA_REF, mirror_normal, eta_cos,
                    eta_at, D_MONTHS, TIMES)
from shadow import compute_eta_sb
from truncation import compute_eta_trunc
from p2_search import full_objective


def hex_layout_grad(tower_xy, w, h, h_inst, df_in, df_out, R_field=350.0,
                    R_clear=100.0):
    """径向渐变间距六角布局：df 随距塔半径从 df_in 线性增至 df_out。"""
    dmin_base = max(w, h) + 5.0
    xt, yt = tower_xy
    rlim = R_field - np.hypot(w, h) / 2.0
    centers = []
    k_max = int(np.ceil((R_field + np.hypot(xt, yt)) / (np.sqrt(3)/2*dmin_base*df_in))) + 1
    for k in range(-k_max, k_max + 1):
        y = yt + k * (np.sqrt(3) / 2.0) * dmin_base * df_in
        if abs(y) > rlim + 0.5 * dmin_base * df_out:
            continue
        # 该行内 df 按距塔半径近似
        r_mid = max(abs(y - yt), R_clear)
        t = min(max((r_mid - R_clear) / (R_field - R_clear), 0.0), 1.0)
        df = df_in + (df_out - df_in) * t
        dmin = dmin_base * df
        off = 0.5 * (k % 2) * dmin
        m_lo = int(np.floor((xt - rlim - off) / dmin))
        m_hi = int(np.ceil((xt + rlim - off) / dmin))
        for m in range(m_lo, m_hi + 1):
            x = xt + off + m * dmin
            if np.hypot(x, y) <= rlim and np.hypot(x - xt, y - yt) > R_clear + 1e-9:
                centers.append([x, y, h_inst])
    return np.array(centers)


if __name__ == '__main__':
    # 1) 附件布局 K=4 验证（应复现问题1 的 ~0.578）
    df = pd.read_excel(r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/附件.xlsx')
    x, y = df['x坐标 (m)'].values, df['y坐标 (m)'].values
    C = np.column_stack([x, y, np.full(len(x), 4.0)])
    t0 = time.time()
    eta_a, E_a = full_objective(C, (0, 0), 6, 6, K=4)
    print(f"附件 K=4 验证: η={eta_a:.5f} E={E_a:.2f}MW (问题1 K=12: 0.5778, 35.3MW) 耗时{time.time()-t0:.0f}s", flush=True)

    # 2) 渐变间距布局候选
    cands = [((0, 0), 7, 7, 1.0, 1.2), ((0, 0), 7, 7, 1.0, 1.3),
             ((0, 0), 7, 7, 1.05, 1.3), ((0, 0), 8, 8, 1.0, 1.2),
             ((0, 0), 7, 6, 1.0, 1.0), ((0, 0), 7, 7, 1.0, 1.15)]
    for tower, w, h, dfi, dfo in cands:
        h_inst = max(4.0, h / 2 + 1.0)
        Cg = hex_layout_grad(tower, w, h, h_inst, dfi, dfo)
        if len(Cg) < 100:
            continue
        t0 = time.time()
        eta, E = full_objective(Cg, tower, w, h, K=4)
        A = len(Cg) * w * h
        print(f"渐变 塔{tower} {w}x{h} dfi={dfi} dfo={dfo} N={len(Cg)} A={A:.0f} "
              f"η={eta:.5f} E={E:.2f}MW 耗时{time.time()-t0:.0f}s", flush=True)
