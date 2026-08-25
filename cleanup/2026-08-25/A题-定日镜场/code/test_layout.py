# -*- coding: utf-8 -*-
"""环形布局测试 + G-Cost 性能预估（目标函数单次评估耗时）"""
import numpy as np
import time
from layout import ring_layout, check_constraints
from common import solar_position, TOWER_H, ETA_REF, mirror_normal, eta_cos, eta_at
from shadow import compute_eta_sb
from truncation import compute_eta_trunc


def fast_objective(C, tower_xy, w, h, sample_times, K_sb=6, K_tr=6):
    """代表性时点面积加权平均光学效率（快速）。返回 (η_avg, E_field_mean_kW)"""
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    eta_sum = 0.0
    e_sum = 0.0
    for D, ST in sample_times:
        s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
        from common import dni
        d = dni(alpha)
        n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
        cos = eta_cos(s, r_hat)
        at = eta_at(dHR)
        sb = compute_eta_sb(C, tower_xy, s, w, h, K=K_sb)
        tr = compute_eta_trunc(C, tower_xy, s, w, h, K=K_tr, Nr=2, Nphi=8)
        eta = cos * at * sb * tr * ETA_REF
        eta_sum += (eta * A).sum() / (A * N)
        e_sum += d * (eta * A).sum()
    n_t = len(sample_times)
    return eta_sum / n_t, e_sum / n_t


if __name__ == '__main__':
    # 环形布局参数
    for (w, h) in [(8, 8), (8, 6), (6, 6), (7, 7)]:
        for dr, da in [(1.6, 1.6), (1.4, 1.8)]:
            C = ring_layout((0, 0), w, h, 5.0, dr, da)
            ok, viol = check_constraints(C, w, h, (0, 0))
            print(f"w={w} h={h} dr={dr} da={da}: N={len(C)}, 约束{'满足' if ok else '违规:'+str(viol)}")

    # G-Cost 性能实测：fast 目标函数（代表性时点）
    print("\n=== G-Cost 性能预估 ===")
    w, h = 8, 8
    C = ring_layout((0, 0), w, h, 5.0, 1.6, 1.6)
    N = len(C)
    print(f"环形布局 (w={w},h={h},dr=1.6,da=1.6,塔中心): N={N}, 总面积={N*w*h:.0f} m2")
    # 代表性时点选择
    times_60 = [(D, ST) for D in [0, 31, 61, 92, 122, 153, 184, 214, 245, 275, -28, -59]
                for ST in [9, 10.5, 12, 13.5, 15]]
    times_12 = [(D, 12.0) for D in [-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]]
    times_20 = [(D, ST) for D in [-59, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
                for ST in [9.0, 12.0, 15.0]]
    for name, times, K in [("12时点K=6", times_12, 6), ("20时点K=6", times_20, 6),
                            ("60时点K=6", times_60, 6)]:
        t0 = time.time()
        eta, e = fast_objective(C, (0, 0), w, h, times, K_sb=K, K_tr=K)
        dt = time.time() - t0
        print(f"  {name}: 耗时 {dt:.1f}s, η_avg={eta:.4f}, E={e/1000:.1f}MW")
