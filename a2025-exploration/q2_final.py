"""Q2 最终优化: 物理热启动候选 + Nelder-Mead 局部精化 + 细精度确认.

1) 构造球心对准导弹视线轴的投放参数候选集
2) 对 top-K 候选局部优化 (Nelder-Mead)
3) 全局最优细精度精化 + 局部敏感度 + 结果落盘
"""
import json
import numpy as np
from scipy.optimize import minimize
import smoke_model as sm

FY1_0 = sm.UAVS['FY1']
M1_0 = sm.MISSILES['M1']
T_C = np.array([0.0, 200.0, 5.0])
BOUNDS = [(0.0, 360.0), (70.0, 140.0), (0.0, 40.0), (0.5, 19.0)]
S_FAST = sm.target_samples(40, 4, 4)
S_FINE = sm.target_samples(96, 5, 6)


def construct(t_c, L, k):
    M = sm.missile_pos(M1_0, t_c)
    u = (T_C - M) / np.linalg.norm(T_C - M)
    C_tc = M + L * u
    C_b = C_tc + np.array([0.0, 0.0, sm.V_SINK * k])
    if C_b[2] > FY1_0[2] - 1e-6:
        return None
    t_b = t_c - k
    if t_b <= 0.0:
        return None
    dt_boom = np.sqrt(2.0 * (FY1_0[2] - C_b[2]) / sm.G)
    t_drop = t_b - dt_boom
    if t_drop < 0.0:
        return None
    vx = (C_b[0] - FY1_0[0]) / t_b
    vy = (C_b[1] - FY1_0[1]) / t_b
    v = np.hypot(vx, vy)
    if not (sm.V_UAV_MIN - 1e-6 <= v <= sm.V_UAV_MAX + 1e-6):
        return None
    heading = np.degrees(np.arctan2(vy, vx)) % 360.0
    return np.array([heading, v, t_drop, dt_boom])


def shadow_approx(P_b, t_b, m0, S, R, dt_scan):
    t_scan = np.arange(0.0, t_b + sm.T_VALID + dt_scan, dt_scan)
    M = sm.missile_pos(m0, t_scan)
    C = sm.cloud_center(P_b, t_b, t_scan)
    valid = (t_scan >= t_b - 1e-9) & (t_scan <= t_b + sm.T_VALID + 1e-9)
    mask = sm.shadowing_bool(M, C, S, R).all(axis=1) & valid
    return float(mask.sum()) * dt_scan


def obj(x, dt_scan=0.02, S=S_FAST):
    heading, v, t_drop, dt_boom = x
    if not (sm.V_UAV_MIN - 1e-6 <= v <= sm.V_UAV_MAX + 1e-6) or dt_boom <= 0 or t_drop < 0:
        return 1e6
    P_b = sm.boom_point(FY1_0, heading, v, t_drop, dt_boom)
    if P_b[2] < 0.0:
        return 1e6
    return -shadow_approx(P_b, t_drop + dt_boom, M1_0, S, sm.R_CLOUD, dt_scan)


def gen_candidates():
    cands = []
    for t_c in np.arange(5.0, 35.0, 1.5):
        for L in [150.0, 250.0, 400.0, 550.0]:
            for k in [2.5, 4.0, 6.0]:
                p = construct(t_c, L, k)
                if p is None:
                    continue
                val = -obj(p, dt_scan=0.05)
                if val > 0.3:
                    cands.append((val, p))
    cands.sort(key=lambda r: r[0], reverse=True)
    return cands


def main():
    cands = gen_candidates()
    print(f"物理热启动候选数: {len(cands)} (遮蔽时长>0.3s)")
    top = cands[:8]

    refined = []
    for val, x0 in top:
        res = minimize(obj, x0, method="Nelder-Mead",
                       bounds=BOUNDS,
                       options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4})
        refined.append((-res.fun, res.x))
        print(f"  start T={val:.3f}s -> refined T={-res.fun:.3f}s  x=({res.x[0]:.1f}°, "
              f"{res.x[1]:.1f}m/s, {res.x[2]:.2f}s, {res.x[3]:.2f}s)")

    refined.sort(key=lambda r: r[0], reverse=True)
    best_val, best_x = refined[0]
    heading, v, t_drop, dt_boom = best_x

    # 细精度确认
    P_d = sm.drop_point(FY1_0, heading, v, t_drop)
    P_b = sm.boom_point(FY1_0, heading, v, t_drop, dt_boom)
    t_b = t_drop + dt_boom
    intv, total = sm.single_shell_shadow(
        FY1_0, heading, v, t_drop, dt_boom, M1_0,
        n_theta=96, n_z=5, n_rad=6, dt_scan=0.002)

    print()
    print("=" * 72)
    print("Q2 最优解 (细精度)")
    print("=" * 72)
    print(f"  航向      heading  = {heading:.3f}°   (x轴正向逆时针)")
    print(f"  速度      v        = {v:.3f} m/s")
    print(f"  投放时刻  t_drop   = {t_drop:.3f} s")
    print(f"  引信延迟  dt_boom  = {dt_boom:.3f} s")
    print(f"  起爆时刻  t_b      = {t_b:.3f} s")
    print(f"  投放点    P_d      = ({P_d[0]:.2f}, {P_d[1]:.2f}, {P_d[2]:.2f})")
    print(f"  起爆点    P_b      = ({P_b[0]:.2f}, {P_b[1]:.2f}, {P_b[2]:.2f})")
    print(f"  遮蔽区间  = {[(f'{a:.3f}', f'{b:.3f}') for a, b in intv]}")
    print(f"  最大遮蔽时长 = {total:.3f} s")

    # 采样收敛性复核 (最优解)
    print()
    print("采样收敛性复核 (最优解):")
    for nth, nz, nrad in [(40, 4, 4), (96, 5, 6), (144, 6, 8)]:
        _, tot = sm.single_shell_shadow(FY1_0, heading, v, t_drop, dt_boom, M1_0,
                                        n_theta=nth, n_z=nz, n_rad=nrad, dt_scan=0.002)
        print(f"  theta={nth} z={nz} rad={nrad}: T={tot:.4f} s")

    # 局部敏感度
    print()
    print("局部敏感度 (单参数微扰, 细精度):")
    base = total
    for i, name, step in [(0, "heading", 2.0), (1, "v", 5.0),
                          (2, "t_drop", 0.5), (3, "dt_boom", 0.5)]:
        for sgn in [-1, 1]:
            x = best_x.copy()
            x[i] = (best_x[i] + sgn * step) if name != "heading" else \
                (best_x[i] + sgn * step) % 360.0
            P_b2 = sm.boom_point(FY1_0, x[0], x[1], x[2], x[3])
            if P_b2[2] >= 0:
                _, tot = sm.single_shell_shadow(FY1_0, x[0], x[1], x[2], x[3], M1_0,
                                                n_theta=96, n_z=5, n_rad=6, dt_scan=0.002)
            else:
                tot = 0.0
            print(f"  {name:7s} {sgn:+.0f}*{step:.1f} -> T={tot:.3f}s (Δ={tot-base:+.3f})")

    out = {
        "Q": 2, "missile": "M1", "uav": "FY1",
        "heading_deg": float(heading), "v_mps": float(v),
        "t_drop": float(t_drop), "dt_boom": float(dt_boom), "t_b": float(t_b),
        "drop_point": P_d.tolist(), "boom_point": P_b.tolist(),
        "shadow_intervals": intv, "shadow_time": float(total),
    }
    with open("q2_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已保存 q2_result.json")


if __name__ == "__main__":
    main()
