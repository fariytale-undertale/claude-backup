"""Q2: FY1 单枚烟幕干扰弹对 M1, 优化 (航向, 速度, 投放时刻, 引信延迟)
使目标整体遮蔽时长最大化。DE 多轮 + 结果精化 + 局部敏感度。
"""
import json
import numpy as np
from scipy.optimize import differential_evolution
import smoke_model as sm

FY1_0 = sm.UAVS['FY1']
M1_0 = sm.MISSILES['M1']
T_ARR = sm.missile_arrival_time(M1_0)

# 决策变量边界: [heading(0~360), v(70~140), t_drop(0~40), dt_boom(0.5~19)]
BOUNDS = [(0.0, 359.999), (70.0, 140.0), (0.0, 40.0), (0.5, 19.0)]

# 优化用粗采样(快速), 精化用细采样
S_FAST = sm.target_samples(32, 3, 3)
S_FINE = sm.target_samples(96, 5, 6)


def shadow_approx(P_b, t_b, m0, S, R, dt_scan):
    """近似遮蔽时长 (优化用, 无边界精化)"""
    t_scan = np.arange(0.0, t_b + sm.T_VALID + dt_scan, dt_scan)
    M = sm.missile_pos(m0, t_scan)
    C = sm.cloud_center(P_b, t_b, t_scan)
    valid = (t_scan >= t_b - 1e-9) & (t_scan <= t_b + sm.T_VALID + 1e-9)
    mask = sm.shadowing_bool(M, C, S, R).all(axis=1) & valid
    return float(mask.sum()) * dt_scan


def obj(x, dt_scan=0.05, S=S_FAST):
    """最小化 -遮蔽时长; 非法参数给大惩罚"""
    heading, v, t_drop, dt_boom = x
    if not (sm.V_UAV_MIN - 1e-6 <= v <= sm.V_UAV_MAX + 1e-6):
        return 1e6
    if not (0.0 <= t_drop <= T_ARR):
        return 1e6
    if dt_boom <= 0.0:
        return 1e6
    P_b = sm.boom_point(FY1_0, heading, v, t_drop, dt_boom)
    if P_b[2] < 0.0:                      # 弹落地前起爆 (z>=0)
        return 1e6
    t_b = t_drop + dt_boom
    return -shadow_approx(P_b, t_b, M1_0, S, sm.R_CLOUD, dt_scan)


def run():
    results = []
    for seed in range(3):
        res = differential_evolution(
            obj, BOUNDS, seed=seed, maxiter=200, popsize=20,
            tol=1e-5, polish=False, workers=1,
            updating='immediate', recombination=0.7, mutation=(0.5, 1.0))
        results.append((res.fun, res.x))
        print(f"seed={seed}: T={-res.fun:.3f}s  x=({res.x[0]:.1f}°, {res.x[1]:.1f}m/s, "
              f"t_drop={res.x[2]:.2f}s, dt_boom={res.x[3]:.2f}s)")

    # 取全局最优, 用细精度精化 (局部优化 refine)
    best = min(results, key=lambda r: r[0])
    x0 = best[1]
    # 局部精化: 细步长 + 重新 DE 微调
    res2 = differential_evolution(
        lambda x: obj(x, dt_scan=0.005, S=S_FINE), BOUNDS, seed=42,
        maxiter=80, popsize=12, tol=1e-6, polish=False, workers=1,
        x0=[x0], updating='immediate')
    heading, v, t_drop, dt_boom = res2.x
    P_d = sm.drop_point(FY1_0, heading, v, t_drop)
    P_b = sm.boom_point(FY1_0, heading, v, t_drop, dt_boom)
    t_b = t_drop + dt_boom

    # 精确遮蔽区间 (细扫描 + 二分)
    intv, total = sm.single_shell_shadow(
        FY1_0, heading, v, t_drop, dt_boom, M1_0,
        n_theta=96, n_z=5, n_rad=6, dt_scan=0.002)

    print()
    print("=" * 72)
    print("Q2 最优解 (精化)")
    print("=" * 72)
    print(f"  航向      heading  = {heading:.2f}°  (x轴正向逆时针)")
    print(f"  速度      v        = {v:.2f} m/s")
    print(f"  投放时刻  t_drop   = {t_drop:.3f} s")
    print(f"  引信延迟  dt_boom  = {dt_boom:.3f} s")
    print(f"  起爆时刻  t_b      = {t_b:.3f} s")
    print(f"  投放点    P_d      = ({P_d[0]:.2f}, {P_d[1]:.2f}, {P_d[2]:.2f})")
    print(f"  起爆点    P_b      = ({P_b[0]:.2f}, {P_b[1]:.2f}, {P_b[2]:.2f})")
    print(f"  遮蔽区间  = {[(f'{a:.3f}', f'{b:.3f}') for a, b in intv]}")
    print(f"  最大遮蔽时长 = {total:.3f} s")

    # 局部敏感度: 对最优解每个参数单点扰动
    print()
    print("=" * 72)
    print("局部敏感度 (最优解单参数扰动, 粗精度)")
    print("=" * 72)
    base = -res2.fun
    x0 = res2.x
    for i, name in enumerate(["heading", "v", "t_drop", "dt_boom"]):
        for delta in [-1.0, -0.2, 0.2, 1.0]:
            dx = x0.copy()
            if name == "heading":
                dx[i] = (x0[i] + delta * 5.0) % 360.0
            elif name in ("v",):
                dx[i] = x0[i] + delta * 5.0
            elif name == "t_drop":
                dx[i] = x0[i] + delta * 0.5
            else:
                dx[i] = x0[i] + delta * 0.5
            val = -obj(dx, dt_scan=0.05, S=S_FINE)
            print(f"  {name:7s} {dx[i]:+10.2f} -> T={val:.3f}s  (Δ={val-base:+.3f})")

    # 保存结果
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
    run()
