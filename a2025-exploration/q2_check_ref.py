"""Q2 判据口径诊断: 对比 参考答案(4.62s, 4.70s) 与不同遮蔽判据的时长.

判据候选:
  体判据    : 目标圆柱所有采样点视线被挡 (全部遮蔽)
  点判据    : 单一参考点视线被挡 (中心/顶面/底面/轮廓)
  部分遮蔽  : 目标上任意一点视线被挡 (exist 判据, 最宽松)

对每个候选判据, 在其最优投放参数下计算遮蔽时长.
"""
import numpy as np
from scipy.optimize import minimize
import smoke_model as sm

FY1_0 = sm.UAVS['FY1']
M1_0 = sm.MISSILES['M1']
T_C = np.array([0.0, 200.0, 5.0])
BOUNDS = [(0.0, 360.0), (70.0, 140.0), (0.0, 40.0), (0.5, 19.0)]

# 参考答案
REF = [4.62, 4.70]


def shadow_for_points(P_b, t_b, S, R, dt_scan, agg):
    """agg='all': 所有点被挡; agg='any': 任一点被挡; agg='center': 单点."""
    t_scan = np.arange(0.0, t_b + sm.T_VALID + dt_scan, dt_scan)
    M = sm.missile_pos(M1_0, t_scan)
    C = sm.cloud_center(P_b, t_b, t_scan)
    mask = sm.shadowing_bool(M, C, S, R)                 # (n_t, n_s)
    valid = (t_scan >= t_b - 1e-9) & (t_scan <= t_b + sm.T_VALID + 1e-9)
    if agg == 'all':
        row = mask.all(axis=1)
    elif agg == 'any':
        row = mask.any(axis=1)
    else:
        row = mask[:, 0]
    return float((row & valid).sum()) * dt_scan


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


def make_obj(S, agg, dt):
    def obj(x):
        heading, v, t_drop, dt_boom = x
        if not (sm.V_UAV_MIN - 1e-6 <= v <= sm.V_UAV_MAX + 1e-6) or dt_boom <= 0 or t_drop < 0:
            return 1e6
        P_b = sm.boom_point(FY1_0, heading, v, t_drop, dt_boom)
        if P_b[2] < 0.0:
            return 1e6
        return -shadow_for_points(P_b, t_drop + dt_boom, S, sm.R_CLOUD, dt, agg)
    return obj


def optimize_criterion(S, agg, dt=0.05, label=""):
    """物理热启动 + NM 求该判据的最优解."""
    cands = []
    for t_c in np.arange(5.0, 15.0, 1.5):
        for L in [250.0, 400.0, 550.0]:
            for k in [3.0, 4.0, 5.0]:
                p = construct(t_c, L, k)
                if p is None:
                    continue
                obj = make_obj(S, agg, 0.1)
                val = -obj(p)
                if val > 0.3:
                    cands.append((val, p))
    cands.sort(key=lambda r: r[0], reverse=True)
    obj = make_obj(S, agg, dt)
    best = None
    for val, x0 in cands[:4]:
        res = minimize(obj, x0, method="Nelder-Mead", bounds=BOUNDS,
                       options={"maxiter": 120, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or -res.fun > best[0]:
            best = (-res.fun, res.x)
    return best


def main():
    print("参考答案: 4.62 s, 4.70 s")
    print()

    # 体判据最优参数 (来自 q2_result.json)
    import json
    r = json.load(open("q2_result.json", encoding="utf-8"))
    x_body = np.array([r["heading_deg"], r["v_mps"], r["t_drop"], r["dt_boom"]])
    print(f"体判据最优参数: heading={x_body[0]:.1f}° v={x_body[1]:.1f} t_drop={x_body[2]:.2f} dt_boom={x_body[3]:.2f}")

    S_center = np.array([[0.0, 200.0, 5.0]])
    S_top = np.array([[0.0, 200.0, 10.0]])
    S_bottom = np.array([[0.0, 200.0, 0.0]])
    S_contour = np.array([[7.0, 200.0, 10.0], [0.0, 207.0, 10.0], [7.0, 200.0, 0.0], [-7.0, 200.0, 10.0]])
    S_full = sm.target_samples(48, 4, 4)

    # 用体判据最优参数, 换判据口径
    P_b = sm.boom_point(FY1_0, *x_body)
    t_b = x_body[2] + x_body[3]
    print()
    print("在 体判据最优参数 下, 各判据口径的时长:")
    print(f"  体判据(all 采样点) : {shadow_for_points(P_b, t_b, S_full, 10.0, 0.01, 'all'):.3f} s")
    print(f"  中心点(0,200,5)   : {shadow_for_points(P_b, t_b, S_center, 10.0, 0.01, 'any'):.3f} s")
    print(f"  顶面中心(0,200,10): {shadow_for_points(P_b, t_b, S_top, 10.0, 0.01, 'any'):.3f} s")
    print(f"  底面中心(0,200,0) : {shadow_for_points(P_b, t_b, S_bottom, 10.0, 0.01, 'any'):.3f} s")
    print(f"  轮廓4点(any)       : {shadow_for_points(P_b, t_b, S_contour, 10.0, 0.01, 'any'):.3f} s")
    print(f"  任一点被挡(exist)  : {shadow_for_points(P_b, t_b, S_full, 10.0, 0.01, 'any'):.3f} s")

    print()
    print("各判据口径独立优化的最优时长:")
    best_body = optimize_criterion(S_full, 'all', 0.05, "体")
    print(f"  体判据独立最优   : {best_body[0]:.3f} s   x=({best_body[1][0]:.1f}°, {best_body[1][1]:.1f}, {best_body[1][2]:.2f}, {best_body[1][3]:.2f})")
    best_ctr = optimize_criterion(S_center, 'any', 0.05)
    print(f"  中心点判据最优   : {best_ctr[0]:.3f} s   x=({best_ctr[1][0]:.1f}°, {best_ctr[1][1]:.1f}, {best_ctr[1][2]:.2f}, {best_ctr[1][3]:.2f})")
    best_top = optimize_criterion(S_top, 'any', 0.05)
    print(f"  顶面中心判据最优 : {best_top[0]:.3f} s   x=({best_top[1][0]:.1f}°, {best_top[1][1]:.1f}, {best_top[1][2]:.2f}, {best_top[1][3]:.2f})")
    best_exist = optimize_criterion(S_full, 'any', 0.05)
    print(f"  任一点判据最优   : {best_exist[0]:.3f} s   x=({best_exist[1][0]:.1f}°, {best_exist[1][1]:.1f}, {best_exist[1][2]:.2f}, {best_exist[1][3]:.2f})")


if __name__ == "__main__":
    main()
