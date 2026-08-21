"""Q2 物理热启动: 构造 '球心在导弹视线轴上' 的投放参数候选, 验证物理判断.

构造: 遮蔽窗口中心 t_c, 球心在该时刻位于视线轴 (M(t_c)->目标中心) 上、距导弹 L.
      起爆时刻 t_b = t_c - k, 球心已下沉 3k.
      反推投放参数 (heading, v, t_drop, dt_boom), 检查无人机可达性, 算遮蔽时长.
"""
import numpy as np
import smoke_model as sm

FY1_0 = sm.UAVS['FY1']
M1_0 = sm.MISSILES['M1']
T_C = np.array([0.0, 200.0, 5.0])     # 目标圆柱中心(近似参考)
DT_SCAN = 0.01


def construct(t_c, L, k):
    """球心在遮蔽中心时刻对准视线轴, 反推投放参数. 返回 (heading,v,t_drop,dt_boom) 或 None."""
    M = sm.missile_pos(M1_0, t_c)
    u = (T_C - M) / np.linalg.norm(T_C - M)
    C_tc = M + L * u                      # 遮蔽中心时刻球心位置 (视线轴上, 距导弹 L)
    C_b = C_tc + np.array([0.0, 0.0, sm.V_SINK * k])   # 回溯下沉 -> 起爆点
    if C_b[2] > FY1_0[2] - 1e-6:
        return None                        # 弹不能上升
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
    return (heading, v, t_drop, dt_boom)


def main():
    print("物理热启动: 球心在视线轴上 (t_c, L, k) 扫描")
    print(f"{'t_c':>5} {'L':>5} {'k':>4} | {'heading':>7} {'v':>6} {'t_drop':>6} {'dt_boom':>6} | {'T_shadow':>8}")
    cands = []
    for t_c in np.arange(6.0, 35.0, 2.0):
        for L in [100.0, 200.0, 350.0, 500.0]:
            for k in [3.0, 5.0, 8.0]:
                p = construct(t_c, L, k)
                if p is None:
                    continue
                heading, v, t_drop, dt_boom = p
                intv, total = sm.single_shell_shadow(
                    FY1_0, heading, v, t_drop, dt_boom, M1_0,
                    n_theta=48, n_z=4, n_rad=4, dt_scan=DT_SCAN)
                cands.append((total, heading, v, t_drop, dt_boom, t_c, L, k))
                if total > 0.5:
                    print(f"{t_c:5.1f} {L:5.0f} {k:4.0f} | "
                          f"{heading:7.2f} {v:6.1f} {t_drop:6.2f} {dt_boom:6.2f} | {total:8.3f}")

    cands.sort(reverse=True)
    print()
    print("最优候选 (遮蔽时长前10):")
    for total, h, v, td, db, tc, L, k in cands[:10]:
        print(f"  T={total:.3f}s  heading={h:.1f}° v={v:.1f} t_drop={td:.2f} dt_boom={db:.2f} "
              f"(t_c={tc:.1f} L={L:.0f} k={k:.0f})")


if __name__ == "__main__":
    main()
