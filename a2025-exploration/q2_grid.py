"""Q2 解空间粗扫描: FY1 单弹对 M1, 决策变量 = (heading, v_uav, t_drop, dt_boom).

先理解目标函数地形, 再决定优化策略。
"""
import numpy as np
import itertools
import smoke_model as sm

FY1_0 = sm.UAVS['FY1']
M1_0 = sm.MISSILES['M1']
T_ARR = sm.missile_arrival_time(M1_0)


def obj(heading, v, t_drop, dt_boom, n_theta=48, n_z=4, n_rad=4, dt_scan=0.01):
    """单弹对 M1 的遮蔽时长 (体判据). 非法参数返回负值(惩罚)."""
    # 物理约束: 起爆点 z >= 0 (弹不落地前起爆); 投放时刻在无人机飞行期间
    P_b = sm.boom_point(FY1_0, heading, v, t_drop, dt_boom)
    if P_b[2] < 0.0:
        return -1.0
    if t_drop < 0.0 or t_drop > T_ARR:
        return -1.0
    if not (sm.V_UAV_MIN <= v <= sm.V_UAV_MAX):
        return -1.0
    intv, total = sm.single_shell_shadow(FY1_0, heading, v, t_drop, dt_boom, M1_0,
                                         n_theta=n_theta, n_z=n_z, n_rad=n_rad,
                                         dt_scan=dt_scan)
    return total


def grid_scan():
    """粗网格: heading 离散 24, v 离散 3, t_drop 离散 30, dt_boom 离散 20.
    先固定扫描看单变量影响."""
    # 先扫描 heading x dt_boom (固定 v=120, t_drop 取若干)
    print("=" * 72)
    print("扫描 1: heading x dt_boom (固定 v=120 m/s)")
    print("=" * 72)
    best = []
    for t_drop in [1.0, 3.0, 5.0]:
        row = []
        for heading in range(0, 360, 15):
            for dt_boom in np.arange(1.0, 11.0, 1.0):
                val = obj(heading, 120.0, t_drop, dt_boom)
                row.append((val, heading, dt_boom, t_drop))
        row.sort(reverse=True)
        best.append((t_drop, row[:3]))
    for t_drop, top in best:
        print(f"  t_drop={t_drop:.1f}s  最优3: "
              + " | ".join(f"T={v:.3f}s h={h}° db={db:.1f}s" for v, h, db, _ in top))

    print()
    print("=" * 72)
    print("扫描 2: t_drop x dt_boom (固定 heading=180°, v=120)")
    print("=" * 72)
    row = []
    for t_drop in np.arange(0.0, 12.0, 0.5):
        for dt_boom in np.arange(0.5, 12.0, 0.5):
            val = obj(180.0, 120.0, t_drop, dt_boom)
            row.append((val, t_drop, dt_boom))
    row.sort(reverse=True)
    print("  最优5:")
    for val, td, db in row[:5]:
        print(f"    T={val:.3f}s  t_drop={td:.1f}s  dt_boom={db:.1f}s")

    print()
    print("=" * 72)
    print("扫描 3: v x heading (固定 t_drop, dt_boom = 扫描2最优)")
    print("=" * 72)
    td0, db0 = row[0][1], row[0][2]
    row3 = []
    for heading in range(0, 360, 10):
        for v in np.arange(70, 141, 5):
            val = obj(heading, v, td0, db0)
            row3.append((val, heading, v))
    row3.sort(reverse=True)
    print(f"  固定 t_drop={td0:.1f}s dt_boom={db0:.1f}s 最优5:")
    for val, h, v in row3[:5]:
        print(f"    T={val:.3f}s  heading={h}°  v={v:.0f} m/s")


if __name__ == "__main__":
    grid_scan()
