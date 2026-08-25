# -*- coding: utf-8 -*-
"""阴影遮挡：区分性测试（点判据 vs 体判据）+ 附件镜场多时点 sanity check"""
import numpy as np
import pandas as pd
from common import solar_position
from shadow import compute_eta_sb, eta_sb_point


def test_artificial_occlusion():
    """人工构造遮挡：A 在塔北300m，B 在 A 南12m（更靠塔），太阳正南。
    解析预测：A 中心反射光被 B 挡 → 点判据 ηsb=0；体判据部分遮挡 ηsb∈(0,1)。
    两判据数值不同 → 跨抽象层区分。"""
    tower_xy = (0.0, 0.0)
    s = np.array([0.0, -0.5, 0.866])   # 正南，高度角60°
    s = s / np.linalg.norm(s)
    A = np.array([0.0, 300.0, 4.0])    # 塔正北300m
    B = np.array([0.0, 288.0, 4.0])    # A 南侧12m，A 与塔之间
    C = np.array([A, B])
    eta = compute_eta_sb(C, tower_xy, s, 6.0, 6.0, K=16)
    ep = eta_sb_point(C, tower_xy, s, 6.0, 6.0)
    print(f"[人工遮挡] A(塔北300m): 体判据 ηsb={eta[0]:.4f}, 点判据={ep[0]:.2f}")
    print(f"            B(塔北288m): 体判据 ηsb={eta[1]:.4f}, 点判据={ep[1]:.2f}")
    occ_ok = eta[0] < 1.0 - 1e-6          # 体判据检测到遮挡
    diff_ok = abs(eta[0] - ep[0]) > 1e-6  # 点/体判据不同答案（跨抽象层区分）
    return occ_ok and diff_ok


def test_artificial_shadow():
    """人工构造阴影：太阳在东方低垂，东镜 B 的阴影落西镜 A。
    构造使 A 中心恰好不在 B 投影内（点判据 ηsb=1），但 A 边缘在（体判据 ηsb<1）。"""
    tower_xy = (0.0, 0.0)
    # 太阳正东低垂，塔在原点，镜 A/B 在塔北偏西/偏东
    s = np.array([1.0, 0.0, 0.35])
    s = s / np.linalg.norm(s)
    # 两镜东西排列，间距8m（> 半宽和6m → 中心投影不重叠；< 对角线和 → 边缘重叠）
    A = np.array([-4.0, 150.0, 4.0])   # 西镜
    B = np.array([4.0, 150.0, 4.0])    # 东镜（更靠近太阳）
    C = np.array([A, B])
    eta = compute_eta_sb(C, tower_xy, s, 6.0, 6.0, K=16)
    ep = eta_sb_point(C, tower_xy, s, 6.0, 6.0)
    print(f"[人工阴影] A(西镜): 体判据 ηsb={eta[0]:.4f}, 点判据={ep[0]:.2f}")
    print(f"            B(东镜): 体判据 ηsb={eta[1]:.4f}, 点判据={ep[1]:.2f}")
    # 期望：A 被 B 阴影覆盖（ηsb_A < 1），且两判据不同
    occ_ok = eta[0] < 1.0 - 1e-6
    diff_ok = abs(eta[0] - ep[0]) > 1e-6
    return occ_ok and diff_ok


def test_field_multitimes():
    """附件镜场多时点：全年应检测到遮挡（上午/下午太阳低）"""
    df = pd.read_excel(r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/附件.xlsx')
    x, y = df['x坐标 (m)'].values, df['y坐标 (m)'].values
    C = np.column_stack([x, y, np.full(len(x), 4.0)])
    cases = [(0, 9.0, '1月21日9时'), (0, 12.0, '1月21日正午'),
             (92, 12.0, '6月21日正午'), (184, 12.0, '9月21日正午'),
             (275, 9.0, '12月21日9时'), (275, 15.0, '12月21日15时')]
    print("\n[附件镜场多时点]")
    n_occ_total = 0
    for D, ST, name in cases:
        s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
        eta = compute_eta_sb(C, (0, 0), s, 6.0, 6.0, K=8)
        n_occ = int(np.sum(eta < 1 - 1e-6))
        n_occ_total += n_occ
        print(f"  {name}: αs={np.degrees(alpha):.1f}°, ηsb mean={eta.mean():.5f}, "
              f"ηsb min={eta.min():.5f}, 遮挡镜数={n_occ}")
    return n_occ_total > 0


if __name__ == '__main__':
    r1 = test_artificial_occlusion()
    r2 = test_artificial_shadow()
    r3 = test_field_multitimes()
    print("\n结论:")
    print(f"  人工遮挡区分性测试: {'PASS' if r1 else 'FAIL'}")
    print(f"  人工阴影区分性测试: {'PASS' if r2 else 'FAIL'}")
    print(f"  附件镜场多时点:     {'PASS' if r3 else 'FAIL'}")
    assert r1 and r2 and r3, "阴影遮挡体判据区分性测试未通过"
