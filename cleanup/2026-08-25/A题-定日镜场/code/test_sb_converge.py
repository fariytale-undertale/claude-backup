# -*- coding: utf-8 -*-
"""ηsb 网格分辨率收敛性测试 + 全年60时点 ηsb 均值"""
import numpy as np
import pandas as pd
import time
from common import solar_position, D_MONTHS, TIMES
from shadow import compute_eta_sb


def main():
    df = pd.read_excel(r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/附件.xlsx')
    x, y = df['x坐标 (m)'].values, df['y坐标 (m)'].values
    C = np.column_stack([x, y, np.full(len(x), 4.0)])

    # K 收敛性：冬季上午（遮挡最严重时点）
    print("=== K 收敛性（12月21日9时，遮挡最严重）===")
    s, alpha, _, _, _ = solar_position(275.0, 9.0, return_all=True)
    prev = None
    for K in [4, 6, 8, 12, 16, 24]:
        t0 = time.time()
        eta = compute_eta_sb(C, (0, 0), s, 6.0, 6.0, K=K)
        dt = time.time() - t0
        m = eta.mean()
        diff = '' if prev is None else f'  Δmean={abs(m-prev):.5f}'
        print(f"  K={K:2d}: mean={m:.5f} min={eta.min():.5f} 耗时{dt:.2f}s{diff}")
        prev = m

    # 全年 60 时点 ηsb 均值（K=8）
    print("\n=== 全年60时点 ηsb 均值（K=8）===")
    t0 = time.time()
    sb_all = np.zeros((12, 5, len(C)))
    for mi, D in enumerate(D_MONTHS):
        for ti, ST in enumerate(TIMES):
            s, _, _, _, _ = solar_position(D, ST, return_all=True)
            sb_all[mi, ti] = compute_eta_sb(C, (0, 0), s, 6.0, 6.0, K=8)
    dt = time.time() - t0
    annual = sb_all.mean()
    print(f"  全年 ηsb 均值 = {annual:.5f}  （耗时 {dt:.1f}s）")
    print(f"  各月平均: " + " ".join(f"{sb_all[i].mean():.4f}" for i in range(12)))
    # 保存供问题1使用
    np.save(r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/output/eta_sb_all.npy', sb_all)
    print("  已保存 output/eta_sb_all.npy")
    assert 0.85 < annual < 0.99, "ηsb 年均偏离合理区间[0.85,0.99]"


if __name__ == '__main__':
    main()
