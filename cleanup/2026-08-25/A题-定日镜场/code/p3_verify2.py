# -*- coding: utf-8 -*-
"""问题3 备选：内7x6外7x7（差异小、面积接近7x7布局）。K=4 验证。"""
import numpy as np, time, json, os
from p3_grad import hex_zone_grad
from problem3 import eval_zone

OUT = r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/output'

if __name__ == '__main__':
    tower = (0.0, 0.0)
    cands = [(7, 6, 7, 7, 120), (7, 6, 7, 7, 150), (7, 6, 7, 7, 200),
             (8, 6, 8, 8, 150)]
    results = []
    for (w_in, h_in, w_out, h_out, R1) in cands:
        C, w_arr, h_arr = hex_zone_grad(tower, R1, w_in, h_in, w_out, h_out)
        t0 = time.time()
        eta, E = eval_zone(C, w_arr, h_arr, tower, K=4)
        A_tot = (w_arr * h_arr).sum()
        results.append(dict(win=w_in, hin=h_in, wout=w_out, hout=h_out, R1=R1,
                            N=len(C), A=A_tot, eta=eta, E=E))
        print(f"内{w_in}x{h_in}外{w_out}x{h_out} R1={R1}: N={len(C)} A={A_tot:.0f} "
              f"η={eta:.5f} E={E:.2f}MW 耗时{time.time()-t0:.0f}s", flush=True)
    feasible = [r for r in results if r['E'] >= 60.0]
    if feasible:
        best = min(feasible, key=lambda r: -r['eta'])
        print(f"\n最优可行: 内{best['win']}x{best['hin']}外{best['wout']}x{best['hout']} "
              f"R1={best['R1']} η={best['eta']:.5f} E={best['E']:.2f}MW A={best['A']:.0f}")
    else:
        best = max(results, key=lambda r: r['E'])
        print(f"\nWARNING 无 E>=60: 最高 E={best['E']:.2f}MW")
    with open(os.path.join(OUT, 'problem3_verify2.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
