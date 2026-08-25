# -*- coding: utf-8 -*-
"""问题2快速搜索：查找表代理目标 + 塔位/间距扫描 + 校准。"""
import numpy as np
import time
import os
import sys

from layout import hex_layout
from common import (D_MONTHS, TIMES, TOWER_H, ETA_REF, solar_position, dni,
                    mirror_normal, eta_cos, eta_at)
from shadow import compute_eta_sb
from truncation import compute_eta_trunc
from problem2 import TRUNC_TABLE, precompute_trunc_tables, trunc_lookup

OUT = r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/output'
TIMES60 = [(D, ST) for D in D_MONTHS for ST in TIMES]


def fast_objective(C, tower_xy, w, h):
    """快速代理：ηsb K=2 全60时点 + ηtrunc 查找表 + 解析 ηcos/at。
    返回 (η_avg 面积加权, E_annual MW)。"""
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    # 每镜年均 ηcos×ηat（解析 60 时点）
    cosat_i = np.zeros(N)
    dni_avg = 0.0
    for D, ST in TIMES60:
        s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
        n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
        cosat_i += eta_cos(s, r_hat) * eta_at(dHR) / len(TIMES60)
        dni_avg += dni(alpha) / len(TIMES60)
    tr_i = trunc_lookup(dHR, w, h)
    # ηsb K=2 全 60 时点
    sb_i = np.zeros(N)
    for D, ST in TIMES60:
        s, _, _, _, _ = solar_position(D, ST, return_all=True)
        sb_i += compute_eta_sb(C, tower_xy, s, w, h, K=2) / len(TIMES60)
    eta_i = cosat_i * tr_i * sb_i * ETA_REF
    eta_avg = (eta_i * A).sum() / (A * N)
    E_mw = dni_avg * (eta_i * A).sum() / 1000.0
    return eta_avg, E_mw


def full_objective(C, tower_xy, w, h, K=2):
    """完整目标（对照用）：ηsb K 全时点 + ηtrunc K 全时点。"""
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    eta_sum = 0.0
    e_sum = 0.0
    for D, ST in TIMES60:
        s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
        d = dni(alpha)
        n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
        cos = eta_cos(s, r_hat)
        at = eta_at(dHR)
        sb = compute_eta_sb(C, tower_xy, s, w, h, K=K)
        tr = compute_eta_trunc(C, tower_xy, s, w, h, K=K, Nr=2, Nphi=8)
        eta = cos * at * sb * tr * ETA_REF
        eta_sum += (eta * A).sum() / (A * N)
        e_sum += d * (eta * A).sum()
    return eta_sum / len(TIMES60), e_sum / len(TIMES60) / 1000.0


if __name__ == '__main__':
    precompute_trunc_tables([(6, 6), (6, 5), (7, 7), (8, 8)])

    # 校准：代理 vs 完整（K=4）
    print("=== 校准（fast vs full K=4，60时点）===", flush=True)
    calib = [((0, 0), 6, 6, 1.0), ((0, -100), 6, 6, 1.0), ((0, 0), 6, 6, 1.15)]
    for tower, w, h, df in calib:
        C = hex_layout(tower, w, h, max(4.0, h / 2 + 1.0), df)
        t0 = time.time()
        eta_f, E_f = fast_objective(C, tower, w, h)
        t1 = time.time()
        eta_c, E_c = full_objective(C, tower, w, h, K=4)
        t2 = time.time()
        print(f"塔{tower} {w}x{h} df={df} N={len(C)}: "
              f"fast η={eta_f:.4f} E={E_f:.1f} ({t1-t0:.0f}s) | "
              f"full η={eta_c:.4f} E={E_c:.1f} ({t2-t1:.0f}s) | "
              f"Δη={abs(eta_f-eta_c):.4f}", flush=True)
