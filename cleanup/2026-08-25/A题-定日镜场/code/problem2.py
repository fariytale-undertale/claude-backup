# -*- coding: utf-8 -*-
"""
问题2：额定功率 60MW 下最大化单位镜面面积年平均输出热功率。
所有镜尺寸/高度相同；优化塔位、镜尺寸(w×h)、安装高度、镜数、镜位。

判据（用户确认）：E≥60MW 取最小面积；三层求解（解析粗搜→快速精修→完整验证）。
布局：环形（径向/切向间距因子 dr, da）。

单位面积功率 = E/A；E≥60MW 时 ≈ DNI×η_avg，故最大化平均光学效率 η_avg。
"""
import numpy as np
import time
import json
import os
from multiprocessing import Pool

from layout import ring_layout
from common import (D_MONTHS, TIMES, TOWER_H, ETA_REF, R_FIELD, solar_position,
                    dni, mirror_normal, eta_cos, eta_at, SUN_HALF_ANGLE)
from shadow import compute_eta_sb
from truncation import compute_eta_trunc

OUT = r'D:/pdf/国赛/国赛历年真题/2023年赛题/A题/output'
TIMES60 = [(D, ST) for D in D_MONTHS for ST in TIMES]

# ================= 截断效率查找表（预计算，单方位） =================
TRUNC_TABLE_PATH = os.path.join(OUT, 'trunc_lookup.npy')
TRUNC_TABLE = {}


def precompute_trunc_tables(sizes):
    """预计算孤立镜 ηtrunc(d_HR) 查找表（塔南镜位，全年平均，K=4）。"""
    if os.path.exists(TRUNC_TABLE_PATH):
        data = np.load(TRUNC_TABLE_PATH, allow_pickle=True).item()
        TRUNC_TABLE.update(data)
        print(f"  加载截断查找表: {list(data.keys())}")
        return
    dgrid = np.arange(110, 470, 15)
    table = {}
    for (w, h) in sizes:
        vals = []
        for d in dgrid:
            r = np.sqrt(max(d * d - 76.0 * 76.0, 1e-6))
            C = np.array([[0.0, -r, 4.0]])     # 塔南侧孤立镜
            ssum = 0.0
            for D, ST in TIMES60:
                s, _, _, _, _ = solar_position(D, ST, return_all=True)
                ssum += compute_eta_trunc(C, (0, 0), s, w, h, K=4, Nr=2, Nphi=8)[0]
            vals.append(ssum / len(TIMES60))
        table[f'{w}x{h}'] = (dgrid, np.array(vals))
        print(f"  预计算截断表 {w}x{h} 完成")
    np.save(TRUNC_TABLE_PATH, table)
    TRUNC_TABLE.update(table)


def trunc_lookup(dHR, w, h):
    """查找表插值 ηtrunc。"""
    key = f'{w}x{h}'
    if key not in TRUNC_TABLE:
        # 就近尺寸
        keys = list(TRUNC_TABLE.keys())
        best = min(keys, key=lambda k: abs(int(k.split('x')[0]) - w) + abs(int(k.split('x')[1]) - h))
        key = best
    dgrid, vals = TRUNC_TABLE[key]
    return np.interp(dHR, dgrid, vals)


# ================= 目标函数 =================
def _eval_chunk(chunk, C, tower_xy, w, h, K):
    """一个时点块的 η 加权和 与 功率和。"""
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    eta_sum = 0.0
    e_sum = 0.0
    for D, ST in chunk:
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
    return eta_sum, e_sum


def objective_full(C, tower_xy, w, h, K=4, nproc=1):
    """完整目标：返回 (η_avg 面积加权, E_annual MW)。"""
    if nproc <= 1:
        et, es = _eval_chunk(TIMES60, C, tower_xy, w, h, K)
        return et / len(TIMES60), es / len(TIMES60) / 1000.0
    chunks = np.array_split(np.array(TIMES60), nproc)
    with Pool(nproc) as p:
        res = p.starmap(_eval_chunk,
                        [(list(ch), C, tower_xy, w, h, K) for ch in chunks])
    et = sum(r[0] for r in res)
    es = sum(r[1] for r in res)
    return et / len(TIMES60), es / len(TIMES60) / 1000.0


def objective_ana(C, tower_xy, w, h):
    """解析目标（层1）：ηcos×ηat×ηref，忽略遮挡/截断。返回 η_avg。"""
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    ssum = 0.0
    for D, ST in TIMES60:
        s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
        n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
        cos = eta_cos(s, r_hat)
        at = eta_at(dHR)
        eta = cos * at * ETA_REF
        ssum += (eta * A).sum() / (A * N)
    return ssum / len(TIMES60)


# ================= 优化流程 =================
def layer1_tower_search():
    """层1：解析目标粗搜塔位（场内，塔区外）。返回最优塔位。"""
    rng = np.random.default_rng(0)
    best = None
    best_eta = -1
    results = []
    t0 = time.time()
    # 候选塔位：环形布局需要塔距场中心不太远（否则布局畸形）
    # 随机采样 + 网格
    cands = []
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        for r in [40, 80, 120, 160, 200, 240]:
            cands.append((r * np.cos(a), r * np.sin(a)))
    cands = np.array(cands)
    # 排除塔区外？塔区指塔周围100m无镜，塔本身可在此
    for i in range(400):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(20, 260)
        cands = np.vstack([cands, [r * np.cos(a), r * np.sin(a)]])
    for (xt, yt) in cands:
        C = ring_layout((xt, yt), 8.0, 8.0, 5.0, 1.2, 1.4)
        if len(C) < 200:
            continue
        eta = objective_ana(C, (xt, yt), 8.0, 8.0)
        results.append((eta, xt, yt, len(C)))
        if eta > best_eta:
            best_eta = eta
            best = (xt, yt)
    print(f"层1 解析塔位搜索: {len(results)} 候选, 耗时 {time.time()-t0:.1f}s")
    print(f"  最优塔位 = ({best[0]:.1f}, {best[1]:.1f}), 解析 η={best_eta:.4f}")
    # 按效率排序 top10
    results.sort(reverse=True)
    top = results[:10]
    print("  Top5 塔位:")
    for eta, xt, yt, n in top[:5]:
        print(f"    ({xt:.1f}, {yt:.1f}) η={eta:.4f} N={n}")
    return best, top


def surrogate_objective(C, tower_xy, w, h, sb_times=None):
    """代理目标：解析 ηcos/at（60时点）+ 查找表 ηtrunc + 粗 ηsb（4时点K=2）。
    返回 (η_avg 近似, E_annual MW 近似)。用于层2快速定位。"""
    N = len(C)
    A = w * h
    tw = np.array([tower_xy[0], tower_xy[1], TOWER_H])
    rv = tw - C
    dHR = np.linalg.norm(rv, axis=1)
    r_hat = rv / dHR[:, None]
    if sb_times is None:
        sb_times = [(D, 12.0) for D in [-59, 31, 92, 214]]   # 1/4/6/10月正午
    # 每镜全年平均 ηcos×ηat
    cosat_i = np.zeros(N)
    for D, ST in TIMES60:
        s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
        n = mirror_normal(np.broadcast_to(s, (N, 3)), r_hat)
        cosat_i += eta_cos(s, r_hat) * eta_at(dHR) / len(TIMES60)
    tr_i = trunc_lookup(dHR, w, h)          # 查找表
    sb_sum = np.zeros(N)
    for D, ST in sb_times:
        s, _, _, _, _ = solar_position(D, ST, return_all=True)
        sb_sum += compute_eta_sb(C, tower_xy, s, w, h, K=2)
    sb_i = sb_sum / len(sb_times)
    eta_i = cosat_i * tr_i * sb_i * ETA_REF
    eta_avg = (eta_i * A).sum() / (A * N)
    # E ≈ DNI_avg × ΣA·η_i（年均，DNI 与 η 弱相关）
    dni_avg = 0.968
    E_mw = dni_avg * (eta_i * A).sum() / 1000.0
    return eta_avg, E_mw


def layer2_refine(tower0):
    """层2：代理目标枚举（塔位/尺寸/间距/高度）定位 top，再完整目标精修 top5。"""
    precompute_trunc_tables([(8, 8), (8, 6), (7, 7), (6, 6), (6, 4)])
    t0 = time.time()
    tower_cands = [tower0, (0, -180), (0, -120), (0, -60), (0, 0),
                   (-40, -160), (40, -160)]
    sizes = [(8, 8), (8, 6), (7, 7), (6, 6), (6, 4)]
    spacings = [(1.0, 1.2), (1.2, 1.4), (1.4, 1.6), (1.8, 2.0)]
    candidates = []
    for (xt, yt) in tower_cands:
        for (w, h) in sizes:
            h_inst = max(4.0, h / 2 + 1.0)
            for (dr, da) in spacings:
                C = ring_layout((xt, yt), w, h, h_inst, dr, da)
                if len(C) < 100:
                    continue
                eta, E = surrogate_objective(C, (xt, yt), w, h)
                fit = -eta if E >= 60.0 else 1e6 + (60.0 - E)
                candidates.append(dict(xt=xt, yt=yt, w=w, h=h, h_inst=h_inst,
                                       dr=dr, da=da, N=len(C), eta_s=eta,
                                       E_s=E, fit=fit))
    print(f"层2 代理枚举: {len(candidates)} 候选, 耗时 {time.time()-t0:.1f}s")
    cands = sorted(candidates, key=lambda d: d['fit'])
    print("  Top8 代理候选:")
    for d in cands[:8]:
        print(f"    塔({d['xt']:.0f},{d['yt']:.0f}) {d['w']:.0f}x{d['h']:.0f} "
              f"h={d['h_inst']:.1f} dr={d['dr']} da={d['da']} N={d['N']} "
              f"η_s={d['eta_s']:.4f} E_s={d['E_s']:.1f}MW")
    # 完整目标精修 top5
    print("\n  完整目标(K=4)精修 top5:")
    t1 = time.time()
    for d in cands[:5]:
        C = ring_layout((d['xt'], d['yt']), d['w'], d['h'], d['h_inst'], d['dr'], d['da'])
        eta, E = objective_full(C, (d['xt'], d['yt']), d['w'], d['h'], K=4, nproc=1)
        d['eta'] = eta
        d['E'] = E
        d['A'] = len(C) * d['w'] * d['h']
        print(f"    塔({d['xt']:.0f},{d['yt']:.0f}) {d['w']:.0f}x{d['h']:.0f} "
              f"dr={d['dr']} N={d['N']} η={eta:.4f} E={E:.1f}MW "
              f"{'可行' if E>=60 else '不足'}")
    print(f"  完整精修耗时 {time.time()-t1:.1f}s")
    feasible = [d for d in cands if d.get('E', 0) >= 60]
    if feasible:
        best = min(feasible, key=lambda d: -d['eta'])
    else:
        best = cands[0]
        print("  WARNING: 无可行（E≥60MW）候选，选 E 最大的")
    return best, cands


def layer3_verify(best, nproc=1):
    """层3：60时点 K=12 完整验证 + 60MW 面积处理。"""
    t0 = time.time()
    C = ring_layout((best['xt'], best['yt']), best['w'], best['h'],
                    best['h_inst'], best['dr'], best['da'])
    eta, E = objective_full(C, (best['xt'], best['yt']), best['w'], best['h'],
                            K=12, nproc=nproc)
    best['eta_full'] = eta
    best['E_full'] = E
    best['N_full'] = len(C)
    best['A_full'] = len(C) * best['w'] * best['h']
    print(f"层3 完整验证(K=12): η={eta:.5f}, E={E:.2f}MW, N={len(C)}, "
          f"A={best['A_full']:.0f}, 耗时 {time.time()-t0:.1f}s")
    # 60MW 最小面积：按每镜年均贡献排序，删镜到 60MW
    if E >= 60.0:
        A = best['w'] * best['h']
        tw = np.array([best['xt'], best['yt'], TOWER_H])
        rv = tw - C
        dHR = np.linalg.norm(rv, axis=1)
        r_hat = rv / dHR[:, None]
        # 每镜年均效率贡献（用 K=12 时点计算近似）
        contrib = np.zeros(len(C))
        from common import TIMES60
        for D, ST in TIMES60:
            s, alpha, _, _, _ = solar_position(D, ST, return_all=True)
            d = dni(alpha)
            n = mirror_normal(np.broadcast_to(s, (len(C), 3)), r_hat)
            cos = eta_cos(s, r_hat)
            at = eta_at(dHR)
            sb = compute_eta_sb(C, (best['xt'], best['yt']), s, best['w'], best['h'], K=12)
            tr = compute_eta_trunc(C, (best['xt'], best['yt']), s, best['w'], best['h'], K=12)
            eta_i = cos * at * sb * tr * ETA_REF
            contrib += d * A * eta_i / len(TIMES60)   # kW
        order = np.argsort(-contrib)
        cum = np.cumsum(contrib[order])
        n_keep = int(np.searchsorted(cum, 60000.0)) + 1
        n_keep = min(n_keep, len(C))
        A_sel = n_keep * A
        E_sel = cum[n_keep - 1]
        unit = E_sel / A_sel
        best['n_keep'] = n_keep
        best['A_sel'] = A_sel
        best['E_sel'] = E_sel
        best['unit_power'] = unit
        print(f"  60MW 最小面积: 保留 {n_keep}/{len(C)} 镜, A={A_sel:.0f}m2, "
              f"E={E_sel:.1f}kW, 单位面积功率={unit:.4f} kW/m2")
    return best


def main():
    print("========== 问题2 求解 ==========")
    tower, top = layer1_tower_search()
    best, cands = layer2_refine(tower)
    best = layer3_verify(best, nproc=1)
    # 保存
    with open(os.path.join(OUT, 'problem2_result.json'), 'w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=2, default=float)
    np.save(os.path.join(OUT, 'problem2_layout.npy'),
            ring_layout((best['xt'], best['yt']), best['w'], best['h'],
                        best['h_inst'], best['dr'], best['da']))
    print("\n问题2最优参数:")
    for k in ['xt', 'yt', 'w', 'h', 'h_inst', 'dr', 'da', 'N_full', 'eta_full',
              'E_full', 'A_sel', 'E_sel', 'unit_power']:
        if k in best:
            print(f"  {k} = {best[k]}")
    print(f"  单位面积年平均输出热功率 = {best.get('unit_power', 'N/A')} kW/m2")


if __name__ == '__main__':
    main()
