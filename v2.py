#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

###############################################################################
# 辅助函数: theil、GramSchmidt、lag_normalize
###############################################################################
def theil(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    slopes = []
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            if x[j] != x[i]:
                slopes.append((y[j]-y[i])/(x[j]-x[i]))
    b = np.median(slopes) if len(slopes)>0 else np.nan
    a = np.median(y - b*x) if len(slopes)>0 else np.nan
    return a, b

def GramSchmidt(A):
    A = np.array(A, dtype=float)
    nrow, ncol = A.shape
    Q = np.zeros((nrow, ncol))
    R = np.zeros((ncol, ncol))
    R[0,0] = np.sqrt(np.dot(A[:,0], A[:,0]))
    Q[:,0] = A[:,0]/R[0,0] if R[0,0]!=0 else A[:,0]
    for j in range(1,ncol):
        v = A[:,j].copy()
        for i in range(j):
            R[i,j] = np.dot(Q[:,i], A[:,j])
            v -= R[i,j]*Q[:,i]
        R[j,j] = np.sqrt(np.dot(v,v))
        if R[j,j]!=0:
            Q[:,j] = v/R[j,j]
    return {"Q": Q, "R": R}

def lag_normalize(x, l=12):
    """
    严格模拟 R 代码: y = rev(-diff(rev(x), lag=l))
    返回长度 = len(x) - l
    """
    arr = np.array(x, dtype=float)
    if len(arr)<=l:
        return np.array([], dtype=float)
    rev_arr = arr[::-1]
    diff_list = []
    for i in range(l, len(rev_arr)):
        diff_list.append(rev_arr[i]-rev_arr[i-l])
    diff_arr = np.array(diff_list, dtype=float)
    y = -diff_arr[::-1]
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.full_like(y, np.nan)
    mu = np.nanmean(y[valid])
    sigma = np.nanstd(y[valid], ddof=1)
    z = (y - mu)/sigma if sigma!=0 else (y - mu)
    return z

###############################################################################
# 函数 1: get_ZidxA
#   1) 先计算 sector 0..5 的“原始指数”(未差分)
#   2) 在函数内部, 对每个 region 写出 "_ZidxA{i_reg+1}.csv"
###############################################################################
def get_ZidxA(a0, a0_ts, a0_anno, i_regs, n_sec, idx0, n_iss_idx0, out_folder="log"):
    """
    参数:
      a0: 读入并 TB过滤后的原始DataFrame, 含 Region等
      a0_ts: 只含时间序列的部分(每行=issue, 每列=时间点)
      a0_anno: 含 Region, Sector等注释列
      i_regs: 4个region的行号list
      n_sec: sector总数(含0)
      idx0, n_iss_idx0: 全局 sector0
      out_folder: 输出文件夹, 默认 "log"

    在函数内部:
      对 i_reg in [0..3], 构造 idx(未差分), 写出:
         log/_ZidxA{i_reg+1}.csv
    """
    n_t = a0_ts.shape[1]
    os.makedirs(out_folder, exist_ok=True)

    for i_reg_idx in range(4):
        reg_inds = i_regs[i_reg_idx]
        a_region = a0_ts.iloc[reg_inds, :]  # shape:(n_issue_region, n_t)
        a_anno_region = a0_anno.iloc[reg_inds, :]

        # sector 1..5 行号
        i_secs = []
        for s in range(1, n_sec):
            i_secs.append(np.where(a_anno_region['Sector'].values == s)[0])

        # idx(n_t, n_sec)
        idx = np.full((n_t, n_sec), np.nan, dtype=float)
        idx[:,0] = idx0
        n_iss_idx = np.zeros((n_t, n_sec), dtype=int)
        n_iss_idx[:,0] = n_iss_idx0

        # 构造 sector(1..5)
        for i_col in range(n_t):
            for j_sec in range(1, n_sec):
                rows_j = i_secs[j_sec-1]
                col_data = a_region.iloc[rows_j, i_col]
                idx[i_col, j_sec] = np.nanmean(col_data)
                n_iss_idx[i_col, j_sec] = col_data.notna().sum()

        # 对 NA 插补 => 用 sector0 拟合
        for sec_col in range(n_sec):
            if np.isnan(idx[:,sec_col]).any():
                jkk = np.where(np.isnan(idx[:,sec_col]))[0]
                good_mask = ~np.isnan(idx[:,sec_col]) & ~np.isnan(idx[:,0])
                if np.sum(good_mask)>1:
                    X_good = idx[good_mask, 0]
                    Y_good = idx[good_mask, sec_col]
                    A_ = np.column_stack((np.ones(len(X_good)), X_good))
                    coefs, _, _, _ = np.linalg.lstsq(A_, Y_good, rcond=None)
                    a_, b_ = coefs[0], coefs[1]
                    idx[jkk, sec_col] = a_ + b_*idx[jkk,0]

        df_idxA = pd.DataFrame(idx, columns=[f"I{k}" for k in range(n_sec)])
        # 写出 => log/_ZidxA{i_reg_idx+1}.csv
        df_idxA.to_csv(os.path.join(out_folder, f"_ZidxA{i_reg_idx+1}.csv"), index=False)

###############################################################################
# 函数 2: get_ZidxBB
#   1) 读取每个 region 的 _ZidxA{i_reg+1}.csv
#   2) 对每列做 lag_normalize => 行数减少12
#   3) 写出 _ZidxB{i_reg+1}.csv
###############################################################################
def get_ZidxBB(n_sec, lag_month=12, in_folder="log", out_folder="log"):
    os.makedirs(out_folder, exist_ok=True)
    for i_reg_idx in range(4):
        # 读入 _ZidxA{i_reg+1}.csv
        path_in = os.path.join(in_folder, f"_ZidxA{i_reg_idx+1}.csv")
        df_idxA = pd.read_csv(path_in)
        # 做差分
        arr = df_idxA.to_numpy(dtype=float)  # shape:(n_t, n_sec)
        n_t_orig = arr.shape[0]

        result_list = []
        for c in range(n_sec):
            col_out = lag_normalize(arr[:, c], lag_month)
            result_list.append(col_out)
        idx_dn = np.column_stack(result_list)  # (n_t_orig-lag_month, n_sec)

        df_idxB = pd.DataFrame(idx_dn, columns=[f"dI{k}" for k in range(n_sec)])
        # 写出
        path_out = os.path.join(out_folder, f"_ZidxB{i_reg_idx+1}.csv")
        df_idxB.to_csv(path_out, index=False)

###############################################################################
# 函数 3: get_ZvB
#   1) 读取每个 region 的 _ZidxB{i_reg+1}.csv
#   2) 对 NA 行删除后做 GramSchmidt => vb => vb0(含 NA回补)
#   3) 写出 _Zvb{i_reg+1}.csv
###############################################################################
def get_ZvB(n_sec, in_folder="log", out_folder="log"):
    os.makedirs(out_folder, exist_ok=True)
    for i_reg_idx in range(4):
        path_in = os.path.join(in_folder, f"_ZidxB{i_reg_idx+1}.csv")
        df_idxB = pd.read_csv(path_in)
        arr_idxB = df_idxB.to_numpy(dtype=float)  # shape: (n_t_new, n_sec)

        mask_na = np.any(np.isnan(arr_idxB), axis=1)
        i_ex = np.where(mask_na)[0]
        if len(i_ex)>0:
            idx_clean = np.delete(arr_idxB, i_ex, axis=0)
        else:
            idx_clean = arr_idxB

        Fi = GramSchmidt(idx_clean)
        Q = Fi["Q"]
        # vb
        if idx_clean.shape[0]>1:
            vb_clean = Q*np.sqrt(idx_clean.shape[0]-1)
        else:
            vb_clean = Q

        # 回填到原行数
        vb0 = np.full_like(arr_idxB, np.nan)
        if len(i_ex)>0:
            good_inds = np.setdiff1d(np.arange(arr_idxB.shape[0]), i_ex)
            vb0[good_inds,:] = vb_clean
        else:
            vb0 = vb_clean

        df_vb = pd.DataFrame(vb0, columns=[f"F{k}" for k in range(n_sec)])
        path_out = os.path.join(out_folder, f"_Zvb{i_reg_idx+1}.csv")
        df_vb.to_csv(path_out, index=False)

###############################################################################
# 函数 4: get_Za
#   1) 对每个 region 的 issues (a)做 lag_normalize
#   2) 写出 _Za{i_reg+1}.csv
###############################################################################
def get_Za(a0_ts, a0_anno, i_regs, lag_month=12, out_folder="log"):
    os.makedirs(out_folder, exist_ok=True)
    for i_reg_idx in range(4):
        reg_inds = i_regs[i_reg_idx]
        a_region = a0_ts.iloc[reg_inds, :]  # shape:(n_issues, n_t)
        # 行行差分
        all_out = []
        for row_i in range(a_region.shape[0]):
            x_in = a_region.iloc[row_i, :].to_numpy(dtype=float)
            x_out = lag_normalize(x_in, lag_month)
            all_out.append(x_out)
        a_dn = np.row_stack(all_out)  # shape: (n_issues, n_t-lag_month)
        df_a = pd.DataFrame(a_dn)
        df_a.to_csv(os.path.join(out_folder, f"_Za{i_reg_idx+1}.csv"), index=False)

###############################################################################
# 主函数: 演示如何调用上述4个函数
###############################################################################
def main():
    # 准备输入数据
    input_file = "data\\input_TB_2020.csv"
    a0 = pd.read_csv(input_file)
    # 如果要 TB.only
    a0 = a0[a0["TB"]==True].copy()

    # 假设前5列为注释, 后面列数为时间序列
    n_anno = 5
    a0_anno = a0.iloc[:, :n_anno].copy()
    a0_ts   = a0.iloc[:, n_anno:].copy()

    # region 索引 (与 R 大致对应)
    i_regs = []
    i_regs.append(np.where(a0['Region']==1)[0]) 
    i_regs.append(np.where((a0['Region']==2)|(a0['Region']==4))[0])
    i_regs.append(np.where((a0['Region']==3)|(a0['Region']==5))[0])
    i_regs.append(np.where((a0['Region']==6)|(a0['Region']==7))[0])

    n_sec = 6
    # 先计算 idx0(全局 sector0)
    n_t = a0_ts.shape[1]
    idx0 = np.zeros(n_t, dtype=float)
    n_iss_idx0 = np.zeros(n_t, dtype=int)
    for i in range(n_t):
        col_data = a0_ts.iloc[:, i]
        idx0[i] = np.nanmean(col_data)
        n_iss_idx0[i] = col_data.notna().sum()

    # 1) 生成并导出 4 个 Region 的 "_ZidxA{i_reg+1}.csv"
    get_ZidxA(a0, a0_ts, a0_anno, i_regs, n_sec, idx0, n_iss_idx0, out_folder="log")

    # 2) 基于 "_ZidxA" 做差分+标准化, 导出 4 个 "_ZidxB{i_reg+1}.csv"
    get_ZidxBB(n_sec, lag_month=12, in_folder="log", out_folder="log")

    # 3) 基于 "_ZidxB" 做 GramSchmidt, 导出 4 个 "_Zvb{i_reg+1}.csv"
    get_ZvB(n_sec, in_folder="log", out_folder="log")

    # 4) 对每个 region 的 issues 做差分, 导出 4 个 "_Za{i_reg+1}.csv"
    get_Za(a0_ts, a0_anno, i_regs, lag_month=12, out_folder="log")

    print("All done. Check 'log' folder for _ZidxA1..4.csv, _ZidxB1..4.csv, _Zvb1..4.csv, _Za1..4.csv")

if __name__=="__main__":
    main()