#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

###############################################################################
# 辅助函数定义
###############################################################################

def theil(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slopes = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    b = np.median(slopes) if slopes else np.nan
    a = np.median(y - b * x) if slopes else np.nan
    return a, b

def GramSchmidt(A):
    A = np.asarray(A, dtype=float)
    nrow, ncol = A.shape
    Q = np.zeros((nrow, ncol))
    R = np.zeros((ncol, ncol))
    R[0,0] = np.sqrt(np.dot(A[:,0], A[:,0]))
    Q[:,0] = A[:,0] / R[0,0] if R[0,0] != 0 else A[:,0]
    for j in range(1, ncol):
        v = A[:,j].copy()
        for i in range(j):
            R[i,j] = np.dot(Q[:,i], A[:,j])
            v -= R[i,j] * Q[:,i]
        R[j,j] = np.sqrt(np.dot(v,v))
        if R[j,j] != 0:
            Q[:,j] = v / R[j,j]
    return {"Q": Q, "R": R}

def lag_normalize(x, l=12):
    """
    优化后的 lag_normalize:
    使用向量化操作替代显式循环，实现 R 中 rev(-diff(rev(x), lag=l)) 的功能。
    返回长度 = len(x) - l
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= l:
        return np.array([], dtype=float)
    rev_x = x[::-1]
    diff_vals = rev_x[l:] - rev_x[:-l]
    y = -diff_vals[::-1]
    valid = ~np.isnan(y)
    if not valid.any():
        return np.full_like(y, np.nan)
    mu = np.nanmean(y[valid])
    sigma = np.nanstd(y[valid], ddof=1)
    return (y - mu) / sigma if sigma != 0 else (y - mu)

###############################################################################
# 计算原始指数并导出 _ZidxA 文件
###############################################################################
def get_ZidxA(a0, a0_ts, a0_anno, i_regs, n_sec, idx0, n_iss_idx0, out_folder="log"):
    n_t = a0_ts.shape[1]
    os.makedirs(out_folder, exist_ok=True)

    for i_reg_idx in range(4):
        reg_inds = i_regs[i_reg_idx]
        a_region = a0_ts.iloc[reg_inds, :]
        a_anno_region = a0_anno.iloc[reg_inds, :]

        i_secs = [np.where(a_anno_region['Sector'].values == s)[0] for s in range(1, n_sec)]

        idx = np.full((n_t, n_sec), np.nan, dtype=float)
        idx[:,0] = idx0
        n_iss_idx = np.zeros((n_t, n_sec), dtype=int)
        n_iss_idx[:,0] = n_iss_idx0

        for j_sec in range(1, n_sec):
            rows = i_secs[j_sec-1]
            if len(rows) > 0:
                sector_data = a_region.iloc[rows, :]
                idx[:, j_sec] = sector_data.mean(axis=0)
                n_iss_idx[:, j_sec] = sector_data.notna().sum(axis=0)

        for sec_col in range(n_sec):
            if np.isnan(idx[:, sec_col]).any():
                jkk = np.where(np.isnan(idx[:, sec_col]))[0]
                good_mask = ~np.isnan(idx[:, sec_col]) & ~np.isnan(idx[:,0])
                if np.sum(good_mask) > 1:
                    X_good = idx[good_mask, 0]
                    Y_good = idx[good_mask, sec_col]
                    A_ = np.column_stack((np.ones(len(X_good)), X_good))
                    coefs, _, _, _ = np.linalg.lstsq(A_, Y_good, rcond=None)
                    idx[jkk, sec_col] = coefs[0] + coefs[1] * idx[jkk,0]

        df_idxA = pd.DataFrame(idx, columns=[f"I{k}" for k in range(n_sec)])
        df_idxA.to_csv(os.path.join(out_folder, f"_ZidxA{i_reg_idx+1}.csv"), index=False)

###############################################################################
# 生成 _ZidxB 文件
###############################################################################
def get_ZidxBB(n_sec, lag_month=12, in_folder="log", out_folder="log"):
    os.makedirs(out_folder, exist_ok=True)
    for i_reg_idx in range(4):
        path_in = os.path.join(in_folder, f"_ZidxA{i_reg_idx+1}.csv")
        df_idxA = pd.read_csv(path_in)
        arr = df_idxA.to_numpy(dtype=float)
        n_t_orig = arr.shape[0]

        result_list = []
        for c in range(n_sec):
            col_out = lag_normalize(arr[:, c], lag_month)
            result_list.append(col_out)
        idx_dn = np.column_stack(result_list)

        df_idxB = pd.DataFrame(idx_dn, columns=[f"dI{k}" for k in range(n_sec)])
        path_out = os.path.join(out_folder, f"_ZidxB{i_reg_idx+1}.csv")
        df_idxB.to_csv(path_out, index=False)

###############################################################################
# 生成 _Zvb 文件
###############################################################################
def get_ZvB(n_sec, in_folder="log", out_folder="log"):
    os.makedirs(out_folder, exist_ok=True)
    for i_reg_idx in range(4):
        path_in = os.path.join(in_folder, f"_ZidxB{i_reg_idx+1}.csv")
        df_idxB = pd.read_csv(path_in)
        arr_idxB = df_idxB.to_numpy(dtype=float)

        mask_na = np.any(np.isnan(arr_idxB), axis=1)
        i_ex = np.where(mask_na)[0]
        idx_clean = np.delete(arr_idxB, i_ex, axis=0) if len(i_ex)>0 else arr_idxB

        Fi = GramSchmidt(idx_clean)
        Q = Fi["Q"]
        vb_clean = Q * np.sqrt(idx_clean.shape[0]-1) if idx_clean.shape[0]>1 else Q

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
# 生成 _Za 文件
###############################################################################
def get_Za(a0_ts, a0_anno, i_regs, lag_month=12, out_folder="log"):
    os.makedirs(out_folder, exist_ok=True)
    for i_reg_idx in range(4):
        reg_inds = i_regs[i_reg_idx]
        a_region = a0_ts.iloc[reg_inds, :]
        all_out = []
        for row_i in range(a_region.shape[0]):
            x_in = a_region.iloc[row_i, :].to_numpy(dtype=float)
            x_out = lag_normalize(x_in, lag_month)
            all_out.append(x_out)
        a_dn = np.row_stack(all_out)
        df_a = pd.DataFrame(a_dn)
        df_a.to_csv(os.path.join(out_folder, f"_Za{i_reg_idx+1}.csv"), index=False)

###############################################################################
# 计算回归和结果导出函数
###############################################################################
def calc_betas_and_results(a0, a0_anno, i_regs, n_sec=6, lag_month=12,
                           log_folder="log", result_folder="result"):
    os.makedirs(result_folder, exist_ok=True)

    # 最终要存放回归系数的矩阵 [n_issues, n_sec + 4]
    # (b0..b5, intercept, n.issue, R-sq, adj.R-sq)
    betas = np.full((a0.shape[0], n_sec+4), np.nan, dtype=float)

    # 逐个 region 做处理
    for i_reg_idx in range(4):
        # 1) 读入 _Zvb{i_reg+1}.csv
        path_vb = os.path.join(log_folder, f"_Zvb{i_reg_idx+1}.csv")
        df_vb = pd.read_csv(path_vb)
        arr_vb = df_vb.to_numpy(dtype=float)  # shape: (n_t-lag_month, n_sec), 但中间含 NA

        # 找 NA 行 => R 中 i.ex
        mask_na_row = np.any(np.isnan(arr_vb), axis=1)
        i_ex = np.where(mask_na_row)[0]

        # 2) 读入 _Za{i_reg+1}.csv
        path_a = os.path.join(log_folder, f"_Za{i_reg_idx+1}.csv")
        df_a = pd.read_csv(path_a)
        arr_a = df_a.to_numpy(dtype=float)  # shape: (n_issue_region, n_t-lag_month)

        # 如果 i_ex 不空, 需要把 arr_a 中对应的列删除 => Python 中 axis=1
        # R 里: a<-a[,-i.ex]
        if len(i_ex)>0:
            arr_vb_clean = np.delete(arr_vb, i_ex, axis=0)  # shape: (some_row, n_sec)
            arr_a_clean  = np.delete(arr_a, i_ex, axis=1)  # shape: (n_issue_region, some_row)
        else:
            arr_vb_clean = arr_vb
            arr_a_clean  = arr_a

        # region 的 issue 索引
        reg_inds = i_regs[i_reg_idx]

        n_issues_region = arr_a_clean.shape[0]
        for i_issue_local in range(n_issues_region):
            x = arr_a_clean[i_issue_local, :]  # shape: (some_row,)
            ii = reg_inds[i_issue_local]       # 在 a0 中的行号

            # 只保留非NA
            valid_mask = ~np.isnan(x)
            if np.sum(valid_mask)>10:
                X_fac = arr_vb_clean[valid_mask,:]  # shape: (k, n_sec)
                y = x[valid_mask]

                # 回归 y ~ X_fac
                # => y = c0 + c1*F1 + ...
                X_design = np.column_stack((np.ones(X_fac.shape[0]), X_fac))
                coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                # coefs[0] = intercept, coefs[1..n_sec] = factor loadings

                # 回归统计
                y_hat = X_design @ coefs
                resid = y - y_hat
                ss_res = np.sum(resid**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r2 = 1-ss_res/ss_tot if ss_tot!=0 else np.nan
                df_reg = X_fac.shape[1]  # n_sec
                n_obs = X_fac.shape[0]
                if n_obs - df_reg - 1>0:
                    adj_r2 = 1-(1-r2)*(n_obs-1)/(n_obs-df_reg-1)
                else:
                    adj_r2 = np.nan

                # 缩放因子 (原R里 Fi$Q*... => beta.SF = apply(vb[k,],2, sqrt(sum(x*x))) ...
                factor_norm = np.sqrt(np.sum(X_fac**2, axis=0))
                x_sd = np.std(y, ddof=1)
                denom = x_sd*np.sqrt(np.sum(valid_mask)-1)
                if denom!=0:
                    beta_sf = factor_norm/denom
                else:
                    beta_sf = np.zeros(n_sec)

                b_original = coefs[1:]  # coefs[1..]
                b_scaled   = b_original*beta_sf

                # 写入 betas
                betas[ii, :n_sec]    = b_scaled
                betas[ii, n_sec]     = coefs[0]     # intercept
                betas[ii, n_sec+1]   = np.sum(valid_mask) # n.issue
                betas[ii, n_sec+2]   = r2
                betas[ii, n_sec+3]   = adj_r2

    # 给 betas 加列名 => b0..b5, intercept, n.issue, R-sq, adj.R-sq
    col_betas = [f"b{k}" for k in range(n_sec)] + ["intercept","n.issue","R-sq","adj.R-sq"]
    df_betas = pd.DataFrame(betas, columns=col_betas)

    # 计算 rho = sum of b0^2..b5^2
    b_sqr = np.sum(df_betas.iloc[:, :n_sec].values**2, axis=1)
    df_betas["rho"] = b_sqr
    good_rho_mask = (~np.isnan(b_sqr)) & (b_sqr <=1)
    df_betas["good.rho"] = good_rho_mask

    # 合并注释
    output = pd.concat([a0_anno.reset_index(drop=True), df_betas], axis=1)
    # 写 betas_all
    fn_all = os.path.join(result_folder, f"betas_all_lag_{lag_month}.csv")
    output.to_csv(fn_all, index=False)

    # 分桶统计 => region, sector, rating => 计算平均/标准差
    # 假设 a0_anno 的列 3=Region, 4=Sector, 5=Rating(自行核对)
    # 计算 pairwise correlation 和分桶统计前的准备
    b_cols = [f"b{k}" for k in range(n_sec)]
    region_col = "Region"
    sector_col = "Sector"
    rating_col = "Rating"

    unique_regions = [0, 1, 2, 3, 4]
    unique_sectors = range(0, 6)
    unique_ratings = range(0, 8)

    avg_rows = []
    sd_rows = []

    df_good = output.copy()  # 未过滤的完整数据

    for i_reg in unique_regions:
        if i_reg == 0:
            output1 = df_good
        else:
            output1 = df_good[df_good[region_col] == i_reg]

        for i_sec in unique_sectors:
            for i_rat in unique_ratings:
                cond = (output1["good.rho"] == True)
                if i_sec > 0:
                    cond &= (output1[sector_col] == i_sec)
                if i_rat > 0:
                    cond &= (output1[rating_col] == i_rat)
                sel_df = output1[cond]

                count_i = len(sel_df)
                if count_i > 0:
                    bmat = sel_df[b_cols].to_numpy()
                    mean_ = np.round(np.nanmean(bmat, axis=0), 4)
                    std_ = np.round(np.nanstd(bmat, axis=0, ddof=1), 4)
                else:
                    mean_ = [np.nan] * n_sec
                    std_ = [np.nan] * n_sec

                avg_rows.append([i_reg, i_sec, i_rat, count_i] + list(mean_))
                sd_rows.append([i_reg, i_sec, i_rat, count_i] + list(std_))

    col_names = ["Region", "Sector", "Rating", "Count"] + [f"b{k}" for k in range(n_sec)]
    df_avg = pd.DataFrame(avg_rows, columns=col_names)
    df_sd = pd.DataFrame(sd_rows, columns=col_names)

    empty_buckets = df_avg[df_avg["Count"] < 5].index
    for k in empty_buckets:
        R_ = df_avg.at[k, "Region"]
        S_ = df_avg.at[k, "Sector"]

        cond1 = (df_avg["Region"] == R_) & (df_avg["Sector"] == S_) & (df_avg["Rating"] == 0) & (df_avg["Count"] >= 5)
        c1 = df_avg[cond1]
        if not c1.empty:
            df_avg.loc[k, b_cols] = c1.iloc[0][b_cols]
            df_sd.loc[k, b_cols] = c1.iloc[0][b_cols]
        else:
            cond2 = (df_avg["Region"] == R_) & (df_avg["Sector"] == 0) & (df_avg["Rating"] == 0) & (df_avg["Count"] >= 5)
            c2 = df_avg[cond2]
            if not c2.empty:
                df_avg.loc[k, b_cols] = c2.iloc[0][b_cols]
                df_sd.loc[k, b_cols] = c2.iloc[0][b_cols]
            else:
                cond3 = (df_avg["Region"] == 0) & (df_avg["Sector"] == 0) & (df_avg["Rating"] == 0) & (df_avg["Count"] >= 5)
                c3 = df_avg[cond3]
                if not c3.empty:
                    df_avg.loc[k, b_cols] = c3.iloc[0][b_cols]
                    df_sd.loc[k, b_cols] = c3.iloc[0][b_cols]

    fn_avg = os.path.join(result_folder, f"betas_mean_lag_{lag_month}.csv")
    fn_sd = os.path.join(result_folder, f"betas_sd_lag_{lag_month}.csv")
    df_avg.to_csv(fn_avg, index=False)
    df_sd.to_csv(fn_sd, index=False)

    print(f"Result files written: {fn_all}, {fn_avg}, {fn_sd}")

###############################################################################
# 主函数入口
###############################################################################
def main():
    input_file = "data\\input_TB_2020.csv"
    a0 = pd.read_csv(input_file)
    a0 = a0[a0["TB"] == True].copy()

    n_anno = 5
    a0_anno = a0.iloc[:, :n_anno].copy()
    a0_ts = a0.iloc[:, n_anno:].copy()

    region_groups = [
    [1],
    [2, 4],
    [3, 5],
    [6, 7]
]
    i_regs = [np.where(a0['Region'].isin(grp))[0] for grp in region_groups]

    n_sec = 6
    n_t = a0_ts.shape[1]
    idx0 = np.zeros(n_t, dtype=float)
    n_iss_idx0 = np.zeros(n_t, dtype=int)
    for i in range(n_t):
        col_data = a0_ts.iloc[:, i]
        idx0[i] = np.nanmean(col_data)
        n_iss_idx0[i] = col_data.notna().sum()

    os.makedirs("log", exist_ok=True)
    os.makedirs("result", exist_ok=True)

    get_ZidxA(a0, a0_ts, a0_anno, i_regs, n_sec, idx0, n_iss_idx0, out_folder="log")
    get_ZidxBB(n_sec, lag_month=12, in_folder="log", out_folder="log")
    get_ZvB(n_sec, in_folder="log", out_folder="log")
    get_Za(a0_ts, a0_anno, i_regs, lag_month=12, out_folder="log")

    # 注意：此处 output 需要由实际回归逻辑生成
    output = pd.DataFrame()  # TODO: 替换为实际生成的 output DataFrame

    calc_betas_and_results(a0, a0_anno, i_regs, n_sec=n_sec, lag_month=12,
                           log_folder="log", result_folder="result")

    print("All done.")

if __name__ == "__main__":
    main()