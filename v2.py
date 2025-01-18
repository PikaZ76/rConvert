#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

def theil(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    slopes = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i])/(x[j] - x[i]))
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
    for j in range(1, ncol):
        v = A[:,j].copy()
        for i in range(j):
            R[i,j] = np.dot(Q[:,i], A[:,j])
            v -= R[i,j]*Q[:,i]
        R[j,j] = np.sqrt(np.dot(v,v))
        if R[j,j]!=0: Q[:,j] = v/R[j,j]
    return {"Q": Q, "R": R}

def lag_normalize(x, l=12):
    arr = np.array(x, dtype=float)
    if len(arr) <= l:
        return np.array([], dtype=float)
    rev_arr = arr[::-1]
    diff_list = []
    for i in range(l, len(rev_arr)):
        diff_list.append(rev_arr[i] - rev_arr[i - l])
    diff_arr = np.array(diff_list)
    y = -diff_arr[::-1]
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.full_like(y, np.nan)
    mu = np.nanmean(y[valid])
    sigma = np.nanstd(y[valid], ddof=1)
    z = (y - mu)/sigma if sigma!=0 else (y - mu)
    return z

def main():
    TB_only = True
    TB_tag = ""
    input_file = "data\\input_TB_2020.csv"
    n_sec = 6
    n_anno = 5
    lag_month = 12
    os.makedirs('log', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    a0 = pd.read_csv(input_file)
    TB = a0['TB']
    if TB_only:
        a0 = a0.loc[TB==True].copy()
        TB_tag = "_TB"

    a0_anno = a0.iloc[:, :n_anno].copy()
    a0_ts = a0.iloc[:, n_anno:].copy().apply(pd.to_numeric, errors='coerce')

    i_regs = []
    i_regs.append(np.where(a0['Region']==1)[0])               # region1
    i_regs.append(np.where((a0['Region']==2)|(a0['Region']==4))[0])
    i_regs.append(np.where((a0['Region']==3)|(a0['Region']==5))[0])
    i_regs.append(np.where((a0['Region']==6)|(a0['Region']==7))[0])

    n_t = a0_ts.shape[1]
    idx0 = np.zeros(n_t, dtype=float)
    n_iss_idx0 = np.zeros(n_t, dtype=int)
    for i in range(n_t):
        col_data = a0_ts.iloc[:, i]
        idx0[i] = np.nanmean(col_data)
        n_iss_idx0[i] = col_data.notna().sum()

    betas = np.full((a0.shape[0], n_sec+4), np.nan, dtype=float)

    for i_reg_idx in range(4):
        reg_inds = i_regs[i_reg_idx]
        a = a0_ts.iloc[reg_inds, :].copy()
        a_anno = a0_anno.iloc[reg_inds, :].copy()

        i_secs = []
        for s in range(1, n_sec):
            i_secs.append(np.where(a_anno['Sector'].values==s)[0])

        idx = np.full((n_t, n_sec), np.nan, dtype=float)
        n_iss_idx = np.zeros((n_t, n_sec), dtype=int)
        idx[:, 0] = idx0
        n_iss_idx[:, 0] = n_iss_idx0

        for i_col in range(n_t):
            for j_sec in range(1, n_sec):
                rows_j = i_secs[j_sec-1]
                col_data = a.iloc[rows_j, i_col]
                idx[i_col, j_sec] = np.nanmean(col_data)
                n_iss_idx[i_col, j_sec] = col_data.notna().sum()

        na_sectors = []
        for sec_col in range(n_sec):
            if np.isnan(idx[:, sec_col]).any():
                na_sectors.append(sec_col)
        if len(na_sectors)>0:
            for jk in na_sectors:
                jkk = np.where(np.isnan(idx[:, jk]))[0]
                good_mask = ~np.isnan(idx[:, jk]) & ~np.isnan(idx[:, 0])
                if np.sum(good_mask)>1:
                    X_good = idx[good_mask, 0]
                    Y_good = idx[good_mask, jk]
                    A_ = np.column_stack((np.ones(len(X_good)), X_good))
                    coef,_,_,_ = np.linalg.lstsq(A_, Y_good, rcond=None)
                    a_, b_ = coef[0], coef[1]
                    idx[jkk, jk] = a_ + b_*idx[jkk, 0]

        df_idxA = pd.DataFrame(idx, columns=[f"I{k}" for k in range(n_sec)])
        df_idxA.to_csv(f"log\\_ZidxA{i_reg_idx+1}.csv", index=False)

        # 差分+标准化
        old_n_t = idx.shape[0]
        result_list = []
        for col_i in range(idx.shape[1]):
            col_out = lag_normalize(idx[:, col_i], lag_month)
            result_list.append(col_out)
        idx_dn = np.column_stack(result_list)
        n_t_new = old_n_t - lag_month

        df_idxB = pd.DataFrame(idx_dn, columns=[f"dI{k}" for k in range(n_sec)])
        df_idxB.to_csv(f"log\\_ZidxB{i_reg_idx+1}.csv", index=False)

        # 覆盖 idx，后续都用“已差分”的 idx_dn
        idx = idx_dn
        n_t = n_t_new

        # 删除 NA 行
        mask_na = np.any(np.isnan(idx), axis=1)
        i_ex = np.where(mask_na)[0]
        if len(i_ex)>0:
            idx_clean = np.delete(idx, i_ex, axis=0)
            n_iss_idx_clean = np.delete(n_iss_idx, i_ex, axis=0)
        else:
            idx_clean = idx
            n_iss_idx_clean = n_iss_idx

        Fi = GramSchmidt(idx_clean)
        Q = Fi["Q"]
        vb = Q*np.sqrt(idx_clean.shape[0]-1)

        vb0 = np.full((idx.shape[0], idx.shape[1]), np.nan, dtype=float)
        if len(i_ex)>0:
            good_inds = np.setdiff1d(np.arange(idx.shape[0]), i_ex)
            vb0[good_inds,:] = vb
        else:
            vb0 = vb
        df_vb = pd.DataFrame(vb0, columns=[f"F{k}" for k in range(n_sec)])
        df_vb.to_csv(f"log\\_Zvb{i_reg_idx+1}.csv", index=False)

        a_matrix = a.to_numpy(dtype=float)
        rows_list = []
        for row_i in range(a_matrix.shape[0]):
            x_in = a_matrix[row_i, :]
            x_out = lag_normalize(x_in, lag_month)
            rows_list.append(x_out)
        a_dn = np.row_stack(rows_list)
        df_a = pd.DataFrame(a_dn)
        df_a.to_csv(f"log\\_Za{i_reg_idx+1}.csv", index=False)

        if len(i_ex)>0:
            a_clean = np.delete(a_dn, i_ex, axis=1)
        else:
            a_clean = a_dn

        n_issues = a_clean.shape[0]
        for i_issue_local in range(n_issues):
            x = a_clean[i_issue_local, :]
            ii = reg_inds[i_issue_local]
            valid_mask = ~np.isnan(x)
            if np.sum(valid_mask)>10:
                X_fac = vb[valid_mask,:]
                y = x[valid_mask]
                X_design = np.column_stack((np.ones(X_fac.shape[0]), X_fac))
                coefs,_,_,_ = np.linalg.lstsq(X_design, y, rcond=None)
                y_hat = X_design@coefs
                resid = y - y_hat
                ss_res = np.sum(resid**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r2 = 1-ss_res/ss_tot if ss_tot!=0 else np.nan
                df_reg = X_fac.shape[1]
                n_obs = X_fac.shape[0]
                if n_obs-df_reg-1>0:
                    adj_r2 = 1-(1-r2)*(n_obs-1)/(n_obs-df_reg-1)
                else:
                    adj_r2 = np.nan

                factor_norm = np.sqrt(np.sum(X_fac**2, axis=0))
                x_sd = np.std(y, ddof=1)
                denom = x_sd*np.sqrt(np.sum(valid_mask)-1)
                beta_sf = factor_norm/denom if denom!=0 else np.zeros(n_sec)
                b_original = coefs[1:]
                b_scaled = b_original*beta_sf
                betas[ii, :n_sec] = b_scaled
                betas[ii, n_sec] = coefs[0]
                betas[ii, n_sec+1] = np.sum(valid_mask)
                betas[ii, n_sec+2] = r2
                betas[ii, n_sec+3] = adj_r2

    col_betas = [f"b{k}" for k in range(n_sec)] + ["intercept","n.issue","R-sq","adj.R-sq"]
    df_betas = pd.DataFrame(betas, columns=col_betas)
    b_sqr = np.sum(df_betas.iloc[:,:n_sec].values**2, axis=1)
    df_betas["rho"] = b_sqr
    good_rho_mask = (~np.isnan(b_sqr))&(b_sqr<=1)
    df_betas["good.rho"] = good_rho_mask
    output = pd.concat([a0_anno.reset_index(drop=True), df_betas], axis=1)
    output.to_csv(f"result\\betas_all_lag_{lag_month}{TB_tag}.csv", index=False)
    # ... 如需继续计算 avg.beta / sd.beta 可在此后再写

if __name__=="__main__":
    main()