#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

###############################################################################
# theil.py 的等效函数
###############################################################################
def theil(x, y):
    """Theil-Sen 回归：返回截距 a 和斜率 b（中位数斜率）"""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    m = len(x)
    slopes = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            if x[j] != x[i]:  # 防止除零
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    b = np.median(slopes) if len(slopes) > 0 else np.nan
    # a = median of (y - b*x)
    a = np.median(y - b * x)
    return a, b

###############################################################################
# GramSchmidt.R 的等效函数
###############################################################################
def GramSchmidt(A):
    """
    A: numpy 2D array, shape=(nrow, ncol)
    返回 Q, R
    Q 为正交化后的列向量；R 为上三角矩阵
    """
    A = np.array(A, dtype=float)
    nrow, ncol = A.shape
    Q = np.zeros((nrow, ncol))
    R = np.zeros((ncol, ncol))

    # 第1列
    R[0, 0] = np.sqrt(np.dot(A[:, 0], A[:, 0]))
    Q[:, 0] = A[:, 0] / R[0, 0] if R[0, 0] != 0 else A[:, 0]

    # 后续列
    for j in range(1, ncol):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.sqrt(np.dot(v, v))
        if R[j, j] != 0:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v
    return {"Q": Q, "R": R}

###############################################################################
# lag.normalize 对应的函数
# R 里: y<-rev(-diff(rev(x), lag=l))
# 这里手动实现与 R 逻辑完全对应
###############################################################################
def lag_normalize(x, l=12):
    """
    参数 x: 序列(1D)，老数据在前，新数据在后
    逻辑:
      1) tmp = rev(x) -> x[::-1]
      2) d = diff(tmp, lag=l) -> tmp[i] - tmp[i-l]
      3) y = rev(-d)
      4) 标准化 (y - mean) / sd
    """
    arr = np.array(x, dtype=float)
    # 1) reverse
    rev_arr = arr[::-1]
    # 2) diff with lag=l (类似 R 的 diff(..., lag=12))
    #    diff[i] = rev_arr[i] - rev_arr[i-12] for i >= 12
    if len(rev_arr) <= l:
        # 差分后几乎为空
        return np.array([np.nan]*len(arr))
    diff_list = []
    for i in range(l, len(rev_arr)):
        diff_list.append(rev_arr[i] - rev_arr[i - l])
    diff_arr = np.array(diff_list, dtype=float)
    # 3) y = rev(-diff_arr)
    y = -diff_arr[::-1]  # 先 -diff_arr 再 reverse
    # 4) 对 y 做标准化
    #    需要与 R 保持完全一致：只对非NA部分计算 mean / sd
    valid = ~np.isnan(y)
    if np.sum(valid) == 0:
        return np.array([np.nan]*len(arr))
    mu = np.mean(y[valid])
    sigma = np.std(y[valid], ddof=1)  # R 默认是 N-1
    if sigma == 0:
        # 防止 sd=0
        z = (y - mu)
    else:
        z = (y - mu) / sigma
    # 但 y 数组长度比原 arr 短 l 个；R 里是前面被 NA 填充
    # R 中: 复原到和 x 同维度
    # R 代码中最末的 return (y - mu)/sigma 实际也是 length(x) 大小
    # 只是其中前 l 个值会是 NA.
    full = np.array([np.nan]*len(arr))
    # R 里是 y<-rev(-diff(rev(x), lag=l)) 之后 length(y) == length(x)-l
    #     y 前面 l 个 NA
    # 这里 Python 最后 l 个位置填 z(0 ~ end-1)
    # 与 R 保持: 末尾 (length - l) 个位置对应差分值
    # 实际在 R 里 y 的序号: (1 ~ length-l) => 变成( (l+1)~length ) 位置
    full[l:] = z
    return full

###############################################################################
# 主逻辑, 对应 main.R
###############################################################################
def main():
    # --------------------- 1. 初始化 & 参数设定 ---------------------
    TB_only = True
    TB_tag = ""
    input_file = "data\\input_TB_2020.csv"
    n_sec = 6    # sector 0 ~ 5
    n_anno = 5
    lag_month = 12

    # 创建输出文件夹
    os.makedirs('log', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    # --------------------- 2. 读入数据 & TB过滤 ---------------------
    a0 = pd.read_csv(input_file)
    # 假设 CSV 中有列 TB, Region, Sector, Rating 等
    if 'TB' not in a0.columns:
        raise ValueError("input file must have a column named 'TB'")
    TB = a0['TB']
    if TB_only:
        a0 = a0.loc[TB == True].copy()
        TB_tag = "_TB"

    # 注释部分(前 n_anno 列), 时间序列部分(其余)
    a0_anno = a0.iloc[:, :n_anno].copy()  # DataFrame
    a0_ts = a0.iloc[:, n_anno:].copy()    # DataFrame of numeric time series
    # 强制转为 float, 以防混杂字符串
    a0_ts = a0_ts.apply(pd.to_numeric, errors='coerce')

    # --------------------- 3. region 划分 ---------------------
    # region index: 1 => NAM
    # 2 => EU,4 => CEEMEA
    # 3 => JAPAN,5 => ASIA
    # 6 => MEXICO,7 => LATAM
    # i.regs[[1]] => which(Region==1)
    # ...
    i_regs = []
    # R中 i.regs[[1]] 对应Python中 i_regs[0]
    # 这里维持1-based语义时, 下标做减一。
    i_regs.append(np.where(a0['Region'] == 1)[0])  # region1
    i_regs.append(np.where((a0['Region'] == 2)|(a0['Region'] == 4))[0]) # region2+4
    i_regs.append(np.where((a0['Region'] == 3)|(a0['Region'] == 5))[0]) # region3+5
    i_regs.append(np.where((a0['Region'] == 6)|(a0['Region'] == 7))[0]) # region6+7

    # --------------------- 4. sector 0 (global) 构建 ---------------------
    # sector 0 的指数为所有行的均值(每个时间点)
    n_t = a0_ts.shape[1]
    idx0 = np.zeros(n_t, dtype=float)
    n_iss_idx0 = np.zeros(n_t, dtype=int)
    for i in range(n_t):
        col_data = a0_ts.iloc[:, i]
        idx0[i] = np.nanmean(col_data)
        n_iss_idx0[i] = col_data.notna().sum()

    # --------------------- 5. 准备存放 betas 的矩阵 ---------------------
    # betas: nrow(a0) x (n_sec+4)
    # col: b0,b1,b2,b3,b4,b5, intercept, n.issue, R-sq, adj.R-sq
    betas = np.full((a0.shape[0], n_sec+4), np.nan, dtype=float)

    # --------------------- 6. 主循环: 依次遍历 4 个 region ---------------------
    for i_reg_idx in range(4):
        # 当前 region 对应的行 (issue)
        reg_inds = i_regs[i_reg_idx]
        # 从 a0_ts 中取出这些行(issues)对应的 time series
        a = a0_ts.iloc[reg_inds, :].copy()  # shape: (n_issues, n_t)
        a_anno = a0_anno.iloc[reg_inds, :].copy()

        # sector 1~5
        i_secs = []
        for s in range(1, n_sec):  # 1..5
            # a_anno['Sector']==s
            i_secs.append(np.where(a_anno['Sector'].values == s)[0])

        # 构造 sector (0~5) 的指数 idx, shape: (n_t, n_sec)
        idx = np.zeros((n_t, n_sec), dtype=float) * np.nan
        n_iss_idx = np.zeros((n_t, n_sec), dtype=int)

        # sector 0 全局指数, 先插入
        idx[:, 0] = idx0
        n_iss_idx[:, 0] = n_iss_idx0

        # sector 1..5
        for i_col in range(n_t):
            for j_sec in range(1, n_sec):
                rows_j = i_secs[j_sec - 1]  # 该sector内对应的行号(相对当前region)
                col_data = a.iloc[rows_j, i_col]  # 这些行, 第 i_col 列
                idx[i_col, j_sec] = np.nanmean(col_data)
                n_iss_idx[i_col, j_sec] = col_data.notna().sum()

        # 处理 NA: 如果某个 sector 列有 NA, 用 sector0 做回归插补
        # R中: na.sectors<-which(apply(idx,2,function(x) any(is.na(x))))
        na_sectors = []
        for sec_col in range(n_sec):
            if np.isnan(idx[:, sec_col]).any():
                na_sectors.append(sec_col)

        if len(na_sectors) > 0:
            for jk in na_sectors:
                # 找到 idx[:, jk] 中NA的位置
                jkk = np.where(np.isnan(idx[:, jk]))[0]
                # lm(idx[,jk] ~ idx[,1]) => Python中:
                # X = idx[:,0], Y= idx[:,jk]
                # 线性回归: y = a + b*x
                good_mask = ~np.isnan(idx[:, jk]) & ~np.isnan(idx[:, 0])
                if np.sum(good_mask) > 1:
                    X_good = idx[good_mask, 0]
                    Y_good = idx[good_mask, jk]
                    A = np.column_stack((np.ones(len(X_good)), X_good))
                    # 最小二乘
                    coef, _, _, _ = np.linalg.lstsq(A, Y_good, rcond=None)
                    a_, b_ = coef[0], coef[1]
                    # 插补
                    idx[jkk, jk] = a_ + b_ * idx[jkk, 0]
                else:
                    # 如果都没法回归，就先维持 NA
                    pass

        # 写出 ZidxA
        df_idxA = pd.DataFrame(idx, columns=[f"I{k}" for k in range(n_sec)])
        df_idxA.to_csv(f"log\\_ZidxA{i_reg_idx+1}.csv", index=False)

        # differencing & normalizing (lag.normalize)
        # 在 R 中: idx<-apply(idx, 2, lag.normalize, lag.month)
        # => 对 idx 的每一列做 lag_normalize
        idx_dn = []
        for col_i in range(idx.shape[1]):
            col_trans = lag_normalize(idx[:, col_i], lag_month)
            idx_dn.append(col_trans)
        idx_dn = np.array(idx_dn).T  # 转置回来 shape: (n_t, n_sec)

        # 写到 idx (覆盖)
        idx = idx_dn
        col_idx = [f"dI{k}" for k in range(n_sec)]
        df_idxB = pd.DataFrame(idx, columns=col_idx)
        df_idxB.to_csv(f"log\\_ZidxB{i_reg_idx+1}.csv", index=False)

        # 删除含 NA 行 => i.ex
        # R中: i.ex<-which(apply(idx,1,function(x) any(is.na(x))))
        mask_na = np.any(np.isnan(idx), axis=1)
        i_ex = np.where(mask_na)[0]

        if len(i_ex) > 0:
            idx_clean = np.delete(idx, i_ex, axis=0)
            n_iss_idx_clean = np.delete(n_iss_idx, i_ex, axis=0)
        else:
            idx_clean = idx
            n_iss_idx_clean = n_iss_idx

        # GramSchmidt
        Fi = GramSchmidt(idx_clean)
        Q = Fi["Q"]  # shape: (nrow, ncol)
        # vb = Fi$Q*sqrt(nrow(idx)-1)
        vb = Q * np.sqrt(idx_clean.shape[0] - 1)

        # 将 vb 填回到跟原 idx 同维度位置
        vb0 = np.full((idx.shape[0], idx.shape[1]), np.nan, dtype=float)
        if len(i_ex) > 0:
            good_inds = np.setdiff1d(np.arange(idx.shape[0]), i_ex)
            vb0[good_inds, :] = vb
        else:
            vb0 = vb

        df_vb = pd.DataFrame(vb0, columns=[f"F{k}" for k in range(n_sec)])
        df_vb.to_csv(f"log\\_Zvb{i_reg_idx+1}.csv", index=False)

        # pairs(...)作图部分忽略，这里只保留核心逻辑。

        # 对 a (issues) 也做 lag.normalize
        # R: a<-t(apply(a,1,lag.normalize,lag.month))
        # a 是 shape: (n_issue, n_t)
        # apply(a,1,...) => 每一行 => Python中 axis=1 => 只好循环
        a_matrix = a.to_numpy(dtype=float)
        a_dn = []
        for row_i in range(a_matrix.shape[0]):
            row_trans = lag_normalize(a_matrix[row_i, :], lag_month)
            a_dn.append(row_trans)
        a_dn = np.array(a_dn)  # shape: (n_issue, n_t)

        df_a = pd.DataFrame(a_dn)
        df_a.to_csv(f"log\\_Za{i_reg_idx+1}.csv", index=False)

        # 同样删除 i.ex 列
        # R: a<-a[,-i.ex]
        if len(i_ex) > 0:
            a_clean = np.delete(a_dn, i_ex, axis=1)
            # 记下列名
            colnames_original = a0_ts.columns
            used_colnames = colnames_original[lag_month:]  # R: names(a0.ts)[-(1:lag.month)]
            used_colnames = used_colnames.drop(used_colnames[i_ex])  # Python 索引
        else:
            a_clean = a_dn
            used_colnames = a0_ts.columns[lag_month:]

        # 现在做回归 x ~ vb (正交因子)
        n_issues = a_clean.shape[0]
        # 这里 region i_reg_idx 的 row 索引 => reg_inds
        for i_issue_local in range(n_issues):
            x = a_clean[i_issue_local, :]
            ii = reg_inds[i_issue_local]  # 在原 a0 中的行号

            # 只用非 NA 的时间点
            valid_mask = ~np.isnan(x)
            if np.sum(valid_mask) > 10:  # >10 才做回归
                X_fac = vb[valid_mask, :]  # shape: (k, n_sec)
                y = x[valid_mask]          # shape: (k,)

                # 线性回归 y ~ X_fac
                # lm => y = c0 + c1*F1 + ... c6*F6
                # 手动添加常数项
                X_design = np.column_stack((np.ones(X_fac.shape[0]), X_fac))
                coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                # coefs[0] = intercept, coefs[1..] = 与每个 factor 对应的系数

                # 回归统计量
                y_hat = X_design @ coefs
                resid = y - y_hat
                ss_res = np.sum(resid**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot!=0 else np.nan
                # adjusted R2
                df_reg = X_fac.shape[1]  # = n_sec
                n_obs = X_fac.shape[0]
                if n_obs - df_reg - 1 > 0:
                    adj_r2 = 1 - (1-r2)*(n_obs-1)/(n_obs-df_reg-1)
                else:
                    adj_r2 = np.nan

                # 缩放因子 beta.SF
                # beta.SF<-apply(vb[k,], 2, function(x) sqrt(sum(x*x)))
                # Python: 这里 X_fac == vb[valid_mask,:]
                # => 每个列 factor_j, sqrt(sum(factor_j^2))
                factor_norm = np.sqrt(np.sum(X_fac**2, axis=0))
                # beta.SF<-beta.SF/sd(x[k])/sqrt(length(k)-1)
                x_sd = np.std(y, ddof=1)
                denom = x_sd * np.sqrt(np.sum(valid_mask)-1)
                if denom == 0:
                    beta_sf = np.zeros(n_sec)
                else:
                    beta_sf = factor_norm / denom

                # l$coef[-1] => coefs[1..]
                b_original = coefs[1:]
                b_scaled = b_original * beta_sf

                # 写入 betas[ii,1:n.sec]
                betas[ii, :n_sec] = b_scaled
                betas[ii, n_sec]   = coefs[0]    # intercept
                betas[ii, n_sec+1] = np.sum(valid_mask)  # n.issue
                betas[ii, n_sec+2] = r2
                betas[ii, n_sec+3] = adj_r2

    # --------------------- 7. 后处理 & 输出 ---------------------
    col_betas = [f"b{k}" for k in range(n_sec)] + ["intercept","n.issue","R-sq","adj.R-sq"]
    df_betas = pd.DataFrame(betas, columns=col_betas)
    # 计算 rho = rowSums of b0^2 + b1^2 + ...
    b_sqr = np.sum(df_betas.iloc[:, :n_sec].values**2, axis=1)
    df_betas["rho"] = b_sqr
    good_rho_mask = (~np.isnan(b_sqr)) & (b_sqr <= 1)
    df_betas["good.rho"] = good_rho_mask

    # 最终 output
    output = pd.concat([a0_anno.reset_index(drop=True), df_betas], axis=1)

    # 写出 betas_all_lag_
    output.to_csv(f"result\\betas_all_lag_{lag_month}{TB_tag}.csv", index=False)

    # 计算 pairwise correlation(可忽略) & 分桶统计
    # R中: betas<-as.matrix(output[jj, 6:11])
    # 这里 columns: a0_anno(5列)=> indices 0..4, betas 0..(n_sec+3)=6..(6+5)=11??
    # 实际 col 索引: a0_anno (width= n_anno=5), betas( n_sec+4=10 ) => range(5..14)
    # => b0..b5 => output.columns[5..10], intercept=>11, n.issue=>12, ...
    # 下面与 R 做同样的选择:
    betas_matrix = output.loc[output["good.rho"]==True, col_betas[:n_sec]].to_numpy()
    if betas_matrix.size>0:
        rho_eff = betas_matrix @ betas_matrix.T
        # print 平均值/标准差
        up_tri = rho_eff[np.triu_indices_from(rho_eff)]
        print(np.mean(up_tri), np.std(up_tri))

    # ------------------ avg.beta / sd.beta ------------------
    # R中: for(i.reg in 0:4) ...
    # 这里 region 有 1..4 => 代码中: i_regs[0..3], 但输出中还包含 region=0(全局)
    # sector=0..5, rating=0..7
    # R中通过 parse(text=...) 动态构造表达式, 这里直接用逻辑筛选
    # output 列: [Region, Sector, Rating, ..., good.rho, ... betas ...]
    # a0_anno 的列顺序是(假设):
    #  1) Type, 2) ID, 3) Region, 4) Sector, 5) Rating (示例)
    # 实际需根据 input_TB_2020.csv 的真实列顺序进行核对
    # 假设 Region=3, Sector=4, Rating=5 => 需根据实际 CSV 内顺序来对号, 这里仅做示例:
    region_col = "Region"
    sector_col = "Sector"
    rating_col = "Rating"

    # 构造 DataFrame 以存放 avg/sd 结果
    col_names_res = ["Region","Sector","Rating","Count"] + [f"b{k}" for k in range(n_sec)]
    avg_rows = []
    sd_rows = []

    for i_reg in range(0,5):  # 0..4
        if i_reg==0:
            output1 = output  # 全局
        else:
            # region i_reg
            output1 = output[output[region_col] == i_reg]
        for i_sec in range(0,6):
            for i_rat in range(0,8):
                # 筛选: good.rho==True
                sub_df = output1[output1["good.rho"]==True]
                if i_sec>0:
                    sub_df = sub_df[sub_df[sector_col]==i_sec]
                if i_rat>0:
                    sub_df = sub_df[sub_df[rating_col]==i_rat]
                count_i = len(sub_df)
                if count_i>0:
                    # 取 output1 的列 6:11 => b0..b5, 这里需要对应 Python col
                    # col_betas[:n_sec] = b0..b5
                    # sub_df[col_betas[:n_sec]]
                    b_mat = sub_df.loc[:, col_betas[:n_sec]]
                    b_mean = b_mat.mean(axis=0).round(4).to_list()
                    b_sd   = b_mat.std(axis=0, ddof=1).round(4).to_list()
                    tmp_avg = [i_reg,i_sec,i_rat,count_i] + b_mean
                    tmp_sd  = [i_reg,i_sec,i_rat,count_i] + b_sd
                else:
                    tmp_avg = [i_reg, i_sec, i_rat, 0] + [np.nan]*n_sec
                    tmp_sd  = [i_reg, i_sec, i_rat, 0] + [np.nan]*n_sec
                avg_rows.append(tmp_avg)
                sd_rows.append(tmp_sd)

    df_avg = pd.DataFrame(avg_rows, columns=col_names_res)
    df_sd = pd.DataFrame(sd_rows, columns=col_names_res)

    # ------------------ empty.buckets 填充逻辑 ------------------
    # for(k in empty.buckets)
    #  region=R, sector=S, rating=T
    #  -> region=R, sector=S, rating=0
    #  -> region=R, sector=0, rating=0
    #  -> region=0, sector=0, rating=0
    #  只针对 Count<5
    def fill_bucket(row_idx, df_main):
        # row 的 Region/Sector/Rating
        R_ = df_main.at[row_idx,"Region"]
        S_ = df_main.at[row_idx,"Sector"]
        T_ = df_main.at[row_idx,"Rating"]
        # 先找 region=R_, sector=S_, rating=0
        cond1 = (df_main["Region"]==R_)&(df_main["Sector"]==S_)&(df_main["Rating"]==0)&(df_main["Count"]>=5)
        cands = df_main[cond1]
        if len(cands)>0:
            return cands.iloc[0,4:].values  # b0..b5
        # 再 region=R_, sector=0, rating=0
        cond2 = (df_main["Region"]==R_)&(df_main["Sector"]==0)&(df_main["Rating"]==0)&(df_main["Count"]>=5)
        cands = df_main[cond2]
        if len(cands)>0:
            return cands.iloc[0,4:].values
        # 最后 region=0, sector=0, rating=0
        cond3 = (df_main["Region"]==0)&(df_main["Sector"]==0)&(df_main["Rating"]==0)&(df_main["Count"]>=5)
        cands = df_main[cond3]
        if len(cands)>0:
            return cands.iloc[0,4:].values
        return None

    def bucket_fill_inplace(df_main):
        for i in range(len(df_main)):
            if df_main.at[i,"Count"]<5:
                new_vals = fill_bucket(i, df_main)
                if new_vals is not None:
                    df_main.iloc[i, 4:] = new_vals

    bucket_fill_inplace(df_avg)
    bucket_fill_inplace(df_sd)

    # ------------------ 写出结果 CSV ------------------
    output.to_csv(f"result\\betas_all_lag_{lag_month}{TB_tag}.csv", index=False)
    df_avg.to_csv(f"result\\betas_mean_lag_{lag_month}{TB_tag}.csv", index=False)
    df_sd.to_csv(f"result\\betas_sd_lag_{lag_month}{TB_tag}.csv", index=False)

    print("All done. Outputs are in 'log\\' and 'result\\' folders.")

# 运行主函数
if __name__ == "__main__":
    main()