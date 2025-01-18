def lag_normalize(x, l=12):
    """
    严格模仿 R 版：
      y = rev(-diff(rev(x), lag=l))
    其长度 = len(x) - l
    不做任何补 NA 的操作，直接返回长度缩短的序列
    """
    arr = np.array(x, dtype=float)
    if len(arr) <= l:
        # R 里如果 x 长度不够 diff，就会得到 length=0 的结果
        return np.array([], dtype=float)

    # 1) rev(x) => arr[::-1]
    rev_arr = arr[::-1]
    # 2) diff(..., lag=l)
    diff_list = []
    for i in range(l, len(rev_arr)):
        diff_list.append(rev_arr[i] - rev_arr[i - l])
    diff_arr = np.array(diff_list, dtype=float)
    # 3) y = rev(-diff_arr)
    y = -diff_arr[::-1]

    # 4) 标准化
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.full_like(y, np.nan)

    mu = np.nanmean(y[valid])
    sigma = np.nanstd(y[valid], ddof=1)  # R 中 sd() 默认是 N-1
    if sigma == 0:
        z = y - mu
    else:
        z = (y - mu) / sigma

    return z  # 注意返回长度 = len(x) - l


def main():
    # 假设 idx.shape = (n_t, n_sec)，先存 n_t:
    n_t_orig = idx.shape[0]

    # 对每列做 lag_normalize
    result_list = []
    for col_i in range(idx.shape[1]):
        col_in = idx[:, col_i]
        col_out = lag_normalize(col_in, lag_month)
        result_list.append(col_out)

    # result_list 里每个元素都长 = n_t_orig - lag_month
    # 组装成 (n_t_orig-lag_month) x n_sec
    idx_dn = np.column_stack(result_list)

    # 更新 n.t
    n_t = n_t_orig - lag_month

    # 写 csv
    df_idxB = pd.DataFrame(idx_dn, columns=[f"dI{k}" for k in range(n_sec)])
    df_idxB.to_csv(f"log\\_ZidxB{i_reg_idx+1}.csv", index=False)