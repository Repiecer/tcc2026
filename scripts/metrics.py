"""统一的预测技巧评价指标（规划书 06 第一节）。

只做一件事：给一对 (真实值 y, 预测值 yhat)，算出所有指标。
后面的 L0/L1 基线、L2 线性、L3 随机森林、L4 XGBoost 全部复用这里，
保证不同模型之间可比。

四个核心指标
------------
RMSE  均方根误差 = sqrt(mean((yhat - y)^2))
      直观含义："平均而言预测偏了几个单位"。对**大误差特别敏感**（平方放大）。
MAE   平均绝对误差 = mean(|yhat - y|)
      和 RMSE 同类，但**更稳健**（不被个别大误差主导）。
R     皮尔逊相关系数 = cov(y, yhat) / (σy · σyhat)
      只看"起伏是否同步"，**完全不看偏差和量级**。
R²    决定系数 = 1 - SS_res / SS_tot
      回答"解释了目标波动的百分之多少"。**可以为负**。

附加指标
--------
Bias        平均偏差 = mean(yhat - y)，正 = 系统性高估
sigma_ratio 预测值标准差 / 真实值标准差，< 1 说明预测过于平滑
Skill Score 1 - MSE_model / MSE_ref，> 0 才说明比参照基线好

用法：
    from metrics import evaluate, bootstrap_ci, skill_score

    m = evaluate(y, yhat)
    ci = bootstrap_ci(y, yhat, years)      # 年分块 bootstrap（自相关数据必须用）
    ss = skill_score(m["RMSE"], m_ref["RMSE"])
"""

import numpy as np
import pandas as pd

CI_KEYS = ("R", "R2", "RMSE", "MAE")


def evaluate(y, yhat):
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    if y.shape != yhat.shape:
        raise ValueError("y 与 yhat 长度不一致")

    e = yhat - y
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if y.std() > 0 and yhat.std() > 0:
        R = float(np.corrcoef(y, yhat)[0, 1])
    else:
        R = float("nan")

    return dict(
        n=int(len(y)),
        R=R,
        R2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        RMSE=float(np.sqrt(np.mean(e ** 2))),
        MAE=float(np.mean(np.abs(e))),
        Bias=float(np.mean(e)),
        sigma_ratio=float(yhat.std() / y.std()) if y.std() > 0 else float("nan"),
    )


def skill_score(mse_model, mse_ref):
    return 1.0 - mse_model / mse_ref


def bootstrap_ci(y, yhat, years, n_boot=2000, alpha=0.05, seed=0):
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    years = np.asarray(years)
    uniq = np.unique(years)
    if len(uniq) < 3:
        return {k: (float("nan"), float("nan")) for k in CI_KEYS}

    rng = np.random.default_rng(seed)
    pos = {yy: np.where(years == yy)[0] for yy in uniq}
    draws = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([pos[yy] for yy in pick])
        m = evaluate(y[idx], yhat[idx])
        for k in CI_KEYS:
            draws[k].append(m[k])

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
            for k, v in draws.items()}


def format_ci(ci):
    return "[%+.3f, %+.3f]" % ci if np.isfinite(ci).all() else "[  --  ,   --  ]"
def results_frame(rows):
    out = []
    for name, split, lead, m, ci, ss in rows:
        rec = dict(lead_week=lead, split=split, baseline=name)
        rec.update({k: v for k, v in m.items()})
        rec["SS_vs_L0"] = ss
        for k in CI_KEYS:
            rec[f"{k}_lo"], rec[f"{k}_hi"] = ci.get(k, (np.nan, np.nan))
        out.append(rec)
    return pd.DataFrame(out)
