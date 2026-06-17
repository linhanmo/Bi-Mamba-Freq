import math
from typing import Dict


def _to_numpy(x):
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        return np.asarray(x)
    except Exception:
        return x


def forecast_metrics(pred, target) -> Dict[str, float]:
    pred = _to_numpy(pred)
    target = _to_numpy(target)
    diff = pred - target
    mse = float((diff * diff).mean())
    mae = float(abs(diff).mean())
    return {"mse": mse, "mae": mae}


def regression_metrics(pred, target) -> Dict[str, float]:
    pred = _to_numpy(pred).reshape(-1)
    target = _to_numpy(target).reshape(-1)
    n = len(pred)
    mse = float(((pred - target) ** 2).mean())
    mae = float(abs(pred - target).mean())
    rmse = math.sqrt(mse)
    mean_target = float(target.mean()) if n > 0 else 0.0
    ss_tot = float(((target - mean_target) ** 2).sum())
    ss_res = float(((target - pred) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def _binary_confusion(y_true, y_pred):
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = _to_numpy(y_pred).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _binary_auc(y_true, y_score) -> float:
    y_true = _to_numpy(y_true).reshape(-1)
    y_score = _to_numpy(y_score).reshape(-1)
    positives = float((y_true == 1).sum())
    negatives = float((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = y_score.argsort()
    ranks = order.argsort() + 1
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def classification_metrics(y_prob, y_true, threshold: float = 0.5) -> Dict[str, float]:
    y_prob = _to_numpy(y_prob).reshape(-1)
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = _binary_confusion(y_true, y_pred)
    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    mcc_num = float(tp * tn - fp * fn)
    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0
    auc = _binary_auc(y_true, y_prob)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
        "mcc": float(mcc),
    }


def multitask_finance_metrics(cls_prob, cls_true, vol_pred, vol_true) -> Dict[str, float]:
    cls = classification_metrics(cls_prob, cls_true)
    reg = regression_metrics(vol_pred, vol_true)
    rmse_norm = reg["rmse"] / max(abs(float(_to_numpy(vol_true).reshape(-1).mean())), 1e-8)
    combined = 0.5 * cls["f1"] + 0.5 * (1.0 - rmse_norm)
    return {
        **cls,
        **reg,
        "combined_score": float(combined),
    }
