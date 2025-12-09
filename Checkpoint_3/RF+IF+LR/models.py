"""
Простой табличный ML на HDFS_v1/preprocessed
Фичи: счётчики E1...E29, sequence_len, Latency, агрегаты TimeInterval (ti_mean/max/std/zeros)
Модели: RandomForest, IsolationForest, LogisticRegression
Метрики: ROC-AUC, PR-AUC (average precision), confusion matrix, precision/recall/F1

Запуск: `python models.py` из папки, рядом с HDFS_v1/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, \
    precision_recall_curve, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_RECALL = 0.99
DATA = Path('HDFS_v1') / 'preprocessed'
OUT = Path('out')
OUT.mkdir(exist_ok=True, parents=True)


def find_threshold_for_recall(y_true, scores, target_recall: float) -> float:
    """Выбирает максимальный порог, при котором recall >= target_recall"""
    prec, rec, th = precision_recall_curve(y_true, scores)
    m = rec[:-1] >= target_recall
    return th[m].max() if m.any() else th.min()

def collect_summary(name, y_true, y_pred, scores=None, threshold=None) -> dict:
    """Собирает ключевые метрики для сравнения моделей"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    roc  = np.nan if scores is None else roc_auc_score(y_true, scores)
    pr   = np.nan if scores is None else average_precision_score(y_true, scores)
    return {
        "Model": name,
        "Threshold": threshold,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

# ============== Загрузка данных ==============
labels = pd.read_csv(DATA / 'anomaly_label.csv')
occurrence = pd.read_csv(DATA / 'Event_occurrence_matrix.csv')
traces = pd.read_csv(DATA / 'Event_traces.csv')
templates = pd.read_csv(DATA / 'HDFS.log_templates.csv')

labels_map = {'Normal': 0, 'Anomaly': 1}
labels['y'] = labels['Label'].map(labels_map).astype(int)

# ============== Фичи ==============
def calc_seq_len(s: str) -> int:
    """Расчет длины последовательности событий"""
    return len(s.split(','))

def parse_ti(s: str) -> list[float]:
    """Приведение типа TimeInterval: str -> list[float]"""
    arr = s[1:-1].split(', ')
    return [float(x) for x in arr]

def ti_stats(s: str):
    """Расчет агрегатов по TimeInterval"""
    a = np.array(parse_ti(s), dtype=float)
    return pd.Series({
        'ti_mean': a.mean(),
        'ti_max': a.max(),
        'ti_std': a.std(ddof=0),
        'ti_zeros': (a==0).mean()
    })
traces_simple = traces[['BlockId', 'Features', 'TimeInterval', 'Latency']].copy()
traces_simple['sequence_len'] = traces_simple['Features'].map(calc_seq_len)
ti_df = traces_simple['TimeInterval'].apply(ti_stats)
traces_simple = pd.concat([traces_simple[['BlockId', 'Latency', 'sequence_len']], ti_df], axis=1)

event_cols = [c for c in occurrence.columns if c.startswith('E')]
df = (occurrence[['BlockId'] + event_cols]
      .merge(traces_simple, on='BlockId', how='inner')
      .merge(labels[['BlockId','y']], on='BlockId', how='inner'))

X = df[event_cols + ['sequence_len','Latency','ti_mean','ti_max','ti_std','ti_zeros']].fillna(0.0)
y = df['y'].values

print(df.head())

# ============== Сплиты traces_simple / valid (для подбора порога) / test ==============
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)
print(f'\nРазмеры: train={len(X_train)}, valid={len(X_val)}, test={len(X_test)}')

# ============== RandomForest ==============
model_rf = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight='balanced_subsample'
)
model_rf.fit(X_train, y_train)

# Подбор порога на valid под целевой recall
val_proba_rf = model_rf.predict_proba(X_val)[:, 1]
threshold_rf = find_threshold_for_recall(y_val, val_proba_rf, TARGET_RECALL)
val_pred_rf = (val_proba_rf >= threshold_rf).astype(int)

# Оценка на тесте
test_proba_rf = model_rf.predict_proba(X_test)[:, 1]
test_pred_rf = (test_proba_rf >= threshold_rf).astype(int)

# Важность признаков
importances_rf = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances_rf.head(15).to_csv(OUT / 'rf_feature_importances_top15.csv')

# ============== IsolationForest ==============
X_train_norm = X_train[y_train == 0]
model_iso = IsolationForest(
    n_estimators=300,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
model_iso.fit(X_train_norm)

# Считаем степень аномальности
val_score_if = -model_iso.score_samples(X_val)
test_score_if = -model_iso.score_samples(X_test)

# Подбор порога на valid под целевой recall
threshold_if = find_threshold_for_recall(y_val, val_score_if, TARGET_RECALL)
val_pred_if = (val_score_if >= threshold_if).astype(int)

# Оценка на тесте
test_pred_if = (test_score_if >= threshold_if).astype(int)

# ============== LogisticRegression ==============
lr = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=False)),
    ('lr', LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ))
])
lr.fit(X_train, y_train)

# Подбор порога на valid под целевой recall
val_proba_lr = lr.predict_proba(X_val)[:,1]
threshold_lr = find_threshold_for_recall(y_val, val_proba_lr, TARGET_RECALL)
val_pred_lr = (val_proba_lr >= threshold_lr).astype(int)

# Оценка на тесте
test_proba_lr = lr.predict_proba(X_test)[:,1]
test_pred_lr = (test_proba_lr >= threshold_lr).astype(int)

importances_lr = pd.Series(lr.named_steps['lr'].coef_.ravel(), index=X.columns)
importances_lr.head(15).to_csv(OUT / 'lr_feature_importances_top15.csv')

# ============== Сравнение моделей на test ==============
summary_rows = [
    collect_summary('RandomForest', y_test, test_pred_rf, scores=test_proba_rf, threshold=threshold_rf),
    collect_summary('IsolationForest', y_test, test_pred_if, scores=test_score_if, threshold=threshold_if),
    collect_summary("LogReg", y_test, test_pred_lr, scores=test_proba_lr, threshold=threshold_lr)
]
summary_df = pd.DataFrame(summary_rows)

cols = ['Model','Threshold','ROC_AUC','PR_AUC','Precision','Recall','F1','FP','FN','TN','TP','Accuracy']
print('\n======= Сравнение моделей (test) =======')
print(summary_df[cols].round(4).to_string(index=False))
summary_df.to_csv(OUT / 'models_comparison.csv', index=False)
