"""
Klasicni regresor 
MultiOutputRegressor
✅ SVR
"""


"""
multi-label 39 izlaza → top-7
feature engineering: lag(5), rolling freq (20/50/100), gap, statistike
vremenski split, poslednjih 100 za back-test
predikcija iz poslednjeg reda CSV-a
validacija 7 jedinstvenih, 1-39, sortirano
back-test hits/7, hit%, AUC, LRAP
snimanje u Regresor_SVR_v2_predikcija.txt
vreme start/stop/elapsed
SVR podešen: RBF, C=2.0, epsilon=0.05, cache_size=1000, n_jobs=1

mapirani:
za svaki red iz poslednjih 100 kola pozicioni SVR predvidi 7 brojeva
iz tih 7 brojeva pravim skor po broju 1..39 = -min_distance_do_najbližeg_predviđenog
top-7 iz tog skora daje hits/7
isti skor ide u AUC i LRAP
"""



import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

os.environ["PYTHONHASHSEED"] = "39"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)


# =========================
# Konfiguracija
# =========================
CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/KvantniRegresor/Regresor_SVR_v2_predikcija.txt")
N_MIN, N_MAX = 1, 39
K = 7
LAG = 5
WINDOWS = (20, 50, 100)
BACKTEST_N = 100

T0 = time.time()
print("START", datetime.today())


# ✅ Load data
df = pd.read_csv(CSV_PATH, header=None).iloc[:, :K].astype(int)
print()
print("✅ Data loaded successfully.")
print()
"""
✅ Data loaded successfully.
"""


print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4620, Broj pozicija: 7
"""

draws = np.sort(df.values, axis=1)
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")


####################################


def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def build_features(draws_arr: np.ndarray, y_multi: np.ndarray) -> np.ndarray:
    n, _ = draws_arr.shape
    lag_blocks = []
    for lag in range(1, LAG + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[lag:] = draws_arr[:-lag]
        lag_blocks.append(shifted)
    lag_block = np.concatenate(lag_blocks, axis=1).astype(float)

    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for w in WINDOWS:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        rolling_blocks.append(rolled / float(w))
    rolling_block = np.concatenate(rolling_blocks, axis=1)

    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stats = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    return np.concatenate([lag_block, rolling_block, gap, stats], axis=1)


def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    scores = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -scores))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d: np.ndarray, y_true: np.ndarray) -> float:
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick: np.ndarray) -> str:
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


########################################


Y_full = draws_to_multihot(draws)
X_full = build_features(draws, Y_full)
START = max(LAG, max(WINDOWS))

X_all = X_full[START:].astype(float)
Y_all = Y_full[START:].astype(float)

n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > 200, "Premalo podataka za back-test."

X_train, Y_train = X_all[:n_train], Y_all[:n_train]
X_back, Y_back = X_all[n_train:], Y_all[n_train:]
X_next_raw = X_full[-1:].astype(float)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_back_scaled = scaler.transform(X_back)
X_next_scaled = scaler.transform(X_next_raw)


# Create the SVR regressor
svr = SVR(
    kernel="rbf",
    C=2.0,
    epsilon=0.05,
    gamma="scale",
    cache_size=1000,
    tol=1e-3,
)

# Create the Multioutput Regressor
mor = MultiOutputRegressor(svr, n_jobs=1)

# Train the regressor
print()
print("Treniranje SVR multi-label modela (39 izlaza) ...")
mor.fit(X_train_scaled, Y_train)
print("✅ SVR treniran.")
print()


# Generate predictions for testing data
scores_back = mor.predict(X_back_scaled)
scores_next = mor.predict(X_next_scaled)[0]

predicted_numbers = topk_from_scores(scores_next)


# ---- Stari "pozicioni mapirani" SVR (vraćeno iz v1) -----------------
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]


def map_to_indexed_range(df_in: pd.DataFrame, min_v, max_v) -> pd.DataFrame:
    out = df_in.copy()
    for i in range(df_in.shape[1]):
        out[i] = df_in[i] - min_v[i]
        if not out[i].between(0, max_v[i] - min_v[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_v[i] - min_v[i]}")
    return out


df_indexed = map_to_indexed_range(df, min_val, max_val).iloc[:, :7]

X_x = df_indexed.shift(1).dropna().values
y_x = df_indexed.iloc[1:].values

n_pos_train = X_x.shape[0] - BACKTEST_N
X_train_x_raw = X_x[:n_pos_train]
y_train_x = y_x[:n_pos_train]
X_back_x_raw = X_x[n_pos_train:]

scaler_pos = StandardScaler()
X_train_x_scaled = scaler_pos.fit_transform(X_train_x_raw)
X_back_x_scaled = scaler_pos.transform(X_back_x_raw)
X_next_x_scaled = scaler_pos.transform(X_x[-1:].astype(float))

svr_pos = SVR(kernel="rbf", C=2.0, epsilon=0.05, gamma="scale", cache_size=1000, tol=1e-3)
mor_pos = MultiOutputRegressor(svr_pos, n_jobs=1)

print("Treniranje SVR pozicionog (skaliran + mapiran) ...")
mor_pos.fit(X_train_x_scaled, y_train_x)
print("✅ SVR pozicioni mapirani treniran.")
print()

raw_pos = mor_pos.predict(X_next_x_scaled)[0]
predicted_numbers2 = np.array(
    [int(round(raw_pos[i])) + min_val[i] for i in range(K)], dtype=int
)
predicted_numbers2 = np.clip(predicted_numbers2, N_MIN, N_MAX)


# Skorovi za 1..39 iz pozicione predikcije (inverzna distanca do najbliže predviđene pozicije)
def positional_scores(raw_2d: np.ndarray, min_v) -> np.ndarray:
    """raw_2d shape (n, 7) — sirov SVR izlaz pre +min_val. Vraća (n, 39) skor."""
    nums = raw_2d + np.array(min_v, dtype=float)  # (n, 7)
    k_axis = np.arange(1, N_MAX + 1).reshape(1, 1, N_MAX)  # (1, 1, 39)
    dist = np.abs(nums[:, :, None] - k_axis)  # (n, 7, 39)
    min_dist = dist.min(axis=1)  # (n, 39)
    return -min_dist  # viši = bliži = bolji


# Skorovi za back-test (mapirani)
raw_pos_back = mor_pos.predict(X_back_x_scaled)
scores_back_pos = positional_scores(raw_pos_back, min_val)
scores_next_pos = positional_scores(raw_pos.reshape(1, -1), min_val)[0]

# Top-7 iz pozicionih skorova (alternativna prezentacija)
predicted_numbers2_topk = topk_from_scores(scores_next_pos)


print()
print("🎯 Predicted Next Lottery Numbers SVR_v2:", predicted_numbers.tolist(), describe(predicted_numbers))
print("🎯 Predicted Next Lottery Numbers predicted_numbers2 skalirani mapirani:", predicted_numbers2.tolist())
print("    (top-7 iz pozicionih skorova):", predicted_numbers2_topk.tolist(), describe(predicted_numbers2_topk))
print()
"""

"""


#######################################


print("Back-test (poslednjih 100 izvlačenja):")
h = avg_hits(scores_back, Y_back)
a = safe_auc(Y_back, scores_back)
l = safe_lrap(Y_back, scores_back)
h_pos = avg_hits(scores_back_pos, Y_back)
a_pos = safe_auc(Y_back, scores_back_pos)
l_pos = safe_lrap(Y_back, scores_back_pos)

print(f"{'model':<14} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'SVR_v2':<14} {h:>8.3f} {100*h/K:>6.1f}% {a:>7.3f} {l:>7.3f}")
print(f"{'SVR_mapirani':<14} {h_pos:>8.3f} {100*h_pos/K:>6.1f}% {a_pos:>7.3f} {l_pos:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()

assert len(set(predicted_numbers.tolist())) == K
assert predicted_numbers.min() >= N_MIN and predicted_numbers.max() <= N_MAX
assert list(predicted_numbers) == sorted(predicted_numbers.tolist())

with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={df.shape[0]}) ---\n")
    f.write(f"SVR_v2                            -> {predicted_numbers.tolist()}  ({describe(predicted_numbers)})\n")
    f.write(f"predicted_numbers2 skal. mapirani -> {predicted_numbers2.tolist()}\n")
    f.write(f"  top-7 iz pozicionih skorova     -> {predicted_numbers2_topk.tolist()}  ({describe(predicted_numbers2_topk)})\n")
    f.write(f"back-test SVR_v2:       hits/7={h:.3f}, AUC={a:.3f}, LRAP={l:.3f}\n")
    f.write(f"back-test SVR_mapirani: hits/7={h_pos:.3f}, AUC={a_pos:.3f}, LRAP={l_pos:.3f}\n")
    f.write(f"  (slučajan baseline ≈ {7*7/39:.3f} hits/7)\n")
print(f"Snimljeno u: {OUT_TXT}")
print()


###################


print()
print("\n✅ Script finished successfully.\n")
print()
"""
✅ Script finished successfully.
"""

elapsed = time.time() - T0
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()


"""
START 2026-05-24 18:18:36.983364

✅ Data loaded successfully.


Učitano kombinacija: 4620, Broj pozicija: 7


Treniranje SVR multi-label modela (39 izlaza) ...
✅ SVR treniran.

Treniranje SVR pozicionog (skaliran + mapiran) ...
✅ SVR pozicioni mapirani treniran.


🎯 Predicted Next Lottery Numbers SVR_v2: [10, 18, 23, 25, 26, 32, 36] suma=170, neparnih=2/7, niskih(<=19)=2/7, raspon=26
🎯 Predicted Next Lottery Numbers predicted_numbers2 skalirani mapirani: [4, 10, 14, 20, 25, 31, 37]
    (top-7 iz pozicionih skorova): [4, 10, 14, 20, 25, 31, 37] suma=141, neparnih=3/7, niskih(<=19)=3/7, raspon=33

Back-test (poslednjih 100 izvlačenja):
model            hits/7    hit%     AUC    LRAP
SVR_v2            1.280   18.3%   0.548   0.247
SVR_mapirani      1.180   16.9%   0.499   0.245
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /Users/4c/Desktop/GHQ/KvantniRegresor/Regresor_SVR_v2_predikcija.txt



✅ Script finished successfully.


STOP 2026-05-24 18:19:23.163619
Ukupno vreme: 0:00:46  (46.2 s)
"""
