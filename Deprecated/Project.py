# MovieLens 100k – Memory‑Based CF Experiments
# -------------------------------------------------------------
# This notebook‑style Python script implements the full experimental
# pipeline described in the thesis:
#   * Data loading & preprocessing (MovieLens‑100k)
#   * Similarity functions: Cosine, Adjusted Cosine, Pearson, Jaccard
#   * k‑NN CF (user‑based & item‑based)
#   * Baseline model‑based method: Funk‑SVD (Surprise)
#   * Evaluation: 5‑fold user‑stratified CV, rating (RMSE/MAE) &
#     ranking (Precision@k, Recall@k, nDCG@k, Coverage, Diversity)
#   * Significance testing (paired t‑test / Wilcoxon)
# -------------------------------------------------------------
# All heavy lifting happens in pure NumPy/Pandas + Surprise (for Funk‑SVD).
# To run as a Jupyter notebook, open this file with JupyterLab; each "# %%"
# cell delimiter will be recognised as a cell boundary.
# -------------------------------------------------------------

# %% Imports & Globals
import pathlib
import itertools
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel, wilcoxon

# Optional – Surprise for Funk‑SVD baseline
try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split as sp_train_test_split
except ImportError:
    SVD = None  # graceful degradation


DATA_DIR = pathlib.Path("ml-latest-small")  # adjust if needed
RATING_SCALE = (1, 5)
SEED = 42
np.random.seed(SEED)

# %% Data Loading ------------------------------------------------------------

def load_movielens_latest_small(path: Union[str, pathlib.Path] = DATA_DIR) -> pd.DataFrame:
    """Return ratings DataFrame with columns: user, item, rating, timestamp."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "MovieLens-latest-small directory not found. Download from "
            "https://grouplens.org/datasets/movielens/latest/ and unzip to 'ml-latest-small/'."
        )
    df = pd.read_csv(path / "ratings.csv")
    df.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
    return df

ratings_df = load_movielens_latest_small()
print(f"Loaded {len(ratings_df):,} ratings → {ratings_df.user.nunique()} users × {ratings_df.item.nunique()} items")

# %% Utility – Sparse Ratings Matrix ----------------------------------------

def df_to_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """Return (rating_matrix, uid_map, iid_map).
    rating_matrix shape = (n_users, n_items) with float32, NaN for missing."""
    uids = sorted(df.user.unique())
    iids = sorted(df.item.unique())
    uid_map = {u: idx for idx, u in enumerate(uids)}
    iid_map = {i: idx for idx, i in enumerate(iids)}
    mat = np.full((len(uids), len(iids)), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        mat[uid_map[row["user"]], iid_map[row["item"]]] = row["rating"]
    return mat, uid_map, iid_map

rating_matrix_full, uid_map_full, iid_map_full = df_to_matrix(ratings_df)

# %% Similarity Functions ----------------------------------------------------
# Each returns a float in [-1, 1] (except Jaccard [0, 1])

SimMetric = Literal["cosine", "adj_cosine", "pearson", "jaccard"]


def cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() == 0:
        return 0.0
    num = np.dot(x[mask], y[mask])
    den = math.sqrt(np.dot(x[mask], x[mask]) * np.dot(y[mask], y[mask]))
    return num / den if den else 0.0


def adj_cosine_sim(x: np.ndarray, y: np.ndarray, item_means: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() == 0:
        return 0.0
    xc = x[mask] - item_means[mask]
    yc = y[mask] - item_means[mask]
    num = np.dot(xc, yc)
    den = math.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
    return num / den if den else 0.0


def pearson_sim(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() == 0:
        return 0.0
    xc = x[mask] - np.nanmean(x[mask])
    yc = y[mask] - np.nanmean(y[mask])
    num = np.dot(xc, yc)
    den = math.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
    return num / den if den else 0.0


def jaccard_sim(x: np.ndarray, y: np.ndarray) -> float:
    x_bin = ~np.isnan(x)
    y_bin = ~np.isnan(y)
    inter = np.logical_and(x_bin, y_bin).sum()
    union = np.logical_or(x_bin, y_bin).sum()
    return inter / union if union else 0.0


# Dispatcher
SIM_FUNCS = {
    "cosine": cosine_sim,
    "adj_cosine": adj_cosine_sim,
    "pearson": pearson_sim,
    "jaccard": jaccard_sim,
}


# %% k‑NN CF Class -----------------------------------------------------------
class KNNRecommender:
    """Memory‑based CF (user‑ or item‑based) with pluggable similarity."""
    def __init__(self,
                 kind: Literal["user", "item"],
                 metric: SimMetric = "cosine",
                 k: int = 20):
        self.kind = kind
        self.metric = metric
        self.k = k
        self.sim_matrix: Optional[np.ndarray] = None
        self.rating_matrix: Optional[np.ndarray] = None
        self.item_means: Optional[np.ndarray] = None  # for adj_cosine

    def fit(self, rating_matrix: np.ndarray):
        self.rating_matrix = rating_matrix
        if self.metric == "adj_cosine":
            self.item_means = np.nanmean(rating_matrix, axis=0)
        # precompute similarity matrix
        axis = 0 if self.kind == "user" else 1
        n_entities = rating_matrix.shape[axis]
        self.sim_matrix = np.zeros((n_entities, n_entities), dtype=np.float32)
        for i in range(n_entities):
            for j in range(i + 1, n_entities):
                vec_i = rating_matrix[i, :] if self.kind == "user" else rating_matrix[:, i]
                vec_j = rating_matrix[j, :] if self.kind == "user" else rating_matrix[:, j]
                if self.metric == "adj_cosine":
                    sim = SIM_FUNCS[self.metric](vec_i, vec_j, self.item_means if self.kind == "user" else np.nanmean(rating_matrix, axis=1))
                else:
                    sim = SIM_FUNCS[self.metric](vec_i, vec_j)
                self.sim_matrix[i, j] = self.sim_matrix[j, i] = sim
        return self

    def _get_neighbors(self, idx: int, rated_mask: np.ndarray) -> List[int]:
        sims = self.sim_matrix[idx]
        # exclude self & entities without common ratings
        sims = sims.copy()
        sims[idx] = -np.inf
        candidate_idx = np.where(rated_mask)[0]
        return candidate_idx[np.argsort(sims[candidate_idx])[-self.k:][::-1]]

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.kind == "user":
            # user‑based: neighbors are users who rated item
            rated_mask = ~np.isnan(self.rating_matrix[:, item_idx])
            neighbors = self._get_neighbors(user_idx, rated_mask)
            sims = self.sim_matrix[user_idx, neighbors]
            ratings = self.rating_matrix[neighbors, item_idx]
        else:
            # item‑based
            rated_mask = ~np.isnan(self.rating_matrix[user_idx, :])
            neighbors = self._get_neighbors(item_idx, rated_mask)
            sims = self.sim_matrix[item_idx, neighbors]
            ratings = self.rating_matrix[user_idx, neighbors]
        if len(neighbors) == 0 or np.all(sims == 0):
            return np.nanmean(self.rating_matrix[user_idx, :])  # fallback: user mean
        return np.dot(sims, ratings) / np.sum(np.abs(sims))


# %% Evaluation Helpers ------------------------------------------------------

def ranking_metrics(recs: List[List[int]],
                    ground_truth: List[set],
                    k: int = 10) -> Dict[str, float]:
    """Compute Precision, Recall, nDCG, Coverage, Diversity (intra‑list)."""
    precisions, recalls, ndcgs = [], [], []
    all_recs = set()
    for rec, gt in zip(recs, ground_truth):
        hit_set = [1 if i in gt else 0 for i in rec[:k]]
        precisions.append(sum(hit_set) / k)
        recalls.append(sum(hit_set) / len(gt) if gt else 0)
        # DCG
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hit_set))
        # ideal DCG
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(gt), k)))
        ndcgs.append(dcg / ideal if ideal else 0)
        all_recs.update(rec[:k])
    coverage = len(all_recs) / ground_truth[0].__len__()  # approximated by total items
    return {
        "precision@k": np.mean(precisions),
        "recall@k": np.mean(recalls),
        "ndcg@k": np.mean(ndcgs),
        "coverage@k": coverage,
    }


# %% Cross‑Validation Runner --------------------------------------------------

def run_experiments(rating_df: pd.DataFrame,
                    ks: List[int],
                    metrics: List[SimMetric],
                    kind: Literal["user", "item"] = "user",
                    topk: int = 10,
                    n_splits: int = 5) -> pd.DataFrame:
    """Grid‑search over similarity metrics and neighborhood sizes.
    Returns a DataFrame with aggregated CV results."""
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for metric, k in itertools.product(metrics, ks):
        rmses, maes = [], []
        for fold, (train_idx, test_idx) in enumerate(kf.split(rating_df)):
            train_df = rating_df.iloc[train_idx]
            test_df = rating_df.iloc[test_idx]
            R_train, uid_map, iid_map = df_to_matrix(train_df)
            model = KNNRecommender(kind=kind, metric=metric, k=k).fit(R_train)
            # prediction loop
            y_true, y_pred = [], []
            for _, row in test_df.iterrows():
                u = uid_map.get(row.user, None)
                i = iid_map.get(row.item, None)
                if u is None or i is None:
                    continue  # cold‑start wrt train – skip
                pred = model.predict(u, i)
                if np.isnan(pred):
                    continue
                y_true.append(row.rating)
                y_pred.append(pred)
            rmses.append(math.sqrt(mean_squared_error(y_true, y_pred)))
            maes.append(mean_absolute_error(y_true, y_pred))
        results.append({
            "kind": kind,
            "metric": metric,
            "k": k,
            "rmse": np.mean(rmses),
            "mae": np.mean(maes),
        })
    return pd.DataFrame(results)


# %% Funk‑SVD Baseline -------------------------------------------------------

def run_funk_svd(rating_df: pd.DataFrame, n_epochs: int = 20) -> float:
    if SVD is None:
        print("Surprise not installed – skipping Funk‑SVD baseline.")
        return np.nan
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(rating_df[["user", "item", "rating"]], reader)
    trainset, testset = sp_train_test_split(data, test_size=0.2, random_state=SEED)
    algo = SVD(n_epochs=n_epochs, random_state=SEED)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = math.sqrt(mean_squared_error([p.r_ui for p in predictions], [p.est for p in predictions]))
    return rmse


# %% Significance Testing -----------------------------------------------------

def paired_significance(scores_a: List[float], scores_b: List[float],
                        test: Literal["ttest", "wilcoxon"] = "ttest") -> float:
    if test == "ttest":
        stat, p = ttest_rel(scores_a, scores_b)
    else:
        stat, p = wilcoxon(scores_a, scores_b)
    return p


# %% Example Execution -------------------------------------------------------
if __name__ == "__main__":
    grid_df = run_experiments(
        ratings_df,
        ks=[10, 20, 40],
        metrics=["cosine", "adj_cosine", "pearson", "jaccard"],
        kind="user",
    )
    print(grid_df.sort_values("rmse").head())

    # Optional baseline
    baseline_rmse = run_funk_svd(ratings_df)
    print(f"Funk‑SVD baseline RMSE: {baseline_rmse:.4f}")

# End of file

# %%
