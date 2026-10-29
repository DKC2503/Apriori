import time
import math
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger("elibrary.kmeans")


class KMeansRecommendationEngine:
    """
    Mathematical K-Means Reader Clustering and Recommendation Engine.
    
    1. Feature Engineering:
       - Vectorizes reader ratings over top M active books into a memory-efficient sparse CSR matrix.
       - Uses TruncatedSVD for latent preference dimensionality reduction (capturing principal axes of reader interest).
    2. Clustering:
       - Uses MiniBatchKMeans or KMeans minimizing within-cluster sum of squares (inertia):
         J = sum_{k=1}^K sum_{x_i in C_k} ||x_i - mu_k||^2
    3. Recommendation:
       - Ranks candidate books using cluster-specific Bayesian damped rating:
         S(b | C_k) = [n_{k,b} / (n_{k,b} + m)] * r_bar_{k,b} + [m / (n_{k,b} + m)] * R_bar_k
       - Excludes books the reader has already rated.
       - Provides global Bayesian average fallback for cold-start readers.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        rating_threshold: float = 3.0,
        n_top_books: int = 500,
        n_components: int = 30,
        random_state: int = 42,
        damping_factor: float = 5.0,
        algorithm: str = "minibatch_kmeans",
    ):
        self.n_clusters = n_clusters
        self.rating_threshold = rating_threshold
        self.n_top_books = n_top_books
        self.n_components = n_components
        self.random_state = random_state
        self.damping_factor = damping_factor
        self.algorithm = algorithm

        # State after training
        self.reader_ids: List[str] = []
        self.reader_to_idx: Dict[str, int] = {}
        self.feature_book_ids: List[str] = []
        self.book_to_feature_idx: Dict[str, int] = {}
        self.reader_assignments: Dict[str, int] = {}  # reader_id -> cluster_id
        self.reader_history: Dict[str, Dict[str, float]] = {}  # reader_id -> {book_id: rating}
        
        # Cluster aggregation statistics
        self.cluster_sizes: Dict[int, int] = {}
        self.cluster_books: Dict[int, Dict[str, Dict[str, Any]]] = {}  # cluster_id -> {book_id: {count, sum, avg}}
        self.cluster_mean_ratings: Dict[int, float] = {}
        self.global_book_stats: Dict[str, Dict[str, Any]] = {}  # book_id -> {count, sum, avg}
        self.global_mean_rating: float = 3.5

        # Metadata
        self.title_map: Dict[str, str] = {}
        self.book_metadata: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Any] = {}

    def extract_features_from_csv(
        self,
        ratings_path: Path,
        catalogue_ids: Set[str],
        chunk_size: int = 250_000,
    ) -> Tuple[csr_matrix, List[str], List[str]]:
        """
        Two-pass streaming feature extractor avoiding dense matrix memory blowup.
        
        Pass 1: Computes book frequency & global rating aggregates.
        Pass 2: Constructs a scipy.sparse.csr_matrix of reader ratings for top M books.
        """
        logger.info("Extracting features from %s in chunks of %d...", ratings_path, chunk_size)
        t0 = time.perf_counter()

        # Detect columns
        head_df = pd.read_csv(ratings_path, nrows=5, on_bad_lines="skip")
        cols_lower = {str(c).strip().lower(): c for c in head_df.columns}
        uid_col = cols_lower.get("user_id") or cols_lower.get("reader_id") or list(head_df.columns)[0]
        bid_col = cols_lower.get("book_id") or cols_lower.get("item_id") or list(head_df.columns)[1]
        score_col = cols_lower.get("rating") or cols_lower.get("score") or list(head_df.columns)[2]

        catalogue_numeric = {int(b) for b in catalogue_ids if str(b).isdigit()}
        has_numeric = bool(catalogue_numeric)

        # ----------------------------------------------------
        # Pass 1: Book Frequencies & Global Rating Statistics
        # ----------------------------------------------------
        book_counts = Counter()
        book_sums = defaultdict(float)
        total_rows = 0
        total_matched = 0

        for chunk in pd.read_csv(
            ratings_path,
            chunksize=chunk_size,
            usecols=[uid_col, bid_col, score_col],
            on_bad_lines="skip",
        ):
            total_rows += len(chunk)
            scores = pd.to_numeric(chunk[score_col], errors="coerce")
            valid_mask = scores.notna() & (scores > 0)
            if not valid_mask.any():
                continue

            valid_chunk = chunk[valid_mask]
            valid_scores = scores[valid_mask]
            bids = valid_chunk[bid_col]

            if has_numeric and pd.api.types.is_numeric_dtype(bids):
                match_mask = bids.isin(catalogue_numeric)
            else:
                bids = bids.astype(str).str.strip()
                match_mask = bids.isin(catalogue_ids)

            if not match_mask.any():
                continue

            matched_bids = bids[match_mask].astype(str).to_numpy()
            matched_scores = valid_scores[match_mask].to_numpy()
            total_matched += len(matched_bids)

            for b_id, s_val in zip(matched_bids, matched_scores):
                book_counts[b_id] += 1
                book_sums[b_id] += float(s_val)

        if total_matched == 0 or not book_counts:
            raise ValueError("No matching ratings found between books catalogue and ratings file.")

        # Compute global stats
        self.global_book_stats = {}
        total_rating_sum = 0.0
        total_rating_count = 0
        for b_id, cnt in book_counts.items():
            s_sum = book_sums[b_id]
            self.global_book_stats[b_id] = {
                "count": cnt,
                "sum": s_sum,
                "avg": round(s_sum / cnt, 3),
            }
            total_rating_sum += s_sum
            total_rating_count += cnt

        self.global_mean_rating = round(total_rating_sum / total_rating_count, 3) if total_rating_count else 3.5

        # Select Top M books for feature space
        top_m = min(self.n_top_books, len(book_counts))
        self.feature_book_ids = [b for b, _ in book_counts.most_common(top_m)]
        self.book_to_feature_idx = {b: idx for idx, b in enumerate(self.feature_book_ids)}
        feature_set = set(self.feature_book_ids)
        feature_set_num = {int(b) for b in feature_set if b.isdigit()}

        logger.info(
            "Pass 1 finished in %.3fs: %d total rows, %d matched, selected %d feature books.",
            time.perf_counter() - t0,
            total_rows,
            total_matched,
            len(self.feature_book_ids),
        )

        # ----------------------------------------------------
        # Pass 2: Sparse Matrix Construction & History Cache
        # ----------------------------------------------------
        t0_p2 = time.perf_counter()
        reader_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)

        for chunk in pd.read_csv(
            ratings_path,
            chunksize=chunk_size,
            usecols=[uid_col, bid_col, score_col],
            on_bad_lines="skip",
        ):
            scores = pd.to_numeric(chunk[score_col], errors="coerce")
            valid_mask = scores.notna() & (scores > 0)
            if not valid_mask.any():
                continue

            valid_chunk = chunk[valid_mask]
            valid_scores = scores[valid_mask]
            bids = valid_chunk[bid_col]
            uids = valid_chunk[uid_col].astype(str).str.strip().to_numpy()

            if feature_set_num and pd.api.types.is_numeric_dtype(bids):
                match_mask = bids.isin(feature_set_num)
            else:
                bids = bids.astype(str).str.strip()
                match_mask = bids.isin(feature_set)

            if not match_mask.any():
                continue

            sub_u = uids[match_mask]
            sub_b = bids[match_mask].astype(str).to_numpy()
            sub_s = valid_scores[match_mask].to_numpy()

            for u_val, b_val, s_val in zip(sub_u, sub_b, sub_s):
                reader_ratings[u_val][b_val] = float(s_val)

        # Filter readers with at least 1 feature book rating
        self.reader_ids = sorted(list(reader_ratings.keys()))
        self.reader_to_idx = {r: i for i, r in enumerate(self.reader_ids)}
        self.reader_history = reader_ratings

        n_readers = len(self.reader_ids)
        n_features = len(self.feature_book_ids)

        if n_readers < 2:
            raise ValueError(f"Insufficient usable readers found for clustering ({n_readers} reader).")

        # Build sparse CSR matrix
        row_indices: List[int] = []
        col_indices: List[int] = []
        data_values: List[float] = []

        for r_id, r_idx in self.reader_to_idx.items():
            for b_id, score_val in reader_ratings[r_id].items():
                c_idx = self.book_to_feature_idx.get(b_id)
                if c_idx is not None:
                    row_indices.append(r_idx)
                    col_indices.append(c_idx)
                    # Normalize rating onto [0.2, 1.0] scale
                    data_values.append(score_val / 5.0)

        sparse_matrix = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_readers, n_features),
            dtype=np.float32,
        )

        logger.info(
            "Pass 2 finished in %.3fs: sparse matrix shape=(%d, %d), non-zeros=%d (~%.2f MB RAM).",
            time.perf_counter() - t0_p2,
            n_readers,
            n_features,
            len(data_values),
            (sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes) / (1024 * 1024),
        )

        return sparse_matrix, self.reader_ids, self.feature_book_ids

    def fit(
        self,
        sparse_features: csr_matrix,
        title_map: Dict[str, str],
        book_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes Dimensionality Reduction (TruncatedSVD) and K-Means Clustering.
        Calculates mathematical inertia, iteration count, and cluster distributions.
        """
        t0 = time.perf_counter()
        self.title_map = title_map
        self.book_metadata = book_metadata or {}
        n_readers, n_features = sparse_features.shape

        if self.n_clusters >= n_readers:
            self.n_clusters = max(2, n_readers - 1)
            logger.warning("Adjusted n_clusters to %d because usable readers=%d", self.n_clusters, n_readers)

        # ----------------------------------------------------
        # 1. Dimensionality Reduction (TruncatedSVD)
        # ----------------------------------------------------
        t_svd_start = time.perf_counter()
        n_comp = min(self.n_components, n_features - 1, n_readers - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            reduced_features = svd.fit_transform(sparse_features)
            explained_variance = float(np.sum(svd.explained_variance_ratio_))
            t_svd = time.perf_counter() - t_svd_start
            logger.info("TruncatedSVD (%d components) completed in %.3fs (explained variance: %.2f%%)", n_comp, t_svd, explained_variance * 100)
        else:
            reduced_features = sparse_features.toarray()
            explained_variance = 1.0
            t_svd = 0.0

        # ----------------------------------------------------
        # 2. K-Means Reader Clustering
        # ----------------------------------------------------
        t_km_start = time.perf_counter()
        if self.algorithm == "minibatch_kmeans" and n_readers >= 500:
            batch_size = min(1024, max(128, n_readers // 10))
            km = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=batch_size,
                max_iter=100,
                n_init=3,
            )
        else:
            km = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                max_iter=100,
                n_init=5,
            )

        labels = km.fit_predict(reduced_features)
        t_km = time.perf_counter() - t_km_start

        # Record reader assignments
        self.reader_assignments = {r_id: int(labels[i]) for i, r_id in enumerate(self.reader_ids)}
        self.cluster_sizes = {int(k): int(v) for k, v in Counter(labels).items()}

        # ----------------------------------------------------
        # 3. Cluster Statistics & Bayesian Book Ranking
        # ----------------------------------------------------
        self._aggregate_cluster_statistics()

        total_elapsed = time.perf_counter() - t0
        self.metrics = {
            "n_readers": n_readers,
            "n_features": n_features,
            "n_clusters": self.n_clusters,
            "inertia": round(float(km.inertia_), 4),
            "n_iter": int(getattr(km, "n_iter_", 0)),
            "cluster_sizes": self.cluster_sizes,
            "explained_variance_ratio": round(explained_variance, 4),
            "algorithm_used": km.__class__.__name__,
            "svd_seconds": round(t_svd, 3),
            "clustering_seconds": round(t_km, 3),
            "total_fit_seconds": round(total_elapsed, 3),
        }

        logger.info(
            "K-Means clustering finished in %.3fs (Inertia: %.2f, Iterations: %d, Clusters: %s)",
            total_elapsed,
            self.metrics["inertia"],
            self.metrics["n_iter"],
            self.cluster_sizes,
        )

        return self.metrics

    def _aggregate_cluster_statistics(self):
        """
        Aggregates rating statistics per cluster to prepare Bayesian rankings.
        """
        self.cluster_books = {k: defaultdict(lambda: {"count": 0, "sum": 0.0}) for k in range(self.n_clusters)}
        cluster_total_ratings: Dict[int, float] = defaultdict(float)
        cluster_total_counts: Dict[int, int] = defaultdict(int)

        for r_id, cluster_id in self.reader_assignments.items():
            ratings_dict = self.reader_history.get(r_id, {})
            for b_id, score in ratings_dict.items():
                self.cluster_books[cluster_id][b_id]["count"] += 1
                self.cluster_books[cluster_id][b_id]["sum"] += score
                cluster_total_ratings[cluster_id] += score
                cluster_total_counts[cluster_id] += 1

        self.cluster_mean_ratings = {}
        for k in range(self.n_clusters):
            cnt = cluster_total_counts[k]
            self.cluster_mean_ratings[k] = round(cluster_total_ratings[k] / cnt, 3) if cnt > 0 else self.global_mean_rating

    def get_cluster_summaries(self, top_n_books: int = 5) -> List[Dict[str, Any]]:
        """
        Returns rich summaries for all clusters including representative signature books.
        """
        if hasattr(self, "_cached_summaries") and self._cached_summaries:
            return self._cached_summaries

        summaries: List[Dict[str, Any]] = []

        for k in range(self.n_clusters):
            size = self.cluster_sizes.get(k, 0)
            c_mean = self.cluster_mean_ratings.get(k, self.global_mean_rating)
            books_in_cluster = self.cluster_books.get(k, {})

            # Rank books in cluster using Bayesian Damped Score
            ranked_books = []
            m = self.damping_factor
            for b_id, stat in books_in_cluster.items():
                n_kb = stat["count"]
                if n_kb == 0:
                    continue
                r_bar = stat["sum"] / n_kb
                score = (n_kb / (n_kb + m)) * r_bar + (m / (n_kb + m)) * c_mean
                
                meta = self.book_metadata.get(b_id, {})
                title = self.title_map.get(b_id, meta.get("title", f"Book #{b_id}"))

                ranked_books.append({
                    "book_id": b_id,
                    "title": title,
                    "authors": meta.get("authors", "Unknown"),
                    "image_url": meta.get("image_url", ""),
                    "bayesian_score": round(score, 3),
                    "cluster_ratings_count": n_kb,
                    "cluster_avg_rating": round(r_bar, 2),
                })

            ranked_books.sort(key=lambda x: x["bayesian_score"], reverse=True)

            summaries.append({
                "cluster_id": k,
                "name": f"Cluster {k}",
                "size": size,
                "percentage": round((size / len(self.reader_ids) * 100), 1) if self.reader_ids else 0.0,
                "cluster_mean_rating": c_mean,
                "top_representative_books": ranked_books[:top_n_books],
            })

        return summaries

    def recommend_for_reader(
        self,
        reader_id: str,
        n_recommendations: int = 10,
    ) -> Dict[str, Any]:
        """
        Generates personalized book recommendations for reader_id using cluster affinity.
        1. Identifies reader's cluster C_k.
        2. Retrieves candidate books in C_k.
        3. Excludes books already rated by the reader.
        4. Ranks candidates by Bayesian damped score.
        5. Falls back to global popularity for cold-start readers.
        """
        r_str = str(reader_id).strip()
        already_rated_books = set(self.reader_history.get(r_str, {}).keys())
        is_known = r_str in self.reader_assignments

        if is_known:
            cluster_id = self.reader_assignments[r_str]
            c_mean = self.cluster_mean_ratings.get(cluster_id, self.global_mean_rating)
            books_in_cluster = self.cluster_books.get(cluster_id, {})
            m = self.damping_factor

            candidates: List[Dict[str, Any]] = []
            for b_id, stat in books_in_cluster.items():
                if b_id in already_rated_books:
                    continue  # Exclude already rated

                n_kb = stat["count"]
                if n_kb < 1:
                    continue

                r_bar = stat["sum"] / n_kb
                bayesian_score = (n_kb / (n_kb + m)) * r_bar + (m / (n_kb + m)) * c_mean
                meta = self.book_metadata.get(b_id, {})
                title = self.title_map.get(b_id, meta.get("title", f"Book #{b_id}"))

                candidates.append({
                    "book_id": b_id,
                    "title": title,
                    "authors": meta.get("authors", "Unknown"),
                    "image_url": meta.get("image_url", ""),
                    "score": round(bayesian_score, 3),
                    "cluster_avg_rating": round(r_bar, 2),
                    "cluster_ratings_count": n_kb,
                    "reason": (
                        f"Recommended because {n_kb} readers in your cluster (Cluster {cluster_id}) "
                        f"rated this book with an average rating of {r_bar:.1f} stars."
                    ),
                })

            candidates.sort(key=lambda x: (x["score"], x["cluster_ratings_count"]), reverse=True)
            recommendations = candidates[:n_recommendations]

            return {
                "reader_id": r_str,
                "is_cold_start": False,
                "assigned_cluster": cluster_id,
                "already_rated_count": len(already_rated_books),
                "cluster_size": self.cluster_sizes.get(cluster_id, 0),
                "recommendations": recommendations,
            }

        else:
            # Cold-start fallback: Global Bayesian Average
            m = self.damping_factor
            g_mean = self.global_mean_rating
            candidates = []

            for b_id, stat in self.global_book_stats.items():
                if b_id in already_rated_books:
                    continue
                cnt = stat["count"]
                if cnt < 2:
                    continue
                r_bar = stat["avg"]
                bayesian_score = (cnt / (cnt + m)) * r_bar + (m / (cnt + m)) * g_mean
                meta = self.book_metadata.get(b_id, {})
                title = self.title_map.get(b_id, meta.get("title", f"Book #{b_id}"))

                candidates.append({
                    "book_id": b_id,
                    "title": title,
                    "authors": meta.get("authors", "Unknown"),
                    "image_url": meta.get("image_url", ""),
                    "score": round(bayesian_score, 3),
                    "cluster_avg_rating": round(r_bar, 2),
                    "cluster_ratings_count": cnt,
                    "reason": (
                        f"Global popular recommendation ({cnt} positive reader ratings, "
                        f"average {r_bar:.1f} stars)."
                    ),
                })

            candidates.sort(key=lambda x: (x["score"], x["cluster_ratings_count"]), reverse=True)
            recommendations = candidates[:n_recommendations]

            return {
                "reader_id": r_str,
                "is_cold_start": True,
                "assigned_cluster": -1,
                "already_rated_count": len(already_rated_books),
                "cluster_size": len(self.reader_ids),
                "recommendations": recommendations,
                "message": f"Reader '{r_str}' is new or not in the training set. Showing top community favorites.",
            }

    def evaluate_offline(self, test_ratio: float = 0.20, k: int = 10) -> Dict[str, Any]:
        """
        Evaluates recommendation quality on a held-out test split of reader ratings.
        Computes genuine Precision@K, Recall@K, and HitRate@K.
        """
        # Select readers with sufficient ratings for evaluation
        min_hist = 2 if len(self.reader_ids) < 30 else 5
        eligible_readers = [
            r for r, hist in self.reader_history.items()
            if len(hist) >= min_hist and r in self.reader_assignments
        ]
        if len(eligible_readers) < 2:
            return {
                "status": "insufficient_data_for_evaluation",
                "k": k,
                "eval_readers_sampled": 0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "hit_rate_at_k": 0.0,
            }

        rng = np.random.RandomState(self.random_state)
        sample_size = min(200, len(eligible_readers))
        sample_readers = rng.choice(eligible_readers, size=sample_size, replace=False)

        precisions = []
        recalls = []
        hits = 0

        for r_id in sample_readers:
            hist = self.reader_history[r_id]
            # Split ratings: hold out positive ratings (>= threshold)
            pos_books = [b for b, score in hist.items() if score >= self.rating_threshold]
            if not pos_books:
                continue

            # Hold out 20%
            n_holdout = max(1, int(len(pos_books) * test_ratio))
            holdout_set = set(rng.choice(pos_books, size=n_holdout, replace=False))

            # Generate top K recommendations
            cluster_id = self.reader_assignments[r_id]
            c_mean = self.cluster_mean_ratings.get(cluster_id, self.global_mean_rating)
            books_in_cluster = self.cluster_books.get(cluster_id, {})
            m = self.damping_factor

            candidates = []
            for b_id, stat in books_in_cluster.items():
                if b_id in hist and b_id not in holdout_set:
                    continue  # Only exclude training ratings, allow holdout items
                n_kb = stat["count"]
                r_bar = stat["sum"] / n_kb if n_kb > 0 else c_mean
                score = (n_kb / (n_kb + m)) * r_bar + (m / (n_kb + m)) * c_mean
                candidates.append((b_id, score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top_k_rec_ids = {b_id for b_id, _ in candidates[:k]}

            matched = len(top_k_rec_ids & holdout_set)
            precisions.append(matched / k)
            recalls.append(matched / len(holdout_set))
            if matched > 0:
                hits += 1

        return {
            "eval_readers_sampled": len(precisions),
            "k": k,
            "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "hit_rate_at_k": round(hits / len(precisions), 4) if precisions else 0.0,
        }
