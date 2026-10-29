import os
import re
import gc
import time
import uuid
import hashlib
import logging
import tempfile
import asyncio
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any, Tuple, Set
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.database import (
    init_db,
    close_db,
    check_db_health,
    is_db_connected,
    get_masked_host,
)
from app.db_service import (
    compute_dataset_version,
    import_books_from_csv,
    import_ratings_from_csv,
    lookup_cached_cluster_model,
    save_cached_cluster_model,
    save_cluster_assignments,
    get_reader_cluster_assignment,
    get_cluster_details,
    record_clustering_run,
    get_analysis_history,
    clear_cluster_cache,
    is_dataset_imported,
    get_dataset_metadata,
    list_dataset_versions,
    estimate_storage_requirements,
)
from app.kmeans_service import KMeansRecommendationEngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [elibrary] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("elibrary")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes MongoDB Atlas connection and indexes on startup;
    closes database connection pool cleanly on shutdown.
    """
    logger.info("Starting up: initializing MongoDB Atlas connection...")
    await asyncio.to_thread(init_db)
    yield
    logger.info("Shutting down: closing MongoDB Atlas connection...")
    await asyncio.to_thread(close_db)


app = FastAPI(
    title="E-Library K-Means Recommendation System API",
    description="Reader Clustering and Personalized Book Recommendations using Scikit-Learn K-Means and MongoDB Atlas",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency safeguard: prevent multiple concurrent heavy clustering jobs from exhausting RAM/CPU
analysis_lock = asyncio.Lock()

# In-flight task coordination: deduplicates simultaneous identical requests while preserving 429 for incompatible analyses
_in_flight_tasks: Dict[str, asyncio.Future] = {}
_in_flight_lock = asyncio.Lock()

# 100 MiB maximum upload limit per file
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MiB in bytes

# Number of rows per chunk when reading large CSV files to maintain low memory profile (<100 MB RAM)
CHUNK_SIZE = 250_000

# Active in-memory model state (allows sub-millisecond personalized reader recommendations)
_active_engine: Optional[KMeansRecommendationEngine] = None
_active_model_id: Optional[str] = None
_active_dataset_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Reusable In-Memory Caches
# ---------------------------------------------------------------------------

class CatalogueCache:
    """In-memory LRU cache for parsed book catalogues to avoid re-parsing identical books.csv files."""
    def __init__(self, max_items: int = 5):
        self.max_items = max_items
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]):
        if len(self._cache) >= self.max_items:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value


catalogue_cache = CatalogueCache(max_items=5)


# ---------------------------------------------------------------------------
# API Health and Metadata Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "E-Library K-Means Recommendation System",
        "algorithm": "Scikit-Learn K-Means Reader Clustering",
        "docs": "/docs",
        "database": get_masked_host(),
    }


@app.get("/health")
def health():
    db_status = check_db_health()
    return {
        "status": "ok",
        "algorithm": "K-Means Reader Clustering (scikit-learn)",
        "active_model_loaded": _active_engine is not None,
        "active_dataset_version": _active_dataset_version,
        "max_upload_size_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
        "database": db_status,
    }


@app.get("/db-health")
@app.get("/db/status")
def db_status():
    """Returns detailed diagnostics of the MongoDB Atlas connection and storage."""
    return check_db_health()


@app.get("/dataset/status")
def dataset_status():
    """Returns status of stored dataset metadata, collections, and storage estimates."""
    db_stat = check_db_health()
    meta = get_dataset_metadata()
    versions = list_dataset_versions()
    storage = estimate_storage_requirements()
    return {
        "database_connected": db_stat.get("connected", False),
        "cluster_host": db_stat.get("cluster_host", "None"),
        "collections": db_stat.get("collections", {}),
        "latest_dataset": meta,
        "dataset_versions": versions,
        "storage_estimate": storage,
    }


@app.get("/history")
def history(limit: int = Query(15, ge=1, le=50)):
    """Returns recent analysis runs audit history from MongoDB."""
    runs = get_analysis_history(limit=limit)
    return {"runs": runs, "count": len(runs)}


@app.delete("/cache")
def clear_cache(dataset_version: Optional[str] = None):
    """Invalidates cached K-Means clustering models and summaries from MongoDB."""
    global _active_engine, _active_model_id, _active_dataset_version
    _active_engine = None
    _active_model_id = None
    _active_dataset_version = None

    deleted = clear_cluster_cache(dataset_version)
    return {
        "status": "ok",
        "deleted_count": deleted,
        "message": f"Cleared {deleted} cached K-Means models and in-memory active models.",
    }


# ---------------------------------------------------------------------------
# CSV Streaming and Upload Helpers
# ---------------------------------------------------------------------------

def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds a column matching any of the candidate names case-insensitively."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    return next((cols[x] for x in candidates if x in cols), None)


async def save_upload_to_temp(upload: UploadFile, max_size: int = MAX_UPLOAD_SIZE) -> Tuple[Path, int, str]:
    """
    Streams an uploaded file directly to a temporary file on disk with chunked reading.
    Validates file extension and ensures total size does not exceed max_size.
    Computes a fast SHA-256 digest of the first 64 KB for catalogue caching.
    """
    filename = upload.filename or "uploaded_file"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files (.csv) are accepted. Received: '{filename}'.",
        )

    fd, raw_path = tempfile.mkstemp(prefix="elibrary_upload_", suffix=".csv")
    temp_path = Path(raw_path)
    total_bytes = 0
    head_hasher = hashlib.sha256()

    try:
        with os.fdopen(fd, "wb") as temp_file:
            while True:
                chunk = await upload.read(64 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes <= 65536:
                    head_hasher.update(chunk)
                if total_bytes > max_size:
                    size_mb = total_bytes // (1024 * 1024)
                    limit_mb = max_size // (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Uploaded file '{filename}' exceeds the {limit_mb} MiB upload limit. "
                            f"Current upload reached {size_mb} MiB."
                        ),
                    )
                temp_file.write(chunk)

        if total_bytes == 0:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"The uploaded file '{filename}' is empty.")

        head_hash = head_hasher.hexdigest()
        return temp_path, total_bytes, head_hash

    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Could not read upload '{filename}': {str(e)}",
        )


def load_csv(
    file_path: Path,
    required_id_candidates: List[str],
    required_second_candidates: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str]]:
    """Loads a small-to-medium CSV and detects required column names with helpful error messages."""
    try:
        header_df = pd.read_csv(file_path, nrows=0, on_bad_lines="skip")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV header from '{file_path.name}': {str(e)}")

    id_col = get_col(header_df, required_id_candidates)
    if not id_col:
        found = list(header_df.columns)
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{file_path.name}' is missing required ID column. "
                f"Expected one of {required_id_candidates}, found: {found}."
            ),
        )

    second_col = None
    if required_second_candidates:
        second_col = get_col(header_df, required_second_candidates)
        if not second_col:
            found = list(header_df.columns)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{file_path.name}' is missing required secondary column. "
                    f"Expected one of {required_second_candidates}, found: {found}."
                ),
            )

    usecols = [id_col]
    if second_col:
        usecols.append(second_col)

    try:
        df = pd.read_csv(file_path, usecols=usecols, on_bad_lines="skip")
        return df, id_col, second_col, None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV content: {str(e)}")


# ---------------------------------------------------------------------------
# Core Analysis Endpoint: K-Means Reader Clustering
# ---------------------------------------------------------------------------

@app.post("/analyze")
@app.post("/cluster")
async def analyze(
    books_file: UploadFile = File(...),
    ratings_file: UploadFile = File(...),
    n_clusters: int = Form(5),
    rating_threshold: float = Form(3.0),
    random_state: int = Form(42),
    algorithm: str = Form("minibatch_kmeans"),
    # Compatibility arguments for older clients
    min_support: Optional[float] = Form(None),
    min_confidence: Optional[float] = Form(None),
    num_rules: Optional[int] = Form(None),
):
    """
    Primary K-Means Reader Clustering Endpoint.
    1. Validates inputs & checks MongoDB cluster cache (returning in ~0.02s on HIT).
    2. Streams ratings.csv in low-memory chunks, vectorizing reader profiles into sparse matrix.
    3. Performs TruncatedSVD dimensionality reduction & Scikit-Learn K-Means reader clustering.
    4. Computes genuine mathematical inertia, iterations, and Bayesian cluster rankings.
    5. Persists model and bucketed reader assignments to MongoDB Atlas.
    """
    global _active_engine, _active_model_id, _active_dataset_version
    request_start_time = time.perf_counter()
    logger.info(
        "--> [START] K-Means Clustering request: k=%d, threshold=%.1f, seed=%d, algo=%s",
        n_clusters,
        rating_threshold,
        random_state,
        algorithm,
    )

    # Validation
    if not (2 <= n_clusters <= 50):
        raise HTTPException(status_code=400, detail="Number of clusters (k) must be between 2 and 50.")
    if not (0.5 <= rating_threshold <= 10.0):
        raise HTTPException(status_code=400, detail="Rating threshold must be between 0.5 and 10.0.")

    books_temp_path: Optional[Path] = None
    ratings_temp_path: Optional[Path] = None
    task_key: Optional[str] = None
    is_leader = False

    try:
        # 1. Stream Uploads Directly to Disk
        t0 = time.perf_counter()
        books_temp_path, books_bytes, books_hash = await save_upload_to_temp(books_file)
        ratings_temp_path, ratings_bytes, _ = await save_upload_to_temp(ratings_file)
        t_ingest = time.perf_counter() - t0

        orig_b = getattr(books_file, "filename", None) or "books.csv"
        orig_r = getattr(ratings_file, "filename", None) or "ratings.csv"
        dataset_version = compute_dataset_version(books_temp_path, ratings_temp_path, original_names=[orig_b, orig_r])
        logger.info("Dataset version fingerprint: [%s]", dataset_version)

        algo_norm = algorithm.strip().lower()
        task_key = f"{dataset_version}:{n_clusters}:{rating_threshold:.2f}:{random_state}:{algo_norm}"

        # 2. Check MongoDB Persistent Cluster Cache (Tier 1 Instant Hit)
        cached_model = await asyncio.to_thread(
            lookup_cached_cluster_model,
            dataset_version=dataset_version,
            n_clusters=n_clusters,
            rating_threshold=rating_threshold,
            random_state=random_state,
            algorithm=algo_norm,
        )
        if cached_model:
            total_elapsed = round(time.perf_counter() - request_start_time, 4)
            cached_model["timing"]["total_seconds"] = total_elapsed
            logger.info("<-- [CACHE HIT] Returning K-Means clustering in %.4fs from MongoDB cache", total_elapsed)

            # Restore active model state for instant subsequent /recommend and /clusters queries
            _active_model_id = cached_model.get("model_id")
            _active_dataset_version = dataset_version
            reconstructed_engine = KMeansRecommendationEngine(
                n_clusters=n_clusters,
                rating_threshold=rating_threshold,
                random_state=random_state,
                algorithm=algo_norm,
            )
            reconstructed_engine.cluster_sizes = cached_model.get("metrics", {}).get("cluster_sizes", {})
            reconstructed_engine.metrics = cached_model.get("metrics", {})
            reconstructed_engine._cached_summaries = cached_model.get("clusters", [])
            reconstructed_engine.n_clusters = len(cached_model.get("clusters", [])) or n_clusters
            _active_engine = reconstructed_engine

            await asyncio.to_thread(
                record_clustering_run,
                dataset_version=dataset_version,
                n_clusters=n_clusters,
                rating_threshold=rating_threshold,
                algorithm=algo_norm,
                inertia=cached_model.get("metrics", {}).get("inertia", 0.0),
                usable_readers=cached_model.get("summary", {}).get("usable_readers", 0),
                usable_books=cached_model.get("summary", {}).get("usable_books", 0),
                total_seconds=total_elapsed,
                source="mongodb_cluster_cache",
                cache_hit=True,
            )
            return cached_model

        # 3. Concurrency Safeguard & In-Flight Duplicate Reuse
        async with _in_flight_lock:
            if task_key in _in_flight_tasks:
                logger.info("[CONCURRENCY] Identical analysis in progress for key [%s]. Awaiting existing task...", task_key)
                existing_fut = _in_flight_tasks[task_key]
            elif len(_in_flight_tasks) > 0 or analysis_lock.locked():
                logger.warning("Rejecting concurrent incompatible analysis request (HTTP 429). Instance busy.")
                raise HTTPException(
                    status_code=429,
                    detail="Another clustering analysis is currently being processed on this instance. Please wait a moment and try again.",
                )
            else:
                leader_fut = asyncio.get_running_loop().create_future()
                _in_flight_tasks[task_key] = leader_fut
                is_leader = True

        if not is_leader:
            try:
                leader_result = await asyncio.wait_for(asyncio.shield(existing_fut), timeout=120.0)
                total_elapsed = round(time.perf_counter() - request_start_time, 4)
                follower_result = dict(leader_result)
                follower_result["timing"] = dict(leader_result.get("timing", {}))
                follower_result["timing"]["total_seconds"] = total_elapsed
                follower_result["timing"]["source"] = "concurrent_in_flight_reuse"
                logger.info("<-- [CONCURRENT REUSE] Safely reused in-flight clustering result for [%s] in %.4fs", task_key, total_elapsed)
                return follower_result
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for concurrent analysis [%s] after 120s", task_key)
                raise HTTPException(status_code=504, detail="Clustering analysis timed out while waiting for existing in-flight task.")

        # 4. Parse Books Catalogue & Metadata (Leader Execution)
        async with analysis_lock:
            t0 = time.perf_counter()
            cached_cat = catalogue_cache.get(books_hash)
            if cached_cat:
                catalogue_ids = cached_cat["catalogue_ids"]
                title_map = cached_cat["title_map"]
                book_metadata = cached_cat["book_metadata"]
                total_catalogue_books = cached_cat["total_books"]
            else:
                try:
                    books_df = pd.read_csv(books_temp_path, on_bad_lines="skip")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Could not parse books.csv: {str(e)}")

                cols_lower = {str(c).strip().lower(): c for c in books_df.columns}
                bid_col = cols_lower.get("book_id") or cols_lower.get("id") or cols_lower.get("isbn")
                title_col = cols_lower.get("title") or cols_lower.get("book_title") or cols_lower.get("name")

                if not bid_col or not title_col:
                    found_cols = list(books_df.columns)
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"books.csv is missing required columns. Found: {found_cols}. "
                            "Required: a book ID column ('book_id') and a title column ('title')."
                        ),
                    )

                books_df["book_id"] = books_df[bid_col].astype(str).str.strip()
                books_df["title"] = books_df[title_col].astype(str).str.strip()
                books_df = books_df.dropna(subset=["book_id", "title"])
                books_df = books_df[(books_df["book_id"] != "") & (books_df["title"] != "")]
                books_df = books_df.drop_duplicates(subset=["book_id"], keep="first")

                if len(books_df) < 2:
                    raise HTTPException(
                        status_code=400,
                        detail=f"The book catalogue must contain at least 2 distinct books. Found {len(books_df)}.",
                    )

                author_col = cols_lower.get("authors") or cols_lower.get("author")
                img_col = cols_lower.get("image_url") or cols_lower.get("small_image_url")

                title_map = dict(zip(books_df["book_id"], books_df["title"]))
                catalogue_ids = set(title_map.keys())
                total_catalogue_books = len(title_map)

                book_metadata = {}
                for row in books_df.itertuples(index=False):
                    r_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(books_df.columns, row))
                    b_id = str(r_dict["book_id"]).strip()
                    book_metadata[b_id] = {
                        "book_id": b_id,
                        "title": str(r_dict["title"]).strip(),
                        "authors": str(r_dict.get(author_col, "Unknown")).strip() if author_col and pd.notna(r_dict.get(author_col)) else "Unknown",
                        "image_url": str(r_dict.get(img_col, "")).strip() if img_col and pd.notna(r_dict.get(img_col)) else "",
                    }

                catalogue_cache.set(
                    books_hash,
                    {
                        "catalogue_ids": catalogue_ids,
                        "title_map": title_map,
                        "book_metadata": book_metadata,
                        "total_books": total_catalogue_books,
                    },
                )

            t_catalogue = time.perf_counter() - t0

            # 5. Check ratings.csv Header & Columns
            try:
                ratings_header = pd.read_csv(ratings_temp_path, nrows=0, on_bad_lines="skip")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not parse ratings.csv: {str(e)}")

            uid = get_col(ratings_header, ["user_id", "userid", "reader_id", "user", "reader"])
            rbid = get_col(ratings_header, ["book_id", "bookid", "item_id", "book"])
            score = get_col(ratings_header, ["rating", "score", "user_rating", "stars"])

            if not all([uid, rbid, score]):
                found_cols = list(ratings_header.columns)
                missing = [c for c, present in [("user_id", uid), ("book_id", rbid), ("rating", score)] if not present]
                raise HTTPException(
                    status_code=400,
                    detail=f"ratings.csv is missing required columns ({', '.join(missing)}). Found columns: {found_cols}.",
                )

            # 6. Execute Mathematical Feature Extraction & K-Means Engine
            engine = KMeansRecommendationEngine(
                n_clusters=n_clusters,
                rating_threshold=rating_threshold,
                random_state=random_state,
                algorithm=algo_norm,
            )

            t0_features = time.perf_counter()
            try:
                sparse_features, reader_ids, feature_book_ids = await asyncio.to_thread(
                    engine.extract_features_from_csv,
                    ratings_path=ratings_temp_path,
                    catalogue_ids=catalogue_ids,
                    chunk_size=CHUNK_SIZE,
                )
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            t_features = time.perf_counter() - t0_features

            t0_cluster = time.perf_counter()
            metrics = await asyncio.to_thread(
                engine.fit,
                sparse_features=sparse_features,
                title_map=title_map,
                book_metadata=book_metadata,
            )
            t_clustering = time.perf_counter() - t0_cluster

            # 7. Generate Cluster Summaries & Sample Recommendations
            cluster_summaries = engine.get_cluster_summaries(top_n_books=6)
            sample_reader = reader_ids[0] if reader_ids else "1"
            sample_recs = engine.recommend_for_reader(sample_reader, n_recommendations=6)

            model_id = f"km_{uuid.uuid4().hex[:10]}"
            total_elapsed = round(time.perf_counter() - request_start_time, 3)

            summary_info = {
                "total_catalogue_books": total_catalogue_books,
                "usable_readers": len(reader_ids),
                "usable_books": len(feature_book_ids),
                "n_clusters": engine.n_clusters,
                "inertia": metrics["inertia"],
                "iterations": metrics["n_iter"],
                "explained_variance_ratio": metrics["explained_variance_ratio"],
                "algorithm": metrics["algorithm_used"],
            }

            timing_info = {
                "total_seconds": total_elapsed,
                "ingest_seconds": round(t_ingest, 3),
                "catalogue_seconds": round(t_catalogue, 3),
                "feature_extraction_seconds": round(t_features, 3),
                "clustering_seconds": round(t_clustering, 3),
                "cache_hit": False,
                "source": "fresh_kmeans_computation",
            }

            # 8. Persist to MongoDB Atlas
            if is_db_connected():
                await asyncio.to_thread(
                    import_books_from_csv,
                    books_path=books_temp_path,
                )
                save_ok = await asyncio.to_thread(
                    save_cached_cluster_model,
                    dataset_version=dataset_version,
                    n_clusters=engine.n_clusters,
                    rating_threshold=rating_threshold,
                    random_state=random_state,
                    model_id=model_id,
                    summary=summary_info,
                    clusters=cluster_summaries,
                    metrics=metrics,
                    timing=timing_info,
                    algorithm=algo_norm,
                    sample_recommendations=sample_recs,
                )
                if not save_ok:
                    logger.error("MongoDB cluster model cache write failed for model [%s].", model_id)
                    timing_info["cache_persisted"] = False
                    timing_info["cache_diagnostic"] = "Cache write failed. Model results computed but not cached."
                else:
                    timing_info["cache_persisted"] = True
                    timing_info["cache_diagnostic"] = None

                await asyncio.to_thread(
                    save_cluster_assignments,
                    dataset_version=dataset_version,
                    model_id=model_id,
                    reader_assignments=engine.reader_assignments,
                )
                await asyncio.to_thread(
                    record_clustering_run,
                    dataset_version=dataset_version,
                    n_clusters=engine.n_clusters,
                    rating_threshold=rating_threshold,
                    algorithm=algo_norm,
                    inertia=metrics["inertia"],
                    usable_readers=len(reader_ids),
                    usable_books=len(feature_book_ids),
                    total_seconds=total_elapsed,
                    source="fresh_kmeans_computation",
                    cache_hit=False,
                )

            # Update active in-memory engine for fast subsequent /recommend queries
            _active_engine = engine
            _active_model_id = model_id
            _active_dataset_version = dataset_version

            logger.info(
                "<-- [COMPLETED] K-Means Clustering finished in %.3fs (Inertia: %.2f, Usable Readers: %d)",
                total_elapsed,
                metrics["inertia"],
                len(reader_ids),
            )

            response_data = {
                "status": "success",
                "model_id": model_id,
                "dataset_version": dataset_version,
                "summary": summary_info,
                "metrics": metrics,
                "clusters": cluster_summaries,
                "sample_recommendations": sample_recs,
                "timing": timing_info,
            }

            if is_leader and task_key in _in_flight_tasks and not _in_flight_tasks[task_key].done():
                _in_flight_tasks[task_key].set_result(response_data)

            return response_data

    except HTTPException as he:
        if is_leader and task_key and task_key in _in_flight_tasks and not _in_flight_tasks[task_key].done():
            _in_flight_tasks[task_key].set_exception(he)
        raise
    except Exception as e:
        if is_leader and task_key and task_key in _in_flight_tasks and not _in_flight_tasks[task_key].done():
            _in_flight_tasks[task_key].set_exception(e)
        total_elapsed = time.perf_counter() - request_start_time
        logger.exception("Unexpected error in /analyze after %.3fs: %s", total_elapsed, e)
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")
    finally:
        if is_leader and task_key:
            async with _in_flight_lock:
                _in_flight_tasks.pop(task_key, None)
        if books_temp_path and books_temp_path.exists():
            try:
                books_temp_path.unlink()
            except Exception:
                pass
        if ratings_temp_path and ratings_temp_path.exists():
            try:
                ratings_temp_path.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Recommendation & Cluster Inspection Endpoints
# ---------------------------------------------------------------------------

@app.get("/recommend/{reader_id}")
def get_recommendations_for_reader(
    reader_id: str,
    n: int = Query(10, ge=1, le=50),
    dataset_version: Optional[str] = None,
    model_id: Optional[str] = None,
):
    """
    Generates personalized recommendations for a specific reader.
    Uses active in-memory trained K-Means engine if available, or looks up assignments from MongoDB.
    Excludes books already rated by the reader.
    Falls back gracefully to global Bayesian average for cold-start readers.
    """
    global _active_engine, _active_model_id, _active_dataset_version
    if _active_engine is not None and bool(getattr(_active_engine, "reader_history", None)):
        rec_data = _active_engine.recommend_for_reader(reader_id, n_recommendations=n)
        return rec_data

    # If no active model in memory with reader_history, check MongoDB cluster assignments
    d_ver = dataset_version or _active_dataset_version
    m_id = model_id or _active_model_id
    if d_ver and m_id:
        c_id = get_reader_cluster_assignment(d_ver, m_id, reader_id)
        if c_id is not None:
            c_details = get_cluster_details(d_ver, m_id, c_id)
            if c_details:
                top_books = c_details.get("top_representative_books", [])[:n]
                return {
                    "reader_id": reader_id,
                    "is_cold_start": False,
                    "assigned_cluster": c_id,
                    "cluster_size": c_details.get("size", 0),
                    "recommendations": top_books,
                    "source": "mongodb_cluster_assignment",
                }
        else:
            # Cold-start reader: fallback to representative books from cluster 0
            c_details = get_cluster_details(d_ver, m_id, 0)
            if c_details:
                top_books = c_details.get("top_representative_books", [])[:n]
                return {
                    "reader_id": reader_id,
                    "is_cold_start": True,
                    "assigned_cluster": 0,
                    "cluster_size": c_details.get("size", 0),
                    "recommendations": top_books,
                    "source": "mongodb_cold_start_fallback",
                }

    if _active_engine is not None:
        rec_data = _active_engine.recommend_for_reader(reader_id, n_recommendations=n)
        return rec_data

    raise HTTPException(
        status_code=400,
        detail="No active clustering model found. Please run K-Means clustering via /analyze first.",
    )


@app.get("/clusters")
def get_clusters():
    """Returns summaries of all reader clusters from the active model."""
    global _active_engine
    if _active_engine is None:
        raise HTTPException(status_code=400, detail="No active clustering model. Run /analyze first.")
    return {
        "model_id": _active_model_id,
        "dataset_version": _active_dataset_version,
        "clusters": _active_engine.get_cluster_summaries(top_n_books=8),
    }


@app.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: int):
    """Returns detailed statistics and top books for a specific cluster ID."""
    global _active_engine
    if _active_engine is None:
        raise HTTPException(status_code=400, detail="No active clustering model. Run /analyze first.")
    if cluster_id < 0 or cluster_id >= _active_engine.n_clusters:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} does not exist.")

    summaries = _active_engine.get_cluster_summaries(top_n_books=15)
    target = next((c for c in summaries if c["cluster_id"] == cluster_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")
    return target


@app.get("/evaluate")
def evaluate_model(test_ratio: float = Query(0.20, ge=0.05, le=0.50), k: int = Query(10, ge=1, le=20)):
    """
    Evaluates recommendation quality (Precision@K, Recall@K, HitRate@K)
    using a held-out test split of reader ratings.
    """
    global _active_engine
    if _active_engine is None:
        raise HTTPException(status_code=400, detail="No active clustering model. Run /analyze first.")

    res = _active_engine.evaluate_offline(test_ratio=test_ratio, k=k)
    return res


# ---------------------------------------------------------------------------
# One-Time Dataset Import Endpoint
# ---------------------------------------------------------------------------

@app.post("/import-dataset")
async def import_dataset_endpoint(
    books_file: UploadFile = File(...),
    ratings_file: Optional[UploadFile] = File(None),
    rating_threshold: float = Form(3.0),
    force: bool = Form(False),
    store_raw_ratings: bool = Form(False),
    max_ratings: int = Form(500_000),
):
    """
    Dedicated endpoint for one-time ingestion of books.csv and optional ratings.csv into MongoDB Atlas.
    """
    if not is_db_connected():
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not connected. Ingestion requires an active database connection.",
        )

    t0 = time.perf_counter()
    books_path, _, _ = await save_upload_to_temp(books_file)
    ratings_path: Optional[Path] = None

    try:
        books_res = await asyncio.to_thread(import_books_from_csv, books_path)
        raw_ratings_res: Optional[Dict[str, Any]] = None

        if ratings_file is not None and store_raw_ratings:
            ratings_path, _, _ = await save_upload_to_temp(ratings_file)
            raw_ratings_res = await asyncio.to_thread(import_ratings_from_csv, ratings_path, max_ratings)

        total_elapsed = round(time.perf_counter() - t0, 3)
        return {
            "status": "success",
            "elapsed_seconds": total_elapsed,
            "books_import": books_res,
            "raw_ratings_import": raw_ratings_res,
        }
    finally:
        if books_path and books_path.exists():
            books_path.unlink(missing_ok=True)
        if ratings_path and ratings_path.exists():
            ratings_path.unlink(missing_ok=True)
