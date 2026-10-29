import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Add backend directory to path if running directly
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database import init_db, close_db, is_db_connected, get_masked_host
from app.db_service import (
    import_books_from_csv,
    import_ratings_from_csv,
    save_cached_cluster_model,
    save_cluster_assignments,
    record_clustering_run,
    compute_dataset_version,
    estimate_storage_requirements,
)
from app.main import (
    load_csv,
    get_col,
    CHUNK_SIZE,
)
from app.kmeans_service import KMeansRecommendationEngine
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [import_cli] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("import_cli")


def main():
    parser = argparse.ArgumentParser(
        description="One-time dataset ingestion and K-Means pre-clustering tool for E-Library Recommendation System with MongoDB Atlas."
    )
    parser.add_argument("--books", required=True, help="Path to books.csv")
    parser.add_argument("--ratings", required=False, help="Optional path to ratings.csv for K-Means clustering")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Positive rating threshold (default: 3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed (default: 42)")
    parser.add_argument("--force", action="store_true", help="Force re-import even if already present")
    parser.add_argument("--store-raw-ratings", action="store_true", help="Also store raw rating records in ratings collection (bounded)")
    parser.add_argument("--max-ratings", type=int, default=500_000, help="Max raw ratings to store (default: 500,000 for Atlas M0 safety)")
    args = parser.parse_args()

    books_path = Path(args.books)
    if not books_path.exists():
        logger.error("books file not found at: %s", books_path)
        sys.exit(1)

    ratings_path = Path(args.ratings) if args.ratings else None
    if ratings_path and not ratings_path.exists():
        logger.error("ratings file not found at: %s", ratings_path)
        sys.exit(1)

    logger.info("Initializing MongoDB Atlas connection...")
    if not init_db():
        logger.error("Could not connect to MongoDB Atlas. Please ensure MONGODB_URI is set correctly in backend/.env")
        sys.exit(1)

    logger.info("Connected to cluster [%s]", get_masked_host())

    storage_est = estimate_storage_requirements()
    logger.info("Atlas Storage Estimation: Recommended architecture = %s", storage_est["recommended_architecture"])

    # Compute dataset version
    dataset_version = compute_dataset_version(books_path, ratings_path) if ratings_path else "books_only"
    logger.info("Dataset Version Hash: [%s]", dataset_version)

    # 1. Ingest books
    logger.info("Step 1: Ingesting books metadata...")
    books_res = import_books_from_csv(books_path)
    logger.info("Books Result: %s", books_res)

    # Optional raw ratings storage
    if ratings_path and args.store_raw_ratings:
        logger.info("Optional: Ingesting bounded raw ratings records (max %d rows)...", args.max_ratings)
        raw_res = import_ratings_from_csv(ratings_path, max_rows=args.max_ratings)
        logger.info("Raw Ratings Result: %s", raw_res)

    # 2. Ingest / cluster ratings if provided
    if ratings_path:
        logger.info("Step 2: Pre-clustering ratings using K-Means and persisting to MongoDB Atlas...")
        t0 = time.perf_counter()

        # Catalogue lookup
        books_df = pd.read_csv(books_path, on_bad_lines="skip")
        cols_lower = {str(c).strip().lower(): c for c in books_df.columns}
        b_col = cols_lower.get("book_id") or cols_lower.get("id") or list(books_df.columns)[0]
        t_col = cols_lower.get("title") or cols_lower.get("book_title") or list(books_df.columns)[1]
        author_col = cols_lower.get("authors") or cols_lower.get("author")
        img_col = cols_lower.get("image_url") or cols_lower.get("small_image_url")

        title_map = {str(b).strip(): str(t).strip() for b, t in zip(books_df[b_col], books_df[t_col])}
        catalogue_ids = set(title_map.keys())

        book_metadata = {}
        for row in books_df.itertuples(index=False):
            r_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(books_df.columns, row))
            b_id = str(r_dict[b_col]).strip()
            book_metadata[b_id] = {
                "book_id": b_id,
                "title": str(r_dict[t_col]).strip(),
                "authors": str(r_dict.get(author_col, "Unknown")).strip() if author_col and pd.notna(r_dict.get(author_col)) else "Unknown",
                "image_url": str(r_dict.get(img_col, "")).strip() if img_col and pd.notna(r_dict.get(img_col)) else "",
            }

        engine = KMeansRecommendationEngine(
            n_clusters=args.clusters,
            rating_threshold=args.threshold,
            random_state=args.seed,
        )

        sparse_features, reader_ids, feature_book_ids = engine.extract_features_from_csv(
            ratings_path=ratings_path,
            catalogue_ids=catalogue_ids,
            chunk_size=CHUNK_SIZE,
        )

        metrics = engine.fit(
            sparse_features=sparse_features,
            title_map=title_map,
            book_metadata=book_metadata,
        )

        cluster_summaries = engine.get_cluster_summaries(top_n_books=6)
        model_id = f"km_cli_{dataset_version[:8]}"
        total_elapsed = round(time.perf_counter() - t0, 3)

        summary_info = {
            "total_catalogue_books": len(catalogue_ids),
            "usable_readers": len(reader_ids),
            "usable_books": len(feature_book_ids),
            "n_clusters": engine.n_clusters,
            "inertia": metrics["inertia"],
            "iterations": metrics["n_iter"],
            "algorithm": metrics["algorithm_used"],
        }

        timing_info = {
            "total_seconds": total_elapsed,
            "feature_extraction_seconds": 0.0,
            "clustering_seconds": metrics["clustering_seconds"],
            "cache_hit": False,
            "source": "cli_kmeans_computation",
        }

        save_cached_cluster_model(
            dataset_version=dataset_version,
            n_clusters=engine.n_clusters,
            rating_threshold=args.threshold,
            random_state=args.seed,
            model_id=model_id,
            summary=summary_info,
            clusters=cluster_summaries,
            metrics=metrics,
            timing=timing_info,
        )

        save_cluster_assignments(
            dataset_version=dataset_version,
            model_id=model_id,
            reader_assignments=engine.reader_assignments,
        )

        record_clustering_run(
            dataset_version=dataset_version,
            n_clusters=engine.n_clusters,
            rating_threshold=args.threshold,
            algorithm="minibatch_kmeans",
            inertia=metrics["inertia"],
            usable_readers=len(reader_ids),
            usable_books=len(feature_book_ids),
            total_seconds=total_elapsed,
            source="cli_kmeans_computation",
            cache_hit=False,
        )

        logger.info(
            "SUCCESS: Pre-clustered %d readers into %d clusters (Inertia: %.2f) and persisted to MongoDB in %.2fs!",
            len(reader_ids),
            engine.n_clusters,
            metrics["inertia"],
            total_elapsed,
        )

    close_db()
    logger.info("Import process completed.")


if __name__ == "__main__":
    main()
