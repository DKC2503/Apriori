import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError

# Automatically load environment variables from backend/.env if present
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

logger = logging.getLogger("elibrary.db")

# Singleton MongoDB Client and Database instances
_client: Optional[MongoClient] = None
_db: Optional[Database] = None
_db_connected: bool = False
_server_version: Optional[str] = None
_cluster_host: Optional[str] = None


def get_mongodb_uri() -> Optional[str]:
    """Retrieves the MongoDB URI exclusively from environment variables."""
    return os.getenv("MONGODB_URI")


def get_db_name() -> str:
    """Returns the target MongoDB database name."""
    return os.getenv("MONGODB_DB_NAME", "elibrary")


def get_masked_host(uri: Optional[str] = None) -> str:
    """
    Safely extracts cluster hostname without exposing any credentials or query parameters.
    Example: 'cluster0.6r26tbv.mongodb.net'
    """
    target = uri or get_mongodb_uri()
    if not target:
        return "Not Configured"
    try:
        if "@" in target:
            host_part = target.split("@", 1)[1]
            host = host_part.split("/")[0].split("?")[0]
            return host
        if "://" in target:
            return target.split("://", 1)[1].split("/")[0].split("?")[0]
        return "Configured"
    except Exception:
        return "Configured"


def get_client() -> Optional[MongoClient]:
    """Returns the singleton PyMongo MongoClient instance."""
    global _client
    return _client


def get_db() -> Optional[Database]:
    """Returns the database instance if connected, otherwise None."""
    global _db, _client, _db_connected
    if _db is not None and _db_connected:
        return _db
    if _client is not None and _db_connected:
        _db = _client[get_db_name()]
        return _db
    return None


def is_db_connected() -> bool:
    """Returns True if MongoDB is actively connected and reachable."""
    global _db_connected
    return _db_connected


def init_db(uri: Optional[str] = None) -> bool:
    """
    Initializes the MongoDB client, validates server version, verifies ping,
    and sets up database indexes across all collections.
    Does not crash the application if MongoDB is unreachable (graceful degradation).
    """
    global _client, _db, _db_connected, _server_version, _cluster_host

    mongo_uri = uri or get_mongodb_uri()
    if not mongo_uri:
        logger.warning("MONGODB_URI not set. Running with ephemeral in-memory caching only.")
        _db_connected = False
        return False

    _cluster_host = get_masked_host(mongo_uri)

    try:
        logger.info("Connecting to MongoDB Atlas cluster at [%s]...", _cluster_host)
        # Configure connection pooling and sensible timeouts for Render free tier
        _client = MongoClient(
            mongo_uri,
            maxPoolSize=20,
            minPoolSize=2,
            maxIdleTimeMS=45000,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=30000,
            retryWrites=True,
            w="majority",
            appname="elibrary-recommendation",
        )

        # Quick ping to verify connectivity
        t0 = time.perf_counter()
        _client.admin.command("ping")
        latency_ms = (time.perf_counter() - t0) * 1000

        # Query build info for version verification
        build_info = _client.admin.command("buildInfo")
        _server_version = build_info.get("version", "unknown")

        _db = _client[get_db_name()]
        _db_connected = True
        logger.info(
            "MongoDB Atlas connected successfully! Host: %s | Version: %s | Ping: %.1fms",
            _cluster_host,
            _server_version,
            latency_ms,
        )

        # Ensure collection indexes exist
        _ensure_indexes(_db)
        return True

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error("MongoDB Atlas connection failed: %s. Application will fallback to ephemeral storage.", e)
        _db_connected = False
        return False
    except Exception as e:
        logger.error("Unexpected error initializing MongoDB: %s", e)
        _db_connected = False
        return False


def _ensure_indexes(db: Database):
    """
    Creates necessary indexes idempotently on startup.
    Ensures optimal lookups for books, cached rules, reader preferences, and analysis runs.
    """
    try:
        # 1. Books collection: unique book_id and text/title index
        db["books"].create_index([("book_id", ASCENDING)], unique=True, background=True)
        db["books"].create_index([("title", ASCENDING)], background=True)

        # 2. Ratings collection: compound index on (user_id, book_id)
        db["ratings"].create_index([("user_id", ASCENDING), ("book_id", ASCENDING)], background=True)
        db["ratings"].create_index([("book_id", ASCENDING)], background=True)

        # 3. Reader preferences: compact bucketed compound index on (dataset_version, threshold, chunk_idx)
        db["reader_preferences"].create_index(
            [("dataset_version", ASCENDING), ("threshold", ASCENDING), ("chunk_idx", ASCENDING)],
            unique=True,
            background=True,
        )
        db["reader_preferences"].create_index(
            [("dataset_version", ASCENDING), ("threshold", ASCENDING)],
            background=True,
        )

        # 4. Association rules cache: compound unique index matching exact parameter set (legacy compatibility)
        db["association_rules"].create_index(
            [
                ("dataset_version", ASCENDING),
                ("rating_threshold", ASCENDING),
                ("min_support", ASCENDING),
                ("min_confidence", ASCENDING),
                ("num_rules", ASCENDING),
            ],
            unique=True,
            background=True,
        )
        db["association_rules"].create_index([("created_at", DESCENDING)], background=True)

        # 5. K-Means Cluster Models Cache: unique compound index on model hyper-parameters
        try:
            cache_indexes = [idx["name"] for idx in db["cluster_models_cache"].list_indexes()]
            if "dataset_version_1_n_clusters_1_rating_threshold_1_random_state_1" in cache_indexes:
                db["cluster_models_cache"].drop_index("dataset_version_1_n_clusters_1_rating_threshold_1_random_state_1")
        except Exception:
            pass

        db["cluster_models_cache"].create_index(
            [
                ("dataset_version", ASCENDING),
                ("n_clusters", ASCENDING),
                ("rating_threshold", ASCENDING),
                ("random_state", ASCENDING),
                ("algorithm", ASCENDING),
            ],
            unique=True,
            background=True,
        )
        db["cluster_models_cache"].create_index([("created_at", DESCENDING)], background=True)

        # 6. Cluster Assignments: bucketed reader cluster assignments
        db["cluster_assignments"].create_index(
            [("dataset_version", ASCENDING), ("model_id", ASCENDING), ("chunk_idx", ASCENDING)],
            unique=True,
            background=True,
        )

        # 7. Cluster Summaries: rich cluster descriptors and signature books
        db["cluster_summaries"].create_index(
            [("dataset_version", ASCENDING), ("model_id", ASCENDING), ("cluster_id", ASCENDING)],
            unique=True,
            background=True,
        )

        # 8. Analysis runs collection: run_id and recency sorting
        db["analysis_runs"].create_index([("run_id", ASCENDING)], unique=True, background=True)
        db["analysis_runs"].create_index([("created_at", DESCENDING)], background=True)
        db["analysis_runs"].create_index([("dataset_version", ASCENDING)], background=True)

        # 9. Dataset metadata collection: unique dataset_version
        db["dataset_metadata"].create_index([("dataset_version", ASCENDING)], unique=True, background=True)

        logger.info("Database indexes verified across all collections.")
    except PyMongoError as e:
        logger.warning("Failed to create or verify some MongoDB indexes: %s", e)


def close_db():
    """Closes the MongoDB connection cleanly during application shutdown."""
    global _client, _db, _db_connected
    if _client is not None:
        try:
            logger.info("Closing MongoDB Atlas connection pool...")
            _client.close()
        except Exception as e:
            logger.warning("Error closing MongoDB connection: %s", e)
        finally:
            _client = None
            _db = None
            _db_connected = False
            logger.info("MongoDB Atlas connection closed cleanly.")


def check_db_health() -> Dict[str, Any]:
    """
    Returns a comprehensive diagnostic report of the MongoDB connection,
    including connection status, masked host, server version, ping latency,
    and collection document counts without exposing any sensitive secrets.
    """
    global _client, _db, _db_connected, _server_version, _cluster_host

    uri = get_mongodb_uri()
    if not uri:
        return {
            "status": "unconfigured",
            "connected": False,
            "message": "MONGODB_URI is not set in environment.",
            "cluster_host": "None",
            "server_version": "N/A",
            "latency_ms": None,
            "collections": {},
        }

    host = _cluster_host or get_masked_host(uri)
    if not _db_connected or _client is None:
        return {
            "status": "disconnected",
            "connected": False,
            "message": "MongoDB is configured but currently disconnected.",
            "cluster_host": host,
            "server_version": _server_version or "N/A",
            "latency_ms": None,
            "collections": {},
        }

    try:
        t0 = time.perf_counter()
        _client.admin.command("ping")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        db = _client[get_db_name()]
        coll_counts = {}
        for coll_name in ["books", "ratings", "reader_preferences", "association_rules", "analysis_runs", "dataset_metadata"]:
            try:
                coll_counts[coll_name] = db[coll_name].estimated_document_count()
            except Exception:
                coll_counts[coll_name] = 0

        return {
            "status": "healthy",
            "connected": True,
            "message": "Connected to MongoDB Atlas cluster.",
            "cluster_host": host,
            "database_name": get_db_name(),
            "server_version": _server_version or "8.0+",
            "latency_ms": latency_ms,
            "collections": coll_counts,
        }
    except Exception as e:
        logger.warning("Health ping failed: %s", e)
        return {
            "status": "degraded",
            "connected": False,
            "message": f"Connection ping failed: {str(e)}",
            "cluster_host": host,
            "server_version": _server_version or "N/A",
            "latency_ms": None,
            "collections": {},
        }
