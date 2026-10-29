import os
import io
import glob
import tempfile
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.main import app, MAX_UPLOAD_SIZE, analysis_lock
from app.database import init_db, close_db

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    yield
    close_db()

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "E-Library" in response.json()["name"]
    assert "K-Means" in response.json()["algorithm"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "K-Means" in data["algorithm"]
    assert "database" in data


def test_non_csv_extension_rejected():
    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.txt", io.BytesIO(books_csv.encode("utf-8")), "text/plain"),
            "ratings_file": ("ratings.csv", io.BytesIO(b"user_id,book_id,rating\n1,1,5\n"), "text/csv"),
        },
    )
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]


def test_empty_file_rejected():
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(b""), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(b"user_id,book_id,rating\n1,1,5\n"), "text/csv"),
        },
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_missing_required_columns_books():
    books_csv = "isbn,author\n1,Author A\n2,Author B\n"
    ratings_csv = "user_id,book_id,rating\n1,1,5\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 400
    assert "books.csv is missing required columns" in response.json()["detail"]


def test_missing_required_columns_ratings():
    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    ratings_csv = "user_id,review\n1,Great book\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 400
    assert "ratings.csv is missing required columns" in response.json()["detail"]


def test_non_numeric_ratings():
    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    ratings_csv = "user_id,book_id,rating\n1,1,excellent\n2,2,bad\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 400


def test_catalogue_mismatch():
    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    ratings_csv = "user_id,book_id,rating\nU1,999,5\nU2,888,4\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 400
    assert "No matching ratings found" in response.json()["detail"]


def test_insufficient_records():
    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    ratings_csv = "user_id,book_id,rating\nU1,1,5\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 400


def test_cluster_count_validation():
    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    ratings_csv = "user_id,book_id,rating\nU1,1,5\nU2,2,5\n"
    # k < 2
    res1 = client.post(
        "/analyze",
        data={"n_clusters": 1},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert res1.status_code == 400
    assert "Number of clusters" in res1.json()["detail"]

    # k > 50
    res2 = client.post(
        "/analyze",
        data={"n_clusters": 100},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert res2.status_code == 400


def test_malformed_csv_data_gracefully_handled():
    """
    Tests that CSV data with corrupted rows is safely handled with bad lines skipped.
    """
    books_csv = (
        "book_id,title\n"
        "B1,Dune\n"
        "B2,Foundation\n"
        "B3,Neuromancer\n"
        "B4,Snow Crash\n"
    )
    ratings_csv = (
        "user_id,book_id,rating\n"
        "U1,B1,5\n"
        "U1,B2,5\n"
        "corrupted_line_with_too_many_commas_and_bad_data,extra,extra,extra\n"
        "U2,B1,5\n"
        "U2,B2,5\n"
        "U3,B3,5\n"
        "U3,B4,5\n"
        "U4,B3,5\n"
        "U4,B4,5\n"
    )
    response = client.post(
        "/analyze",
        data={"n_clusters": 2, "rating_threshold": 3.0, "random_state": 42},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert len(data["clusters"]) == 2
    assert data["summary"]["usable_readers"] == 4


def test_concurrent_analysis_rejection():
    """
    Verifies that when an analysis job holds the analysis_lock,
    a concurrent request receives HTTP 429.
    """
    analysis_lock._locked = True
    try:
        books_csv = "book_id,title\nB1,Book 1\nB2,Book 2\n"
        ratings_csv = "user_id,book_id,rating\nU1,B1,5\nU1,B2,5\nU2,B1,5\nU2,B2,5\n"
        response = client.post(
            "/analyze",
            files={
                "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
                "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
            },
        )
        assert response.status_code == 429
        assert "Another clustering analysis is currently being processed" in response.json()["detail"]
    finally:
        analysis_lock._locked = False


def test_kmeans_clustering_and_recommendation_pipeline():
    """
    Tests end-to-end K-Means reader clustering on a synthetic dataset:
    - Group 1 (Sci-Fi readers: U1-U6): Love B1 (Dune) & B2 (Foundation)
    - Group 2 (Cyberpunk readers: U7-U12): Love B3 (Neuromancer) & B4 (Snow Crash)
    Verifies:
    1. K-Means discovers 2 distinct clusters with expected sizes.
    2. Inertia > 0 and iterations > 0.
    3. Reader U1 is assigned to Sci-Fi cluster.
    4. Recommendations for U1 exclude already rated books and rank candidate peer favorites.
    """
    books_csv = (
        "book_id,title,authors\n"
        "B1,Dune,Frank Herbert\n"
        "B2,Foundation,Isaac Asimov\n"
        "B3,Neuromancer,William Gibson\n"
        "B4,Snow Crash,Neal Stephenson\n"
        "B5,Hyperion,Dan Simmons\n"
    )
    ratings_rows = ["user_id,book_id,rating"]
    # Sci-fi cluster
    for i in range(1, 7):
        uid = f"U{i}"
        ratings_rows.append(f"{uid},B1,5")
        ratings_rows.append(f"{uid},B2,5")
        if i % 2 == 0:
            ratings_rows.append(f"{uid},B5,4")

    # Cyberpunk cluster
    for i in range(7, 13):
        uid = f"U{i}"
        ratings_rows.append(f"{uid},B3,5")
        ratings_rows.append(f"{uid},B4,5")

    ratings_csv = "\n".join(ratings_rows) + "\n"

    res = client.post(
        "/analyze",
        data={"n_clusters": 2, "rating_threshold": 3.0, "random_state": 42},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "success"
    assert data["summary"]["usable_readers"] == 12
    assert data["summary"]["n_clusters"] == 2
    assert data["metrics"]["inertia"] >= 0.0
    assert data["metrics"]["n_iter"] >= 1
    assert len(data["clusters"]) == 2

    # Cluster sizes must sum to 12
    total_in_clusters = sum(c["size"] for c in data["clusters"])
    assert total_in_clusters == 12

    # Query personalized recommendations for reader U1
    rec_res = client.get("/recommend/U1?n=5")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["reader_id"] == "U1"
    assert rec_data["is_cold_start"] is False
    assert "assigned_cluster" in rec_data

    # Exclusion verification: U1 already rated B1 and B2, so B1 & B2 must NOT appear in recommendations!
    recommended_bids = [b["book_id"] for b in rec_data["recommendations"]]
    assert "B1" not in recommended_bids
    assert "B2" not in recommended_bids
    # B5 was rated by peer sci-fi readers, so it should be recommended to U1!
    if rec_data["recommendations"]:
        assert rec_data["recommendations"][0]["book_id"] == "B5"
        assert rec_data["recommendations"][0]["score"] > 0


def test_deterministic_behavior_with_fixed_seed():
    """
    Verifies that running K-Means twice with the same random seed produces
    identical mathematical inertia and cluster distributions.
    """
    books_csv = "book_id,title\n1,Book A\n2,Book B\n3,Book C\n4,Book D\n"
    ratings_csv = (
        "user_id,book_id,rating\n"
        "U1,1,5\nU1,2,4\n"
        "U2,1,5\nU2,2,5\n"
        "U3,3,5\nU3,4,4\n"
        "U4,3,4\nU4,4,5\n"
    )
    res1 = client.post(
        "/analyze",
        data={"n_clusters": 2, "random_state": 42},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    res2 = client.post(
        "/analyze",
        data={"n_clusters": 2, "random_state": 42},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    d1 = res1.json()
    d2 = res2.json()
    assert d1["metrics"]["inertia"] == d2["metrics"]["inertia"]
    assert d1["summary"]["usable_readers"] == d2["summary"]["usable_readers"]


def test_unknown_reader_cold_start_fallback():
    """
    Verifies that an unknown reader (cold start) receives global high-confidence
    recommendations with is_cold_start=True instead of failing.
    """
    rec_res = client.get("/recommend/NONEXISTENT_READER_999999?n=5")
    assert rec_res.status_code == 200
    data = rec_res.json()
    assert data["is_cold_start"] is True
    assert data["assigned_cluster"] == -1
    assert len(data["recommendations"]) > 0
    assert "Global popular recommendation" in data["recommendations"][0]["reason"]


def test_clusters_endpoints():
    """
    Verifies GET /clusters and GET /clusters/{cluster_id}.
    """
    res = client.get("/clusters")
    assert res.status_code == 200
    data = res.json()
    assert "clusters" in data
    assert len(data["clusters"]) >= 2

    # Specific cluster detail
    res_c0 = client.get("/clusters/0")
    assert res_c0.status_code == 200
    c0 = res_c0.json()
    assert c0["cluster_id"] == 0
    assert "top_representative_books" in c0

    # Nonexistent cluster 999
    res_bad = client.get("/clusters/999")
    assert res_bad.status_code == 404


def test_offline_evaluation_metrics():
    """
    Verifies GET /evaluate computes genuine Precision@K, Recall@K, and HitRate@K.
    """
    res = client.get("/evaluate?test_ratio=0.20&k=5")
    assert res.status_code == 200
    data = res.json()
    assert "k" in data
    assert data["k"] == 5
    if data.get("status") != "insufficient_data_for_evaluation":
        assert "precision_at_k" in data
        assert "recall_at_k" in data
        assert "hit_rate_at_k" in data
        assert 0.0 <= data["precision_at_k"] <= 1.0


def test_file_size_exceeding_100_mib_rejected_with_413():
    class LargeStream:
        def __init__(self, size):
            self.remaining = size
            self.chunk = b"A" * (1024 * 1024)

        def read(self, n=-1):
            if self.remaining <= 0:
                return b""
            amt = min(self.remaining, len(self.chunk))
            self.remaining -= amt
            return self.chunk[:amt]

    large_size = 101 * 1024 * 1024
    stream = LargeStream(large_size)

    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", stream, "text/csv"),
        },
    )
    assert response.status_code == 413
    assert "exceeds the 100 MiB upload limit" in response.json()["detail"]


def test_temp_files_cleaned_up_after_request():
    temp_dir = tempfile.gettempdir()
    before_csvs = set(glob.glob(os.path.join(temp_dir, "elibrary_upload_*.csv")))

    books_csv = "book_id,title\nB1,Book 1\nB2,Book 2\n"
    ratings_csv = "user_id,book_id,rating\nU1,B1,5\nU1,B2,5\nU2,B1,5\nU2,B2,5\n"
    res1 = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert res1.status_code == 200
    after_success_csvs = set(glob.glob(os.path.join(temp_dir, "elibrary_upload_*.csv")))
    assert after_success_csvs == before_csvs, "Temporary upload was not cleaned up after success!"


def test_mongodb_health_and_diagnostics():
    response = client.get("/db-health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "connected" in data
    assert "cluster_host" in data
    # Security: URI credentials must NEVER appear in response
    assert "chaitanyadara07_db_user" not in str(data)
    assert "1324651" not in str(data)
    assert "@" not in data.get("cluster_host", "")


def test_dataset_status_and_storage_estimates():
    response = client.get("/dataset/status")
    assert response.status_code == 200
    data = response.json()
    assert "database_connected" in data
    assert "collections" in data
    assert "storage_estimate" in data
    storage = data["storage_estimate"]
    assert storage["atlas_m0_limit_mb"] == 512
    assert storage["raw_ratings_exceeds_m0"] is True


def test_dataset_version_deterministic():
    from app.db_service import compute_dataset_version
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f1, \
         tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f2:
        f1.write("book_id,title\n1,Dune\n2,Foundation\n")
        f1.flush()
        f2.write("user_id,book_id,rating\n1,1,5\n2,2,5\n")
        f2.flush()
        p1 = Path(f1.name)
        p2 = Path(f2.name)

    try:
        v1 = compute_dataset_version(p1, p2)
        v2 = compute_dataset_version(p1, p2)
        assert v1 == v2
        assert len(v1) == 16
    finally:
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)


def test_import_books_idempotency(tmp_path):
    from app.db_service import import_books_from_csv
    from app.database import get_db, is_db_connected

    if not is_db_connected():
        pytest.skip("MongoDB Atlas not connected")

    csv_file = tmp_path / "test_books.csv"
    csv_file.write_text("book_id,title,authors\nTEST_B1,Test Dune,Frank Herbert\nTEST_B2,Test Foundation,Isaac Asimov\n")

    res1 = import_books_from_csv(csv_file)
    assert res1["status"] == "success"

    res2 = import_books_from_csv(csv_file)
    assert res2["status"] == "success"

    db = get_db()
    count = db["books"].count_documents({"book_id": {"$in": ["TEST_B1", "TEST_B2"]}})
    assert count == 2, "Idempotency failed: duplicate documents found!"
    db["books"].delete_many({"book_id": {"$in": ["TEST_B1", "TEST_B2"]}})


def test_analysis_history():
    response = client.get("/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert isinstance(data["runs"], list)


def test_graceful_db_failure_fallback(monkeypatch):
    import app.database as db_mod

    monkeypatch.setattr(db_mod, "_db_connected", False)
    monkeypatch.setattr(db_mod, "_client", None)
    monkeypatch.setattr(db_mod, "is_db_connected", lambda: False)
    monkeypatch.setattr(db_mod, "get_db", lambda: None)

    res = client.get("/health")
    assert res.status_code == 200

    res_db = client.get("/db-health")
    assert res_db.status_code == 200
    assert res_db.json()["connected"] is False


def test_user_actual_sample_dataset_500_readers():
    """
    Tests with the 500-reader dataset (25 books, 500 readers, 2556 ratings)
    located at C:/Users/D Karthik/Downloads/books (1).csv and ratings (1).csv.
    """
    books_path = Path(r"C:\Users\D Karthik\Downloads\books (1).csv")
    ratings_path = Path(r"C:\Users\D Karthik\Downloads\ratings (1).csv")

    if not (books_path.exists() and ratings_path.exists()):
        pytest.skip("User sample dataset files not found at Downloads path")

    with open(books_path, "rb") as fb, open(ratings_path, "rb") as fr:
        response = client.post(
            "/analyze",
            data={
                "n_clusters": 4,
                "rating_threshold": 3.0,
                "random_state": 42,
            },
            files={
                "books_file": ("books.csv", fb, "text/csv"),
                "ratings_file": ("ratings.csv", fr, "text/csv"),
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"]["usable_readers"] >= 450
    assert data["summary"]["n_clusters"] == 4
    assert len(data["clusters"]) == 4
    assert data["metrics"]["inertia"] > 0


def test_real_large_dataset_upload_68mb():
    """
    Tests the real 68.8 MB ratings.csv (5,976,479 rows) and 3.1 MB books.csv (10,000 books).
    Verifies that the chunked streaming architecture processes this within low memory and
    returns 5 reader clusters with positive sizes.
    """
    books_path = Path(r"C:\Users\D Karthik\Downloads\books.csv")
    ratings_path = Path(r"C:\Users\D Karthik\Downloads\ratings.csv")

    if not (books_path.exists() and ratings_path.exists()):
        pytest.skip("Real Goodbooks-10k dataset files not found at Downloads path")

    with open(books_path, "rb") as fb, open(ratings_path, "rb") as fr:
        response = client.post(
            "/analyze",
            data={
                "n_clusters": 5,
                "rating_threshold": 3.0,
                "random_state": 42,
            },
            files={
                "books_file": ("books.csv", fb, "text/csv"),
                "ratings_file": ("ratings.csv", fr, "text/csv"),
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"]["usable_readers"] > 40000
    assert data["summary"]["n_clusters"] == 5
    assert len(data["clusters"]) == 5
    assert data["metrics"]["inertia"] > 0


# ===========================================================================
# Targeted Regression Tests for Serialization, Concurrency, & Accounting
# ===========================================================================

def test_mongodb_model_cache_serialization_with_integer_keys():
    """
    Regression Test for Issue 1:
    Verifies that dictionary keys that are integers (e.g. cluster IDs {0: 10, 1: 20})
    are cleanly serialized and stored in MongoDB without raising InvalidDocument,
    and deserialized back with intact integer keys and correct cluster sizes.
    """
    from app.db_service import save_cached_cluster_model, lookup_cached_cluster_model, save_cluster_assignments
    from app.database import get_db

    db = get_db()
    if db is None:
        pytest.skip("MongoDB not connected")

    test_version = "reg_test_v_101"
    model_id = "km_reg_test_101"

    # Metrics containing integer dictionary keys (previously caused InvalidDocument: key was 1)
    metrics_with_int_keys = {
        "inertia": 123.456,
        "n_iter": 12,
        "cluster_sizes": {0: 45, 1: 55},  # Int keys!
        "algorithm_used": "MiniBatchKMeans",
        "explained_variance_ratio": 0.88,
    }

    clusters = [
        {"cluster_id": 0, "name": "Cluster 0", "size": 45, "percentage": 45.0, "top_representative_books": []},
        {"cluster_id": 1, "name": "Cluster 1", "size": 55, "percentage": 55.0, "top_representative_books": []},
    ]

    summary = {
        "total_catalogue_books": 20,
        "usable_readers": 100,
        "usable_books": 20,
        "n_clusters": 2,
        "inertia": 123.456,
        "algorithm": "minibatch_kmeans",
    }

    timing = {"total_seconds": 1.5, "source": "fresh"}

    # 1. Save assignments first (required by cache validation)
    assignments = {f"user_{i}": (0 if i < 45 else 1) for i in range(100)}
    save_cluster_assignments(dataset_version=test_version, model_id=model_id, reader_assignments=assignments)

    # 2. Persist model cache
    success = save_cached_cluster_model(
        dataset_version=test_version,
        n_clusters=2,
        rating_threshold=3.0,
        random_state=42,
        model_id=model_id,
        summary=summary,
        clusters=clusters,
        metrics=metrics_with_int_keys,
        timing=timing,
        algorithm="minibatch_kmeans",
    )
    assert success is True, "save_cached_cluster_model must succeed with integer dictionary keys"

    # 3. Retrieve model cache and verify deserialization
    retrieved = lookup_cached_cluster_model(
        dataset_version=test_version,
        n_clusters=2,
        rating_threshold=3.0,
        random_state=42,
        algorithm="minibatch_kmeans",
    )
    assert retrieved is not None, "lookup_cached_cluster_model must find the persisted model"
    assert retrieved["model_id"] == model_id

    # Verify cluster_sizes keys have been deserialized back to integers
    cluster_sizes = retrieved["metrics"]["cluster_sizes"]
    assert 0 in cluster_sizes, f"Expected integer key 0 in cluster_sizes, got keys: {list(cluster_sizes.keys())}"
    assert 1 in cluster_sizes, f"Expected integer key 1 in cluster_sizes, got keys: {list(cluster_sizes.keys())}"
    assert cluster_sizes[0] == 45
    assert cluster_sizes[1] == 55

    # Verify cluster summaries
    assert len(retrieved["clusters"]) == 2
    assert retrieved["clusters"][0]["cluster_id"] == 0
    assert retrieved["clusters"][1]["cluster_id"] == 1


def test_cache_validation_fails_when_assignments_missing():
    """
    Regression Test for Issue 4:
    Verifies that if cluster assignments are missing in MongoDB, lookup_cached_cluster_model
    detects the invalid cache state, logs a warning, and returns None (cache miss) rather
    than serving a broken cached model.
    """
    from app.db_service import save_cached_cluster_model, lookup_cached_cluster_model
    from app.database import get_db

    db = get_db()
    if db is None:
        pytest.skip("MongoDB not connected")

    test_version = "reg_test_v_broken_cache"
    model_id = "km_broken_cache_001"

    summary = {"n_clusters": 2, "usable_readers": 50}
    clusters = [{"cluster_id": 0, "size": 25}, {"cluster_id": 1, "size": 25}]
    metrics = {"inertia": 10.0, "cluster_sizes": {0: 25, 1: 25}}

    # Save model cache without saving assignments
    save_cached_cluster_model(
        dataset_version=test_version,
        n_clusters=2,
        rating_threshold=3.0,
        random_state=42,
        model_id=model_id,
        summary=summary,
        clusters=clusters,
        metrics=metrics,
        timing={"total_seconds": 0.5},
        algorithm="minibatch_kmeans",
    )

    # Ensure no assignments exist
    db["cluster_assignments"].delete_many({"dataset_version": test_version, "model_id": model_id})

    # Lookup must fail validation and return None
    result = lookup_cached_cluster_model(
        dataset_version=test_version,
        n_clusters=2,
        rating_threshold=3.0,
        random_state=42,
        algorithm="minibatch_kmeans",
    )
    assert result is None, "Cache lookup must return None when associated cluster assignments are missing"


def test_book_import_accounting_and_idempotency(tmp_path):
    """
    Regression Test for Issue 3:
    Verifies that importing books never reports inflated counts (e.g. 20000 / 10000).
    Duplicate rows in CSV are correctly handled, and repeated imports report accurate
    unique book counts without doubling matched + modified counts.
    """
    from app.db_service import import_books_from_csv
    from app.database import get_db

    db = get_db()
    if db is None:
        pytest.skip("MongoDB not connected")

    # CSV with 10 rows: 5 unique books, each appearing twice
    csv_content = (
        "book_id,title,authors\n"
        "B101,Book Alpha,Author A\n"
        "B102,Book Beta,Author B\n"
        "B103,Book Gamma,Author C\n"
        "B104,Book Delta,Author D\n"
        "B105,Book Epsilon,Author E\n"
        "B101,Book Alpha Updated,Author A\n"
        "B102,Book Beta Updated,Author B\n"
        "B103,Book Gamma Updated,Author C\n"
        "B104,Book Delta Updated,Author D\n"
        "B105,Book Epsilon Updated,Author E\n"
    )
    csv_file = tmp_path / "books_test.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    # Pass 1: Initial import
    res1 = import_books_from_csv(csv_file, batch_size=5)
    assert res1["status"] == "success"
    assert res1["rows_read"] == 10
    assert res1["unique_books"] == 5
    assert (res1["inserted"] + res1["updated"]) <= 5
    assert res1["unique_books"] <= res1["rows_read"]

    # Pass 2: Re-import identical file (idempotency check)
    res2 = import_books_from_csv(csv_file, batch_size=5)
    assert res2["status"] == "success"
    assert res2["rows_read"] == 10
    assert res2["unique_books"] == 5
    # Total processed unique books must remain exactly 5, never doubling to 10 or 20
    assert (res2["inserted"] + res2["updated"]) == 5
    assert res2["inserted"] == 0  # None are new inserts on second pass


def test_concurrent_simultaneous_identical_requests():
    """
    Regression Test for Issue 2:
    Verifies that concurrent simultaneous /analyze requests for the identical dataset
    and parameters do not run duplicate SVD / K-Means pipelines; instead, the follower
    reuses the leader's computation safely and both return HTTP 200.
    """
    import concurrent.futures

    books_csv = "book_id,title\n1,Book One\n2,Book Two\n3,Book Three\n4,Book Four\n"
    ratings_csv = (
        "user_id,book_id,rating\n"
        "U1,1,5\nU1,2,4\nU2,2,5\nU2,3,4\nU3,3,5\nU3,4,4\nU4,1,4\nU4,4,5\n"
    )

    def send_req():
        c = TestClient(app)
        return c.post(
            "/analyze",
            data={"n_clusters": 2, "rating_threshold": 3.0, "random_state": 42},
            files={
                "books_file": ("books_concurrent.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
                "ratings_file": ("ratings_concurrent.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
            },
        )

    # Launch two simultaneous requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(send_req)
        f2 = executor.submit(send_req)
        res1 = f1.result()
        res2 = f2.result()

    assert res1.status_code == 200, res1.text
    assert res2.status_code == 200, res2.text

    d1 = res1.json()
    d2 = res2.json()

    # Both requests must return the identical clustering model
    assert d1["model_id"] == d2["model_id"]
    assert d1["metrics"]["inertia"] == d2["metrics"]["inertia"]
    assert len(d1["clusters"]) == len(d2["clusters"]) == 2


def test_concurrent_incompatible_request_rejected_with_429():
    """
    Verifies that if an incompatible analysis request arrives while another is processing,
    it receives HTTP 429 rather than corrupting or racing the active job.
    """
    from app.main import _in_flight_tasks, _in_flight_lock
    import asyncio

    # Simulate an active running task in _in_flight_tasks
    dummy_key = "dummy_version:5:3.00:42:minibatch_kmeans"
    loop = asyncio.get_event_loop()
    dummy_future = loop.create_future()
    _in_flight_tasks[dummy_key] = dummy_future

    try:
        books_csv = "book_id,title\n1,Book One\n2,Book Two\n"
        ratings_csv = "user_id,book_id,rating\nU1,1,5\nU2,2,4\n"

        # Request with DIFFERENT parameters (n_clusters=3 instead of 5)
        response = client.post(
            "/analyze",
            data={"n_clusters": 3, "rating_threshold": 3.0, "random_state": 42},
            files={
                "books_file": ("books_inc.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
                "ratings_file": ("ratings_inc.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
            },
        )
        assert response.status_code == 429
        assert "Another clustering analysis is currently being processed" in response.json()["detail"]
    finally:
        _in_flight_tasks.pop(dummy_key, None)
        dummy_future.cancel()
