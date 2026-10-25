import os
import io
import glob
import tempfile
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.main import app, MAX_UPLOAD_SIZE, create_arff, parse_weka_rules, analysis_lock

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "E-Library Recommendation System" in response.json()["name"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["weka_jar_found"] is True
    assert "weka_heap" in data


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
    assert "None of the book IDs in ratings.csv match" in response.json()["detail"]


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


def test_no_positive_ratings():
    books_csv = "book_id,title\n1,Book A\n2,Book B\n"
    ratings_csv = "user_id,book_id,rating\nU1,1,2\nU1,2,3\nU2,1,1\nU2,2,2\n"
    response = client.post(
        "/analyze",
        data={"rating_threshold": 4.0},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 400
    assert "No ratings met the positive rating threshold" in response.json()["detail"]


def test_malformed_csv_data_gracefully_handled():
    """
    Tests that CSV data with malformed rows or irregular whitespace
    is safely parsed with bad lines skipped rather than crashing the server.
    """
    books_csv = (
        "book_id,title\n"
        "B1,Dune\n"
        "B2,Foundation\n"
        "B3,Neuromancer\n"
    )
    # ratings.csv contains a corrupted row with extra unquoted columns
    ratings_csv = (
        "user_id,book_id,rating\n"
        "U1,B1,5\n"
        "U1,B2,5\n"
        "corrupted_line_with_too_many_commas_and_bad_data,extra,extra,extra\n"
        "U2,B1,5\n"
        "U2,B2,5\n"
        "U3,B1,5\n"
        "U3,B2,5\n"
        "U4,B1,5\n"
        "U4,B2,5\n"
    )
    response = client.post(
        "/analyze",
        data={"rating_threshold": 4.0, "min_support": 0.20, "min_confidence": 0.50},
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"]["rules"] > 0


def test_concurrent_analysis_rejection():
    """
    Verifies that when an analysis job holds the analysis_lock,
    a concurrent request receives HTTP 429 rather than running multiple
    memory-heavy Java processes simultaneously.
    """
    import asyncio

    # Simulate active lock
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
        assert "Another analysis is currently being processed" in response.json()["detail"]
    finally:
        analysis_lock._locked = False


def test_weka_execution_failure_handled(monkeypatch):
    """
    Verifies that if WEKA exits with a non-zero exit code or error,
    it returns a clean HTTP 500 error without exposing internal paths.
    """
    import subprocess

    def fake_popen(*args, **kwargs):
        class FakeProc:
            returncode = 1
            def communicate(self, timeout=None):
                return ("", "Exception in thread 'main' java.lang.OutOfMemoryError: Java heap space")
            def kill(self):
                pass
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    books_csv = "book_id,title\nB1,Book 1\nB2,Book 2\n"
    ratings_csv = "user_id,book_id,rating\nU1,B1,5\nU1,B2,5\nU2,B1,5\nU2,B2,5\n"
    response = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert response.status_code == 500
    assert "WEKA execution error" in response.json()["detail"]
    assert "OutOfMemoryError" in response.json()["detail"]


def test_valid_mining_with_known_rules():
    """
    Creates a small synthetic dataset where:
    - Readers U1-U8 all rate Book A and Book B with 5 stars.
    - Readers U1-U4 also rate Book C with 5 stars.
    - min_support=0.20, min_confidence=0.5
    Verifies that WEKA Apriori generates positive rules between Book A and Book B.
    """
    books_csv = (
        "book_id,title\n"
        "B1,Dune\n"
        "B2,Foundation\n"
        "B3,Neuromancer\n"
        "B4,Snow Crash\n"
    )
    ratings_rows = ["user_id,book_id,rating"]
    for i in range(1, 11):
        uid = f"U{i}"
        ratings_rows.append(f"{uid},B1,5")
        ratings_rows.append(f"{uid},B2,5")
        if i <= 5:
            ratings_rows.append(f"{uid},B3,4")
        if i >= 6:
            ratings_rows.append(f"{uid},B4,4")

    ratings_csv = "\n".join(ratings_rows) + "\n"

    response = client.post(
        "/analyze",
        data={
            "rating_threshold": 4.0,
            "min_support": 0.20,
            "min_confidence": 0.50,
            "num_rules": 10,
        },
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"]["rules"] > 0
    assert len(data["rules"]) > 0

    rule = data["rules"][0]
    assert "antecedent_title" in rule
    assert "consequent_title" in rule
    assert "confidence" in rule
    assert "lift" in rule
    assert rule["confidence"] >= 0.50

    for r in data["rules"]:
        assert "=no" not in r["antecedent"]
        assert "=no" not in r["consequent"]


def test_zero_rules_diagnostic_message():
    """
    Tests that when thresholds are higher than the dataset supports,
    the application returns 0 rules with an informative diagnostic message
    explaining the dataset's maximum support and recommending threshold adjustments.
    """
    books_csv = "book_id,title\nB1,Book One\nB2,Book Two\nB3,Book Three\n"
    ratings_csv = (
        "user_id,book_id,rating\n"
        "U1,B1,5\n"
        "U2,B2,5\n"
        "U3,B3,5\n"
        "U4,B1,5\n"
        "U5,B2,5\n"
    )
    response = client.post(
        "/analyze",
        data={
            "rating_threshold": 4.0,
            "min_support": 0.80,
            "min_confidence": 0.50,
            "num_rules": 10,
        },
        files={
            "books_file": ("books.csv", io.BytesIO(books_csv.encode("utf-8")), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["rules"] == 0
    assert len(data["rules"]) == 0
    assert data["message"] is not None
    assert "No association rules found because the minimum support threshold" in data["message"]
    assert "Adjust the minimum support slider to" in data["message"]


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
                "rating_threshold": 4.0,
                "min_support": 0.05,
                "min_confidence": 0.50,
                "num_rules": 20,
            },
            files={
                "books_file": ("books.csv", fb, "text/csv"),
                "ratings_file": ("ratings.csv", fr, "text/csv"),
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"]["books"] == 25
    assert data["summary"]["readers_mined"] == 496
    assert data["summary"]["rules"] == 20
    assert len(data["rules"]) == 20

    titles_found = [r["consequent_title"] for r in data["rules"]]
    assert any("Harry Potter" in t for t in titles_found)


def test_real_large_dataset_upload_68mb():
    """
    Tests the real 68.8 MB ratings.csv (5,976,479 rows) and 3.1 MB books.csv (10,000 books).
    Verifies that the chunked streaming architecture processes this within low memory and
    returns 20 positive association rules without crashing or timing out.
    """
    books_path = Path(r"C:\Users\D Karthik\Downloads\books.csv")
    ratings_path = Path(r"C:\Users\D Karthik\Downloads\ratings.csv")

    if not (books_path.exists() and ratings_path.exists()):
        pytest.skip("Real Goodbooks-10k dataset files not found at Downloads path")

    with open(books_path, "rb") as fb, open(ratings_path, "rb") as fr:
        response = client.post(
            "/analyze",
            data={
                "rating_threshold": 4.0,
                "min_support": 0.05,
                "min_confidence": 0.50,
                "num_rules": 20,
            },
            files={
                "books_file": ("books.csv", fb, "text/csv"),
                "ratings_file": ("ratings.csv", fr, "text/csv"),
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"]["books"] == 10000
    assert data["summary"]["readers_mined"] > 40000
    assert data["summary"]["rules"] == 20
    assert len(data["rules"]) == 20


def test_file_size_exceeding_100_mib_rejected_with_413():
    """
    Verifies that an uploaded file exceeding 100 MiB returns HTTP 413.
    """
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
    """
    Verifies that temporary files created on disk during file upload
    are strictly cleaned up after the request finishes (even after exceptions).
    """
    temp_dir = tempfile.gettempdir()
    before_csvs = set(glob.glob(os.path.join(temp_dir, "tmp*.csv")))

    # 1. Successful request
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
    after_success_csvs = set(glob.glob(os.path.join(temp_dir, "tmp*.csv")))
    assert after_success_csvs == before_csvs, "Temporary CSV was not cleaned up after success!"

    # 2. Failing request (e.g. invalid columns)
    res2 = client.post(
        "/analyze",
        files={
            "books_file": ("books.csv", io.BytesIO(b"bad_header\n1\n"), "text/csv"),
            "ratings_file": ("ratings.csv", io.BytesIO(ratings_csv.encode("utf-8")), "text/csv"),
        },
    )
    assert res2.status_code == 400
    after_error_csvs = set(glob.glob(os.path.join(temp_dir, "tmp*.csv")))
    assert after_error_csvs == before_csvs, "Temporary CSV was not cleaned up after error!"
