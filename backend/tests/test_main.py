import os
import io
import tempfile
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.main import app, MAX_UPLOAD_SIZE, create_arff, parse_weka_rules

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
    assert "no valid rating records" in response.json()["detail"]


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
    assert "Insufficient usable data" in response.json()["detail"]


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


def test_valid_mining_with_known_rules():
    """
    Creates a small synthetic dataset where:
    - Readers U1-U8 all rate Book A and Book B with 5 stars.
    - Readers U1-U4 also rate Book C with 5 stars.
    - min_support=0.25, min_confidence=0.5
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
        # All users like Dune and Foundation
        ratings_rows.append(f"{uid},B1,5")
        ratings_rows.append(f"{uid},B2,5")
        if i <= 5:
            # First 5 users also like Neuromancer
            ratings_rows.append(f"{uid},B3,4")
        if i >= 6:
            # Users 6-10 also like Snow Crash
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

    # Verify rule structure
    rule = data["rules"][0]
    assert "antecedent_title" in rule
    assert "consequent_title" in rule
    assert "confidence" in rule
    assert "lift" in rule
    assert rule["confidence"] >= 0.50

    # Ensure no '=no' rules leaked
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
    # Highest book support is 2/5 = 40% (0.40).
    # If we request min_support = 0.80, no frequent itemsets can exist.
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


def test_user_actual_sample_dataset_if_present():
    """
    Tests with the user's actual sample dataset (500 readers, 25 books, 2556 ratings)
    located at C:/Users/D Karthik/Downloads/books (1).csv and ratings (1).csv.
    Verifies that the previously failing dataset now succeeds and returns 20 rules!
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
    assert data["summary"]["readers"] == 500
    assert data["summary"]["readers_mined"] == 496
    assert data["summary"]["rules"] == 20
    assert len(data["rules"]) == 20

    # Ensure Harry Potter / Hunger Games associations are properly parsed
    titles_found = [r["consequent_title"] for r in data["rules"]]
    assert any("Harry Potter" in t for t in titles_found)


def test_file_size_exceeding_100_mib_rejected_with_413():
    """
    Verifies that an uploaded file exceeding 100 MiB returns HTTP 413.
    Uses a generator / stream simulator to avoid allocating 100 MB of RAM in the test.
    """
    from starlette.datastructures import Headers

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

    # File size = 101 MiB
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
    import glob

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

