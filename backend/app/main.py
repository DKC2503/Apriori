import os
import re
import gc
import time
import logging
import tempfile
import subprocess
import asyncio
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any, Tuple, Set

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [elibrary] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("elibrary")

app = FastAPI(title="E-Library Recommendation System API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency safeguard: prevent multiple concurrent heavy WEKA analyses from exhausting RAM/CPU
analysis_lock = asyncio.Lock()

# 100 MiB maximum upload limit per file
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MiB in bytes

# Default heap for WEKA Java subprocess (192m fits safely in Render's 512 MB free instance)
DEFAULT_WEKA_HEAP = "192m"

# Number of rows per chunk when reading large CSV files to maintain low memory profile (<100 MB RAM)
CHUNK_SIZE = 150_000


@app.get("/")
def root():
    return {"name": "E-Library Recommendation System", "docs": "/docs"}


@app.get("/health")
def health():
    jar = os.getenv("WEKA_JAR", str(Path(__file__).resolve().parents[1] / "weka" / "weka.jar"))
    jar_exists = Path(jar).exists()
    return {
        "status": "ok",
        "weka_jar_found": jar_exists,
        "max_upload_size_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
        "weka_heap": os.getenv("WEKA_HEAP", DEFAULT_WEKA_HEAP),
    }


def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds a column matching any of the candidate names case-insensitively."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    return next((cols[x] for x in candidates if x in cols), None)


async def save_upload_to_temp(upload: UploadFile, max_size: int = MAX_UPLOAD_SIZE) -> Tuple[Path, int]:
    """
    Streams an uploaded file directly to a temporary file on disk with chunked reading.
    Validates file extension and ensures total size does not exceed max_size.
    Avoids loading entire large files into server memory.
    Returns (temp_path, total_bytes).
    """
    filename = upload.filename or "uploaded_file"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format for '{filename}'. Only CSV files (.csv) are supported.",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_path = Path(temp_file.name)
    total_bytes = 0

    try:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1 MiB chunk
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_size:
                temp_file.close()
                temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{filename}' exceeds the 100 MiB upload limit. Maximum allowed size is 100 MiB.",
                )
            temp_file.write(chunk)
        temp_file.close()

        if total_bytes == 0:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"The uploaded file '{filename}' is empty (0 bytes).",
            )

        return temp_path, total_bytes
    except HTTPException:
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Could not read upload '{filename}': {str(e)}",
        )


def arff_quote(value: Any) -> str:
    """Escapes and quotes an attribute name or value for WEKA ARFF syntax."""
    val_str = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{val_str}'"


def create_arff(
    user_books: Dict[str, Set[str]],
    chosen_book_ids: List[str],
    sanitized_cols: List[str],
    path: Path,
):
    """
    Generates a valid standard WEKA ARFF file directly to disk with zero DataFrame overhead.
    Attributes are declared as binary nominal {no,yes}.
    When used with WEKA's -Z flag, Apriori treats 'no' (the 0th index) as missing,
    ensuring it only mines positive associations (yes) between books.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("@RELATION elibrary_preferences\n\n")
        for col in sanitized_cols:
            f.write(f"@ATTRIBUTE {arff_quote(col)} {{no,yes}}\n")
        f.write("\n@DATA\n")
        for liked in user_books.values():
            row_tokens = ["yes" if bid_v in liked else "no" for bid_v in chosen_book_ids]
            f.write(",".join(row_tokens) + "\n")


def parse_weka_rules(
    weka_stdout: str,
    col_to_title: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Parses association rules from WEKA Apriori stdout.
    Handles single and multi-item antecedents, cleans book titles,
    and extracts confidence, lift, and item counts.
    """
    rules: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*\d+\.\s+(.*?)\s+==>\s+(.*?)\s+<conf:\(([\d.]+)\)>\s+lift:\(([\d.]+)\)",
        re.MULTILINE,
    )

    def extract_titles(side_text: str) -> Tuple[str, str]:
        # Strip trailing item count number (e.g. "B1=yes B2=yes 99" -> "B1=yes B2=yes")
        items_part = re.sub(r"\s+\d+$", "", side_text.strip())
        tokens = items_part.split()
        titles = []
        for token in tokens:
            if "=yes" in token:
                attr_name = token.split("=yes")[0].strip()
                title = col_to_title.get(attr_name)
                if not title:
                    title = re.sub(r"^B\d+_", "", attr_name).replace("_", " ")
                titles.append(title)
            elif "=no" in token:
                attr_name = token.split("=no")[0].strip()
                title = col_to_title.get(attr_name, attr_name)
                titles.append(f"NOT {title}")
        clean_title = " + ".join(titles) if titles else items_part
        return clean_title, side_text.strip()

    for match in pattern.finditer(weka_stdout):
        lhs_raw, rhs_raw, conf_str, lift_str = match.groups()

        # Only retain rules that represent positive preferences
        if "=no" in lhs_raw or "=no" in rhs_raw:
            continue
        if "=yes" not in lhs_raw or "=yes" not in rhs_raw:
            continue

        antecedent_title, antecedent_expr = extract_titles(lhs_raw)
        consequent_title, consequent_expr = extract_titles(rhs_raw)

        rules.append(
            {
                "antecedent_title": antecedent_title,
                "consequent_title": consequent_title,
                "confidence": round(float(conf_str), 4),
                "lift": round(float(lift_str), 2),
                "antecedent": antecedent_expr,
                "consequent": consequent_expr,
            }
        )

    return rules


@app.post("/analyze")
async def analyze(
    books_file: UploadFile = File(...),
    ratings_file: UploadFile = File(...),
    rating_threshold: float = Form(4.0),
    min_support: float = Form(0.05),
    min_confidence: float = Form(0.5),
    num_rules: int = Form(20),
):
    request_start_time = time.time()
    logger.info(
        "--> [START] Analysis requested. books_file='%s', ratings_file='%s', threshold=%.1f, min_support=%.3f, min_confidence=%.2f, num_rules=%d",
        books_file.filename,
        ratings_file.filename,
        rating_threshold,
        min_support,
        min_confidence,
        num_rules,
    )

    # Concurrency guard: reject if another heavy job is actively executing on this instance
    if analysis_lock.locked():
        logger.warning("Rejecting concurrent analysis request (HTTP 429). Instance busy.")
        raise HTTPException(
            status_code=429,
            detail="Another analysis is currently being processed on this instance. Please wait a moment and try again.",
        )

    # Validate threshold ranges
    if not (0.5 <= rating_threshold <= 10.0):
        raise HTTPException(status_code=400, detail="Rating threshold must be between 0.5 and 10.0.")
    if not (0.001 <= min_support <= 1.0):
        raise HTTPException(status_code=400, detail="Minimum support must be between 0.001 and 1.0.")
    if not (0.01 <= min_confidence <= 1.0):
        raise HTTPException(status_code=400, detail="Minimum confidence must be between 0.01 and 1.0.")
    if not (1 <= num_rules <= 100):
        raise HTTPException(status_code=400, detail="Number of rules must be between 1 and 100.")

    # Check WEKA JAR existence early
    jar = os.getenv("WEKA_JAR", str(Path(__file__).resolve().parents[1] / "weka" / "weka.jar"))
    if not Path(jar).exists():
        logger.error("WEKA JAR missing at path: %s", jar)
        raise HTTPException(
            status_code=503,
            detail="WEKA JAR is missing. Please ensure backend/weka/weka.jar exists or configure WEKA_JAR.",
        )

    # Temporary files to be cleaned up guaranteed in finally block
    books_temp_path: Optional[Path] = None
    ratings_temp_path: Optional[Path] = None

    try:
        # ----------------------------------------------------
        # 1. Stream Uploads Directly to Disk (Low RAM Ingestion)
        # ----------------------------------------------------
        t_ingest = time.time()
        books_temp_path, books_bytes = await save_upload_to_temp(books_file)
        ratings_temp_path, ratings_bytes = await save_upload_to_temp(ratings_file)
        logger.info(
            "Streamed uploads to disk in %.2fs: books=%.2f MB, ratings=%.2f MB",
            time.time() - t_ingest,
            books_bytes / (1024 * 1024),
            ratings_bytes / (1024 * 1024),
        )

        # ----------------------------------------------------
        # 2. Inspect & Load books.csv Catalogue
        # ----------------------------------------------------
        try:
            books_header = pd.read_csv(books_temp_path, nrows=0)
        except Exception as e:
            logger.warning("Failed to parse books header: %s", e)
            raise HTTPException(status_code=400, detail=f"Could not parse books.csv: {str(e)}")

        bid = get_col(books_header, ["book_id", "bookid", "id", "item_id", "goodreads_book_id"])
        title = get_col(books_header, ["title", "book_title", "name", "original_title"])

        if not bid or not title:
            found_cols = list(books_header.columns)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"books.csv is missing required columns. Found: {found_cols}. "
                    "Required: a book ID column ('book_id', 'bookid', or 'id') and a title column ('title', 'book_title', or 'name')."
                ),
            )

        try:
            books_df = pd.read_csv(books_temp_path, usecols=[bid, title], dtype=str)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading books.csv: {str(e)}")

        books_df["book_id"] = books_df[bid].astype(str).str.strip()
        books_df["title"] = books_df[title].astype(str).str.strip()
        books_df = books_df.dropna(subset=["book_id", "title"])
        books_df = books_df[(books_df["book_id"] != "") & (books_df["title"] != "")]
        books_df = books_df.drop_duplicates(subset=["book_id"], keep="first")

        if len(books_df) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"The book catalogue must contain at least 2 distinct books. Found {len(books_df)}.",
            )

        catalogue_ids: Set[str] = set(books_df["book_id"])
        title_map: Dict[str, str] = books_df.set_index("book_id")["title"].to_dict()
        total_catalogue_books = len(books_df)
        logger.info("Catalogue validation: %d valid books loaded.", total_catalogue_books)

        # ----------------------------------------------------
        # 3. Inspect ratings.csv Header & Columns
        # ----------------------------------------------------
        try:
            ratings_header = pd.read_csv(ratings_temp_path, nrows=0)
        except Exception as e:
            logger.warning("Failed to parse ratings header: %s", e)
            raise HTTPException(status_code=400, detail=f"Could not parse ratings.csv: {str(e)}")

        uid = get_col(ratings_header, ["user_id", "userid", "reader_id", "user", "reader"])
        rbid = get_col(ratings_header, ["book_id", "bookid", "item_id", "book"])
        score = get_col(ratings_header, ["rating", "score", "user_rating", "stars"])

        if not all([uid, rbid, score]):
            found_cols = list(ratings_header.columns)
            missing = []
            if not uid:
                missing.append("user_id")
            if not rbid:
                missing.append("book_id")
            if not score:
                missing.append("rating")
            raise HTTPException(
                status_code=400,
                detail=(
                    f"ratings.csv is missing required columns ({', '.join(missing)}). "
                    f"Found columns: {found_cols}. Required: user_id, book_id, rating."
                ),
            )

        # ----------------------------------------------------
        # 4. Pass 1: Chunked Streaming Validation & Top Books
        # ----------------------------------------------------
        # Reads ratings.csv in small chunks to keep memory usage under 80 MB even on 70 MB CSVs
        t_pass1 = time.time()
        pos_book_counts: Counter = Counter()
        total_ratings_rows = 0
        total_matched_ratings = 0
        min_score = float("inf")
        max_score = float("-inf")
        sample_readers: Set[str] = set()

        try:
            reader_iter1 = pd.read_csv(
                ratings_temp_path,
                chunksize=CHUNK_SIZE,
                usecols=[uid, rbid, score],
                dtype={uid: str, rbid: str},
                on_bad_lines="skip",
            )
            for chunk in reader_iter1:
                total_ratings_rows += len(chunk)
                chunk[uid] = chunk[uid].astype(str).str.strip()
                chunk[rbid] = chunk[rbid].astype(str).str.strip()
                chunk["_rating"] = pd.to_numeric(chunk[score], errors="coerce")
                chunk = chunk.dropna(subset=[uid, rbid, "_rating"])
                chunk = chunk[(chunk[uid] != "") & (chunk[rbid] != "")]

                # Filter to books in catalogue
                matched = chunk[chunk[rbid].isin(catalogue_ids)]
                if matched.empty:
                    continue

                total_matched_ratings += len(matched)
                min_score = min(min_score, matched["_rating"].min())
                max_score = max(max_score, matched["_rating"].max())

                # Collect sample readers count
                if len(sample_readers) < 100_000:
                    sample_readers.update(matched[uid].unique())

                # Positive rating filter
                pos = matched[matched["_rating"] >= rating_threshold]
                if not pos.empty:
                    # Deduplicate within chunk to avoid overcounting re-ratings
                    pos = pos.drop_duplicates(subset=[uid, rbid])
                    pos_book_counts.update(pos[rbid])

        except Exception as e:
            logger.error("Error in streaming Pass 1: %s", e)
            raise HTTPException(status_code=400, detail=f"Error processing ratings.csv: {str(e)}")

        if total_ratings_rows == 0:
            raise HTTPException(status_code=400, detail="ratings.csv contains no rows.")

        if total_matched_ratings == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "None of the book IDs in ratings.csv match the book IDs in books.csv. "
                    "Please verify that both files share the same book ID system."
                ),
            )

        if min_score < 0 or max_score > 10:
            raise HTTPException(
                status_code=400,
                detail=f"Rating values are out of the expected range (0 to 10). Found values from {min_score} to {max_score}.",
            )

        if not pos_book_counts:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No ratings met the positive rating threshold (>= {rating_threshold}). "
                    f"The maximum rating in the dataset is {max_score}. Try lowering the positive rating threshold."
                ),
            )

        if len(pos_book_counts) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Only {len(pos_book_counts)} book received positive ratings (>= {rating_threshold}). "
                    "At least 2 distinct books with positive ratings are required to discover association rules. "
                    "Try lowering the positive rating threshold."
                ),
            )

        # Select top books (up to 40 books) to keep mining performant and mathematically sound
        chosen_book_ids = [b for b, _ in pos_book_counts.most_common(40)]
        chosen_set = set(chosen_book_ids)
        logger.info(
            "Pass 1 complete in %.2fs: %d rows checked, %d matched catalogue, %d top books selected.",
            time.time() - t_pass1,
            total_ratings_rows,
            total_matched_ratings,
            len(chosen_book_ids),
        )

        # ----------------------------------------------------
        # 5. Pass 2: Stream Preferences Only for Chosen Books
        # ----------------------------------------------------
        # Memory-efficient dictionary mapping user_id -> set of liked chosen book IDs
        t_pass2 = time.time()
        user_books: Dict[str, Set[str]] = defaultdict(set)

        try:
            reader_iter2 = pd.read_csv(
                ratings_temp_path,
                chunksize=CHUNK_SIZE,
                usecols=[uid, rbid, score],
                dtype={uid: str, rbid: str},
                on_bad_lines="skip",
            )
            for chunk in reader_iter2:
                chunk[rbid] = chunk[rbid].astype(str).str.strip()
                # Fast early filter: discard 90%+ rows that are not in chosen top books
                chunk = chunk[chunk[rbid].isin(chosen_set)]
                if chunk.empty:
                    continue
                chunk[uid] = chunk[uid].astype(str).str.strip()
                chunk["_rating"] = pd.to_numeric(chunk[score], errors="coerce")
                pos = chunk[chunk["_rating"] >= rating_threshold]
                for uid_v, bid_v in zip(pos[uid], pos[rbid]):
                    user_books[uid_v].add(bid_v)

        except Exception as e:
            logger.error("Error in streaming Pass 2: %s", e)
            raise HTTPException(status_code=400, detail=f"Error extracting preferences: {str(e)}")

        readers_mined = len(user_books)
        books_mined = len(chosen_book_ids)

        if readers_mined < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Fewer than 2 readers have positive ratings for the top books ({readers_mined} reader found).",
            )

        logger.info(
            "Pass 2 complete in %.2fs: %d readers mined across %d top books.",
            time.time() - t_pass2,
            readers_mined,
            books_mined,
        )

        # ----------------------------------------------------
        # 6. Stream ARFF Directly to Disk & Calculate Supports
        # ----------------------------------------------------
        book_freq = Counter()
        for liked in user_books.values():
            book_freq.update(liked)

        max_item_support = 0.0
        top_book_title = ""
        col_to_title: Dict[str, str] = {}
        sanitized_cols = []

        for i, bid_v in enumerate(chosen_book_ids):
            btitle = title_map.get(bid_v, str(bid_v))
            sname = re.sub(r"[^A-Za-z0-9_]+", "_", f"B{i+1}_{btitle}").strip("_")[:80]
            sanitized_cols.append(sname)
            col_to_title[sname] = btitle
            supp = book_freq[bid_v] / readers_mined
            if supp > max_item_support:
                max_item_support = supp
                top_book_title = btitle

        # Free Python temporary collections before launching Java
        del pos_book_counts
        gc.collect()

        # ----------------------------------------------------
        # 7. Execute WEKA Apriori with Concurrency Protection
        # ----------------------------------------------------
        weka_stdout = ""
        heap = os.getenv("WEKA_HEAP", DEFAULT_WEKA_HEAP)

        with tempfile.TemporaryDirectory() as tmp_dir:
            arff_path = Path(tmp_dir) / "preferences.arff"
            create_arff(user_books, chosen_book_ids, sanitized_cols, arff_path)

            arff_size_mb = arff_path.stat().st_size / (1024 * 1024)
            logger.info("ARFF created: %.2f MB (%d instances)", arff_size_mb, readers_mined)

            cmd = [
                "java",
                f"-Xmx{heap}",
                "--add-opens",
                "java.base/java.lang=ALL-UNNAMED",
                "-cp",
                jar,
                "weka.associations.Apriori",
                "-N",
                str(num_rules),
                "-T",
                "0",
                "-C",
                str(min_confidence),
                "-D",
                "0.05",
                "-U",
                "1.0",
                "-M",
                str(min_support),
                "-S",
                "-1.0",
                "-c",
                "-1",
                "-Z",
                "-t",
                str(arff_path),
            ]

            logger.info("Executing WEKA Apriori with heap=%s, timeout=120s", heap)
            t_weka = time.time()

            async with analysis_lock:
                def run_weka_sync():
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    try:
                        stdout, stderr = proc.communicate(timeout=120)
                        return proc.returncode, stdout, stderr
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate()
                        raise

                try:
                    returncode, stdout, stderr = await asyncio.to_thread(run_weka_sync)
                except subprocess.TimeoutExpired:
                    logger.error("WEKA Apriori timed out after 120s.")
                    raise HTTPException(
                        status_code=408,
                        detail=(
                            "WEKA Apriori timed out after 120 seconds. The dataset or parameter combinations "
                            "may require too much search time. Try increasing minimum support or rating threshold."
                        ),
                    )

            weka_elapsed = time.time() - t_weka
            logger.info("WEKA Apriori finished in %.2fs (exit code %d)", weka_elapsed, returncode)

            if returncode != 0:
                err_msg = (stderr or stdout or "Unknown WEKA error")[-1000:]
                err_msg = re.sub(r"[A-Za-z]:\\[^ \n\r\t]+", "[internal path]", err_msg)
                logger.error("WEKA failed with exit code %d: %s", returncode, err_msg)
                raise HTTPException(status_code=500, detail=f"WEKA execution error: {err_msg}")

            weka_stdout = stdout

        # ----------------------------------------------------
        # 8. Parse Discovered Rules and Build Diagnostics
        # ----------------------------------------------------
        rules = parse_weka_rules(weka_stdout, col_to_title)
        message: Optional[str] = None

        if len(rules) == 0:
            if min_support > max_item_support:
                suggested_supp = max(0.01, round(max_item_support * 0.75, 2))
                message = (
                    f"No association rules found because the minimum support threshold ({min_support*100:.1f}%) "
                    f"is higher than the maximum book preference support in this dataset ({max_item_support*100:.1f}% for '{top_book_title}'). "
                    f"In this dataset, no single book is positively rated by {min_support*100:.1f}% of readers. "
                    f"Adjust the minimum support slider to {suggested_supp:.2f} ({suggested_supp*100:.0f}%) or lower."
                )
            else:
                message = (
                    f"No association rules satisfied both the minimum support ({min_support*100:.1f}%) "
                    f"and minimum confidence ({min_confidence*100:.0f}%) thresholds simultaneously. "
                    "Try lowering the minimum support or minimum confidence sliders."
                )
            logger.info("Zero rules returned. Diagnostic: %s", message)
        else:
            logger.info("Successfully discovered %d association rules.", len(rules))

        total_elapsed = time.time() - request_start_time
        logger.info("<-- [COMPLETED] Analysis completed in %.2fs", total_elapsed)

        return {
            "summary": {
                "books": total_catalogue_books,
                "readers": len(sample_readers) if len(sample_readers) < 100_000 else total_matched_ratings,
                "ratings": total_matched_ratings,
                "books_mined": books_mined,
                "readers_mined": readers_mined,
                "rules": len(rules),
                "max_item_support": round(max_item_support, 4),
            },
            "rules": rules,
            "message": message,
            "weka_output": weka_stdout[-10000:],
        }

    except HTTPException:
        raise
    except Exception as e:
        total_elapsed = time.time() - request_start_time
        logger.exception("Unexpected error in /analyze after %.2fs: %s", total_elapsed, e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Guarantee cleanup of all temporary uploaded CSV files
        t_clean = time.time()
        cleaned = 0
        if books_temp_path and books_temp_path.exists():
            try:
                books_temp_path.unlink()
                cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", books_temp_path, e)
        if ratings_temp_path and ratings_temp_path.exists():
            try:
                ratings_temp_path.unlink()
                cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", ratings_temp_path, e)
        logger.info("Temporary cleanup finished in %.3fs (%d files removed)", time.time() - t_clean, cleaned)
