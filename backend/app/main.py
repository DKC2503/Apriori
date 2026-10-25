import os
import re
import tempfile
import subprocess
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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


@app.get("/")
def root():
    return {"name": "E-Library Recommendation System", "docs": "/docs"}


@app.get("/health")
def health():
    jar = os.getenv("WEKA_JAR", str(Path(__file__).resolve().parents[1] / "weka" / "weka.jar"))
    return {"status": "ok", "weka_jar_found": Path(jar).exists()}


def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    return next((cols[x] for x in candidates if x in cols), None)


async def save_upload_to_temp(upload: UploadFile, max_size: int = MAX_UPLOAD_SIZE) -> Path:
    """
    Streams an uploaded file directly to a temporary file on disk with chunked reading.
    Validates file extension and ensures total size does not exceed max_size.
    Avoids loading entire large files into server memory.
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

        return temp_path
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
    """Escapes and quotes an attribute or value for WEKA ARFF syntax."""
    val_str = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{val_str}'"


def create_arff(matrix: pd.DataFrame, path: Path):
    """
    Generates a valid standard WEKA ARFF file.
    Attributes are declared as binary nominal {no,yes}.
    When used with WEKA's -Z flag, Apriori treats 'no' (the 0th index) as missing,
    ensuring it only mines positive associations (yes) between books.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("@RELATION elibrary_preferences\n\n")
        for col in matrix.columns:
            f.write(f"@ATTRIBUTE {arff_quote(col)} {{no,yes}}\n")
        f.write("\n@DATA\n")
        for row in matrix.itertuples(index=False):
            f.write(",".join(row) + "\n")


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
    # WEKA Apriori rule format:
    # 1. B11_Harry_Potter=yes B6_Order_of_Phoenix=yes 56 ==> B27_Half_Blood_Prince=yes 52    <conf:(0.93)> lift:(3.51) lev:(0.08) [40] conv:(2.18)
    pattern = re.compile(
        r"^\s*\d+\.\s+(.*?)\s+==>\s+(.*?)\s+<conf:\(([\d.]+)\)>\s+lift:\(([\d.]+)\)",
        re.MULTILINE,
    )

    def extract_titles(side_text: str) -> Tuple[str, str]:
        # side_text e.g. "B11_Harry_Potter=yes B6_Order=yes 56" or "B27_Half_Blood_Prince=yes 52"
        # Strip trailing item count number:
        items_part = re.sub(r"\s+\d+$", "", side_text.strip())
        tokens = items_part.split()
        titles = []
        for token in tokens:
            if "=yes" in token:
                attr_name = token.split("=yes")[0].strip()
                title = col_to_title.get(attr_name)
                if not title:
                    # Fallback to cleaning the attribute name
                    title = re.sub(r"^B\d+_", "", attr_name).replace("_", " ")
                titles.append(title)
            elif "=no" in token:
                # In case a negative token appears
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
        raise HTTPException(
            status_code=503,
            detail="WEKA JAR is missing. Please ensure backend/weka/weka.jar exists or configure WEKA_JAR.",
        )

    # Temporary files to be cleaned up guaranteed in finally block
    books_temp_path: Optional[Path] = None
    ratings_temp_path: Optional[Path] = None

    try:
        # Stream uploads directly to temporary files with size validation (up to 100 MiB)
        books_temp_path = await save_upload_to_temp(books_file)
        ratings_temp_path = await save_upload_to_temp(ratings_file)

        # ----------------------------------------------------
        # 1. Inspect and Validate books.csv Header & Columns
        # ----------------------------------------------------
        try:
            books_header = pd.read_csv(books_temp_path, nrows=0)
        except Exception as e:
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

        # Read books with only the required columns and string dtypes for memory efficiency
        try:
            books_df = pd.read_csv(books_temp_path, usecols=[bid, title], dtype=str)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading books.csv: {str(e)}")

        books_df = books_df.rename(columns={bid: "book_id", title: "title"})
        books_df["book_id"] = books_df["book_id"].str.strip()
        books_df["title"] = books_df["title"].str.strip()
        books_df = books_df.dropna(subset=["book_id", "title"])
        books_df = books_df[(books_df["book_id"] != "") & (books_df["title"] != "")]
        books_df = books_df.drop_duplicates(subset=["book_id"], keep="first")

        if len(books_df) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"The book catalogue must contain at least 2 distinct books. Found {len(books_df)}.",
            )

        # ----------------------------------------------------
        # 2. Inspect and Validate ratings.csv Header & Columns
        # ----------------------------------------------------
        try:
            ratings_header = pd.read_csv(ratings_temp_path, nrows=0)
        except Exception as e:
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

        # Read ratings with selected columns
        try:
            ratings_df = pd.read_csv(
                ratings_temp_path,
                usecols=[uid, rbid, score],
                dtype={uid: str, rbid: str},
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading ratings.csv: {str(e)}")

        ratings_df = ratings_df.rename(columns={uid: "user_id", rbid: "book_id", score: "rating"})
        ratings_df["user_id"] = ratings_df["user_id"].astype(str).str.strip()
        ratings_df["book_id"] = ratings_df["book_id"].astype(str).str.strip()
        ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")

        # Drop missing values
        initial_rating_count = len(ratings_df)
        ratings_df = ratings_df.dropna(subset=["user_id", "book_id", "rating"])
        ratings_df = ratings_df[(ratings_df["user_id"] != "") & (ratings_df["book_id"] != "")]

        if ratings_df.empty:
            raise HTTPException(
                status_code=400,
                detail="ratings.csv contains no valid rating records after filtering missing values or non-numeric scores.",
            )

        # Validate rating score sanity (e.g. 0 to 10 scale)
        min_score, max_score = ratings_df["rating"].min(), ratings_df["rating"].max()
        if max_score < 0 or min_score > 10:
            raise HTTPException(
                status_code=400,
                detail=f"Rating values are out of the expected range (0 to 10). Found values from {min_score} to {max_score}.",
            )

        # Check catalogue cross-reference
        catalogue_ids = set(books_df["book_id"])
        matched_ratings = ratings_df[ratings_df["book_id"].isin(catalogue_ids)].copy()

        if matched_ratings.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "None of the book IDs in ratings.csv match the book IDs in books.csv. "
                    "Please verify that both files use the same book identification system."
                ),
            )

        # Deduplicate reader-book pairs, keeping last rating
        matched_ratings = matched_ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")

        total_readers = int(matched_ratings["user_id"].nunique())
        total_catalogue_books = len(books_df)
        total_rated_books = int(matched_ratings["book_id"].nunique())

        if total_readers < 2 or total_rated_books < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Insufficient usable data for association mining: found {total_readers} reader(s) "
                    f"and {total_rated_books} book(s). At least 2 readers and 2 catalogue books are required."
                ),
            )

        # ----------------------------------------------------
        # 3. Apply Positive Rating Threshold
        # ----------------------------------------------------
        pos = matched_ratings[matched_ratings["rating"] >= rating_threshold].copy()
        if pos.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No ratings met the positive rating threshold (>= {rating_threshold}). "
                    f"The maximum rating in the dataset is {matched_ratings['rating'].max()}. "
                    "Try lowering the positive rating threshold."
                ),
            )

        pos_book_count = pos["book_id"].nunique()
        if pos_book_count < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Only {pos_book_count} book received positive ratings (>= {rating_threshold}). "
                    "At least 2 distinct books with positive ratings are required to discover association rules. "
                    "Try lowering the positive rating threshold."
                ),
            )

        # ----------------------------------------------------
        # 4. Form Reader-Book Binary Matrix
        # ----------------------------------------------------
        # Select the most frequently positively rated books (up to 40 books) to keep mining performant
        counts = pos.groupby("book_id")["user_id"].nunique().sort_values(ascending=False)
        chosen_book_ids = counts.head(40).index.tolist()
        pos = pos[pos["book_id"].isin(chosen_book_ids)]

        titles_dict = books_df.set_index("book_id")["title"].to_dict()

        # Build pivot table: only readers who positively rated at least one chosen book
        matrix = (
            pos.assign(val="yes")
            .pivot_table(index="user_id", columns="book_id", values="val", aggfunc="first")
            .reindex(columns=chosen_book_ids)
            .fillna("no")
        )

        readers_mined = len(matrix)
        books_mined = len(matrix.columns)

        if books_mined < 2 or readers_mined < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Insufficient records after matrix preparation: {readers_mined} reader(s) "
                    f"and {books_mined} book(s) qualified. At least 2 readers and 2 books are required."
                ),
            )

        # Map matrix column names to sanitized ARFF attribute names and build title lookup
        col_to_title: Dict[str, str] = {}
        sanitized_cols = []
        for i, bid_val in enumerate(matrix.columns):
            book_title = titles_dict.get(bid_val, str(bid_val))
            sanitized_name = re.sub(r"[^A-Za-z0-9_]+", "_", f"B{i+1}_{book_title}").strip("_")[:80]
            sanitized_cols.append(sanitized_name)
            col_to_title[sanitized_name] = book_title

        matrix.columns = sanitized_cols

        # Compute dataset support statistics for diagnostics
        item_supports = (matrix == "yes").sum(axis=0) / readers_mined
        max_item_support = float(item_supports.max())
        top_attr = item_supports.idxmax()
        top_title = col_to_title.get(top_attr, top_attr)

        logger.info(
            "Mining prepared: %d readers, %d books. Max single item support: %.3f (%s)",
            readers_mined,
            books_mined,
            max_item_support,
            top_title,
        )

        # ----------------------------------------------------
        # 5. Execute WEKA Apriori with Concurrency Protection
        # ----------------------------------------------------
        weka_stdout = ""
        heap = os.getenv("WEKA_HEAP", "512m")

        with tempfile.TemporaryDirectory() as tmp_dir:
            arff_path = Path(tmp_dir) / "preferences.arff"
            create_arff(matrix, arff_path)

            # WEKA command line:
            # -N: number of rules
            # -T 0: metric type = confidence
            # -C: min confidence
            # -D: delta for support decrement
            # -U: upper bound support
            # -M: lower bound min support
            # -S -1.0: no significance testing
            # -c -1: no class index (unsupervised association)
            # -Z: treat zero (nominal index 0 i.e. 'no') as missing -> only mine positive ('yes') itemsets!
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

            # Acquire lock to safeguard server memory from concurrent heavy Java runs
            async with analysis_lock:
                try:
                    # Run synchronously in threadpool to avoid blocking event loop
                    run = await asyncio.to_thread(
                        subprocess.run,
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    raise HTTPException(
                        status_code=408,
                        detail=(
                            "WEKA Apriori timed out after 120 seconds. The dataset or parameter combinations "
                            "may require too much search time. Try increasing minimum support or rating threshold."
                        ),
                    )

            if run.returncode != 0:
                err_msg = (run.stderr or run.stdout or "Unknown WEKA error")[-1000:]
                # Sanitize any local file system paths from error output
                err_msg = re.sub(r"[A-Za-z]:\\[^ \n\r\t]+", "[internal path]", err_msg)
                raise HTTPException(status_code=500, detail=f"WEKA execution error: {err_msg}")

            weka_stdout = run.stdout

        # ----------------------------------------------------
        # 6. Parse Discovered Rules and Build Diagnostics
        # ----------------------------------------------------
        rules = parse_weka_rules(weka_stdout, col_to_title)

        message: Optional[str] = None
        if len(rules) == 0:
            if min_support > max_item_support:
                suggested_supp = max(0.01, round(max_item_support * 0.75, 2))
                message = (
                    f"No association rules found because the minimum support threshold ({min_support*100:.1f}%) "
                    f"is higher than the maximum book preference support in this dataset ({max_item_support*100:.1f}% for '{top_title}'). "
                    f"In this dataset, no single book is positively rated by {min_support*100:.1f}% of readers. "
                    f"Adjust the minimum support slider to {suggested_supp:.2f} ({suggested_supp*100:.0f}%) or lower."
                )
            else:
                message = (
                    f"No association rules satisfied both the minimum support ({min_support*100:.1f}%) "
                    f"and minimum confidence ({min_confidence*100:.0f}%) thresholds simultaneously. "
                    "Try lowering the minimum support or minimum confidence sliders."
                )

        return {
            "summary": {
                "books": total_catalogue_books,
                "readers": total_readers,
                "ratings": len(matched_ratings),
                "books_mined": books_mined,
                "readers_mined": readers_mined,
                "rules": len(rules),
                "max_item_support": round(max_item_support, 4),
            },
            "rules": rules,
            "message": message,
            "weka_output": weka_stdout[-10000:],
        }

    finally:
        # Guarantee cleanup of all temporary uploaded CSV files
        if books_temp_path and books_temp_path.exists():
            try:
                books_temp_path.unlink()
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", books_temp_path, e)
        if ratings_temp_path and ratings_temp_path.exists():
            try:
                ratings_temp_path.unlink()
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", ratings_temp_path, e)
