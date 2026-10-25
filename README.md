# E-Library Recommendation System

An intelligent, data-driven book recommendation system powered by **FastAPI**, **React (TypeScript + Vite)**, and **WEKA 3.8 Apriori Association Rule Mining**.

The application accepts library catalogue data (`books.csv`) and reader ratings (`ratings.csv`), builds a binary transaction-style preference matrix, executes WEKA Apriori via Java subprocess, and returns actionable book association rules (e.g. *If a reader likes Book A, recommend Book B*).

---

## Architecture & Visual Theme

- **Frontend**: React 18, TypeScript, Vite. Designed with an editorial aesthetic using beige (`#f7f5ef`) and deep dark-green (`#202b24`, `#304738`) tones, typography using *DM Sans*, *DM Mono*, and *Playfair Display*.
- **Backend**: Python 3.11+ / FastAPI, Pandas. Enforces streaming disk-backed file ingestion and structured ARFF generation.
- **Data Mining**: Official WEKA 3.8 (`backend/weka/weka.jar`) executing `weka.associations.Apriori` with Java 17+.

---

## CSV File Requirements & Validation

Both files support up to **100 MiB per file** (enforced on both client and server).

### 1. `books.csv` (Catalogue)
- **Required Columns**:
  - Book ID: `book_id`, `bookid`, or `id` (e.g., `B001`, `1`)
  - Title: `title`, `book_title`, or `name`
- **Validation**:
  - Strips leading/trailing whitespace.
  - Automatically deduplicates multiple entries with the same `book_id` (keeps first).
  - Catalogue must contain at least 2 distinct books.

### 2. `ratings.csv` (Reader Ratings)
- **Required Columns**:
  - Reader/User ID: `user_id`, `userid`, or `reader_id`
  - Book ID: `book_id`, `bookid`, or `item_id`
  - Rating: `rating`, `score`, or `user_rating`
- **Validation**:
  - Rejects empty, missing, or non-numeric rating entries.
  - Rejects scores outside standard scale (0 to 10).
  - Cross-references all book IDs against `books.csv`. Book IDs absent from the catalogue are safely discarded.
  - Automatically deduplicates multiple ratings by the same reader for the same book (retains the latest entry).
  - Dataset must have at least 2 distinct readers and 2 distinct catalogue books.

---

## WEKA Apriori Mining & The Zero-Rules Fix

### Root Cause of the Zero-Rules Issue
In transaction datasets derived from reader ratings (e.g. 500 readers, 25 books, 2,556 ratings):
1. **Mathematical Item Support**: In sparse rating data where each user rates ~5 books out of 25, the highest single-item frequency is typically 10%–20% (e.g. 18.75% in the sample dataset). When `min_support` is configured higher than this (e.g. 0.20 or 20%), **no frequent itemsets can mathematically exist** ($L_1 = \emptyset$), causing WEKA to return `No large itemsets and rules found!`.
2. **Negative Rules Dominance without `-Z`**: Standard binary matrices contain mostly `no` values (~80%–90%). Without the `-Z` flag, WEKA Apriori generates rules between absent preferences (e.g. `Book A=no ==> Book B=no`). Python filtering for `=yes` rules subsequently discards all returned rules, resulting in 0 displayed rules.
3. **Invalid ARFF Syntax**: The original code generated `@ATTRIBUTE ... {yes,?}` where `?` was treated as a literal nominal value rather than standard ARFF nominal values.

### The Solution Implemented
1. **Valid Standard ARFF**: Attributes are written as `@ATTRIBUTE 'B1_Title' {no,yes}`.
2. **WEKA `-Z` Flag**: Passed `-Z` to WEKA Apriori, which treats the first nominal value (`no`, index 0) as missing. This focuses the search space **exclusively** on positive reader preferences (`yes`).
3. **Dynamic Support Diagnostics**: If the configured `min_support` exceeds the highest single-item support in the dataset, the backend calculates the dataset's maximum support and immediately returns an informative diagnostic message:
   > *"No association rules found because the minimum support threshold (20.0%) is higher than the maximum book preference support in this dataset (18.8% for 'The Great Gatsby'). In this dataset, no single book is positively rated by 20.0% of readers. Adjust the minimum support slider to 0.14 (14%) or lower."*
4. **Sensible Defaults**: Default `min_support` set to `0.05` (5%) with slider range `0.01`–`0.50` (step `0.01`), matching WEKA's standard recommendations.

---

## Memory Efficiency & Server Safeguards

- **Direct-to-Disk Streaming**: Uploaded files stream directly to temporary files in 1 MiB chunks. If total bytes exceed 100 MiB, the upload is aborted immediately with **HTTP 413**, preventing server RAM exhaustion.
- **Selective Column Parsing**: Pandas parses only the necessary columns (`book_id`, `title`, `user_id`, `rating`) with memory-efficient string dtypes.
- **Top-Book Capping**: The matrix selects up to the 40 most frequently positively rated books, keeping the Apriori itemset lattice manageable.
- **Guaranteed Cleanup**: All temporary files (`.csv` uploads and `.arff` files) are strictly unlinked in `try...finally` blocks upon request completion or error.
- **Concurrency Lock**: An asynchronous lock ensures only one heavy WEKA Java process runs at a time, preventing CPU and RAM contention.
- **Subprocess Timeout**: 120-second timeout prevents runaway subprocess execution.
- **Practical Constraints**: Apriori's theoretical complexity is $O(2^d \cdot N)$ where $d$ is the number of items and $N$ is the number of transactions. For very large catalogues, limiting candidate books and setting reasonable support thresholds (e.g. 0.02–0.10) ensures fast completion.

---

## Local Setup & Execution Commands

### Prerequisites
- Python 3.11+
- Java Runtime Environment (JRE) 17+ (verify with `java -version`)
- Node.js 18+ and npm (verify with `node -v`)
- WEKA JAR present at `backend/weka/weka.jar`

### 1. Backend Setup & Run
```bash
# Navigate to backend directory
cd backend

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend test suite
pytest tests/ -v

# Start FastAPI backend server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Backend will be live at `http://127.0.0.1:8000` (docs at `http://127.0.0.1:8000/docs`).

### 2. Frontend Setup & Run
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Build production bundle to verify types
npm run build

# Start Vite development server
npm run dev
```
Frontend will be live at `http://localhost:5173`.

---

## Future Storage Expansion Roadmap

While this release maintains the lightweight, zero-database upload-and-analyze workflow with strict cleanup, the application architecture can be expanded for persistent storage and multi-gigabyte datasets without a complete rewrite:

1. **Persistent Object Storage (S3 / MinIO / Azure Blob)**
   - Replace temporary file storage with an S3-compatible bucket.
   - When a user uploads large CSVs, generate a pre-signed upload URL so files stream directly from the browser to cloud storage, bypassing backend server memory and bandwidth entirely.
   - Keep dataset versioning and lifecycle policies (e.g. 7-day auto-purge for ephemeral analysis, or persistent retention for registered library accounts).

2. **Asynchronous Background Task Queue (Celery / RQ + Redis)**
   - For datasets with hundreds of thousands of readers, move WEKA execution off the HTTP request/response cycle into background worker processes.
   - The `/analyze` endpoint returns a `job_id` (HTTP 202 Accepted).
   - The frontend polls or uses Server-Sent Events (SSE) / WebSockets to show a live progress bar ("Validating CSVs" -> "Building Preference Matrix" -> "Running Apriori Cycle 5/15" -> "Completed").

3. **Relational Database (PostgreSQL / SQLite)**
   - Store book catalogue metadata, reader profile mappings, and pre-calculated rating aggregations.
   - Persist mined association rules in a `rules` table with columns `(rule_id, dataset_id, antecedent_book_id, consequent_book_id, support, confidence, lift, created_at)`.
   - Allows instant recommendation queries (`SELECT * FROM rules WHERE antecedent_book_id = :id ORDER BY confidence DESC LIMIT 5`) in sub-millisecond response time without re-running Apriori on every lookup.

4. **Scalable Pre-Processing with DuckDB or Spark**
   - For CSVs in the 100 MiB to 10 GiB range, use DuckDB or PySpark to compute positive ratings, reader-book pivots, and co-occurrence counts directly on disk before generating the ARFF file.
   - This allows the system to process massive interaction logs while keeping the final itemset passed to Apriori bounded and performant.
