# E-Library Recommendation System (K-Means Reader Clustering)

An intelligent, data-driven book recommendation system powered by **FastAPI (Python)**, **Scikit-Learn K-Means Clustering**, **MongoDB Atlas**, and **React 18 (TypeScript + Vite)**.

The system groups readers into behavioral clusters based on their rating preferences and generates personalized book recommendations using **Bayesian damped score rankings**, while automatically filtering out books the reader has already experienced.

---

## 1. Academic Formulation & Mathematical Foundations

### The Problem Statement
Traditional association-rule mining (e.g. Apriori) discovers coarse global co-occurrence patterns ($A \implies B$) without taking individual user identity into account. In contrast, **K-Means Reader Clustering** models readers as personalized agents in a latent preference space, enabling:
1. Discovery of reader taste communities (e.g., Sci-Fi enthusiasts, classic literature lovers, thriller readers).
2. Elimination of already-rated books to ensure genuine serendipity and novelty.
3. Resilient recommendations for cold-start readers via community-wide Bayesian priors.

### Mathematical Objective of K-Means
Given $N$ readers, each represented by a normalized preference vector $x_i \in \mathbb{R}^d$, K-Means partitions the readers into $K$ disjoint clusters $C = \{C_1, C_2, \dots, C_K\}$ by minimizing the within-cluster sum of squares (inertia):

$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

where:
- $K$ is the number of clusters (configurable, typically $2 \le K \le 20$).
- $C_k$ is the set of readers assigned to cluster $k$.
- $x_i$ is reader $i$'s numerical feature vector.
- $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ is the centroid of cluster $k$.

### Memory-Conscious Feature Engineering Pipeline
In the Goodbooks-10k dataset (`ratings.csv` contains ~6 million rows, ~53,000 readers, ~10,000 books). Constructing a dense $53,000 \times 10,000$ matrix would require **~2.12 GB of RAM**, causing an immediate out-of-memory crash on Render's 512 MB free tier.

To solve this, the engine uses a two-pass streaming sparse representation:
1. **Pass 1 (Streaming Aggregation):** Computes book rating frequencies and global book averages. Selects the top $M$ most active books ($M = 500$) that capture $>80\%$ of reader interactions.
2. **Pass 2 (Sparse CSR Matrix):** Extracts ratings into a `scipy.sparse.csr_matrix` of shape $(N, M)$. For 53,000 readers, this matrix has $\sim 1.5\text{M}$ non-zero entries, consuming **less than 15 MB of RAM**.
3. **Latent Factor Dimensionality Reduction (TruncatedSVD):** Projects the sparse matrix onto $d = 30$ latent preference components, capturing the principal axes of reader interest. The resulting dense embedding consumes only **~6.3 MB of RAM** ($53,000 \times 30 \times 4\text{ bytes}$).
4. **MiniBatchKMeans:** Fits incrementally in batches of 1,024 samples, ensuring low memory consumption ($< 60\text{ MB}$) and fast execution ($< 2.5\text{ seconds}$).

```mermaid
flowchart TD
    RatingsCSV["ratings.csv (68.8 MB, 6M rows)"] -->|Pass 1 Streaming| TopBooks["Select Top M Active Books (M=500)"]
    RatingsCSV -->|Pass 2 Streaming| SparseCSR["scipy.sparse.csr_matrix (N readers x M books) ~15MB"]
    SparseCSR -->|TruncatedSVD (d=30)| LatentMatrix["Dense Latent Embedding (N x 30) ~6.3MB"]
    LatentMatrix -->|MiniBatchKMeans| FitClusters["K-Means Reader Clusters & Centroids"]
    FitClusters --> ClusterSummaries["Compute Cluster Statistics & Bayesian Rankings"]
```

---

## 2. Recommendation Layer & Ranking Formula

K-Means creates reader clusters; it does not directly recommend books. The recommendation layer translates reader cluster affinity into personalized recommendations:

### Candidate Generation & Bayesian Ranking
For reader $u$ assigned to cluster $C_k$:
1. Identify all books rated by readers in cluster $C_k$.
2. **Exclude** any book that reader $u$ has already rated.
3. Score each remaining candidate book $b$ using a **Bayesian Damped Rating**:

$$S(b \mid C_k) = \frac{n_{k, b}}{n_{k, b} + m} \cdot \bar{r}_{k, b} + \frac{m}{n_{k, b} + m} \cdot \bar{R}_k$$

where:
- $n_{k, b}$ is the number of readers in cluster $C_k$ who rated book $b$.
- $\bar{r}_{k, b}$ is the mean rating given to book $b$ within cluster $C_k$.
- $\bar{R}_k$ is the prior mean rating across all books in cluster $C_k$.
- $m$ is the regularization damping factor ($m = 5.0$). This prevents a book with a single 5-star rating from outranking a book with dozens of 4.8-star ratings.

### Cold-Start Handling
If a reader is unknown or not present in the dataset:
- The system automatically triggers the **Cold-Start Fallback Engine**.
- Computes community-wide Bayesian average ratings $S_{\text{global}}(b)$ across the entire library.
- Clearly flags `is_cold_start: true` in the API response and displays a helpful badge in the frontend UI.

---

## 3. Offline Evaluation Protocol

To validate recommendation accuracy academically without data leakage:
- Evaluates readers with $\ge 5$ ratings.
- Holds out $20\%$ of positive test ratings ($\ge 3.0$ stars).
- Generates top $K$ recommendations from the remaining training ratings.
- Computes genuine metrics:
  - **Precision@K**: Proportion of recommended books that match held-out items:
    $$\text{Precision@K} = \frac{|\text{Top } K \cap \text{Holdout}|}{K}$$
  - **Recall@K**: Proportion of held-out items successfully retrieved:
    $$\text{Recall@K} = \frac{|\text{Top } K \cap \text{Holdout}|}{|\text{Holdout}|}$$
  - **Hit Rate@K**: Fraction of users with at least one successful recommendation.

---

## 4. MongoDB Atlas Architecture

### Atlas Free Tier (M0 512 MB) Safeguard
Storing 6 million raw rating rows as separate MongoDB documents requires ~798 MB (BSON overhead + indexes), which would exceed the Atlas M0 512 MB quota.

**Solution:**
- Store the book catalogue (`books`, 10,029 documents, ~2.5 MB).
- Store reader-to-cluster assignments in **compact bucketed documents** (`cluster_assignments`, 5,000 readers per doc, ~10 documents total, ~500 KB).
- Store cluster models and summaries in `cluster_models_cache` and `cluster_summaries` (< 1 MB).
- Total database footprint is **< 5 MB** (< 1% of the 512 MB quota).

### MongoDB Collections Summary

| Collection | Description | Index |
| :--- | :--- | :--- |
| **`books`** | Book metadata (ID, title, author, year, image URL) | `book_id` (Unique), `title` |
| **`cluster_models_cache`** | Cached K-Means models, inertia, iterations, timings | Compound: `(dataset_version, n_clusters, rating_threshold, random_state)` (Unique) |
| **`cluster_assignments`** | Bucketed reader cluster assignments (5,000 readers/doc) | Compound: `(dataset_version, model_id, chunk_idx)` |
| **`cluster_summaries`** | Cluster sizes, percentages, and signature books | Compound: `(dataset_version, model_id, cluster_id)` |
| **`analysis_runs`** | Execution audit history, timings, and sources | `run_id` (Unique), `created_at` |
| **`dataset_metadata`** | Tracked dataset version, row counts, import status | `dataset_version` (Unique) |

---

## 5. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/analyze` or `/cluster` | Primary K-Means clustering pipeline (streams CSVs, fits model, caches in Atlas) |
| `GET` | `/recommend/{reader_id}` | Generates personalized recommendations for reader (excludes read books) |
| `GET` | `/clusters` | Lists all reader clusters and signature books |
| `GET` | `/clusters/{id}` | Detailed statistics for a specific cluster |
| `GET` | `/evaluate` | Offline train/test evaluation (Precision@K, Recall@K, HitRate@K) |
| `GET` | `/health` | Backend status, active model state, database connection |
| `GET` | `/db-health` | MongoDB connection latency, masked host, collection stats |
| `GET` | `/dataset/status` | Stored books count, versions, storage estimates |
| `GET` | `/history` | Recent analysis runs audit trail |
| `DELETE` | `/cache` | Invalidates cached cluster models and assignments |
| `POST` | `/import-dataset` | Standalone one-time dataset ingestion |

---

## 6. Local Setup & Execution

### Prerequisites
- Python 3.11+ (tested on Python 3.14)
- Node.js 18+ and npm
- Active MongoDB Atlas cluster or local MongoDB instance

### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create & activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (copy from template)
cp .env.example .env
# Edit .env and set your MONGODB_URI

# Run automated tests (27 unit and integration tests)
pytest tests/ -v

# Start FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Build for production
npm run build

# Start development server
npm run dev
```

---

## 7. Deployment Guide

### A. MongoDB Atlas Configuration
1. Go to MongoDB Atlas $\to$ **Network Access**.
2. Add IP Address: `0.0.0.0/0` (Allow Access from Anywhere) so Render dynamic backend IPs can connect.
3. In **Database Access**, ensure your database user has read/write privileges on `elibrary`.

### B. Render Deployment (Backend)
1. In Render Dashboard $\to$ New Web Service $\to$ Connect Repository.
2. Root Directory: `backend` (or run from root with `-m backend.app.main`).
3. Build Command: `pip install -r backend/requirements.txt`
4. Start Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
5. Environment Variables:
   - `MONGODB_URI`: Your MongoDB Atlas connection URI (`mongodb+srv://...`)
   - `MONGODB_DB_NAME`: `elibrary`

### C. Vercel Deployment (Frontend)
1. In Vercel $\to$ Import Repository $\to$ Set Root Directory to `frontend`.
2. Build Command: `npm run build`
3. Output Directory: `dist`
4. Environment Variables:
   - `VITE_API_URL`: Your deployed Render backend URL (`https://your-app.onrender.com`).
