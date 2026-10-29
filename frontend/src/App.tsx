import { useState, useRef, useEffect } from 'react';

const API = (import.meta as any).env.VITE_API_URL || 'http://localhost:8000';
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MiB in bytes

interface BookItem {
  book_id: string;
  title: string;
  authors: string;
  image_url?: string;
  bayesian_score?: number;
  score?: number;
  cluster_avg_rating?: number;
  cluster_ratings_count?: number;
  reason?: string;
}

interface ClusterSummary {
  cluster_id: number;
  name: string;
  size: number;
  percentage: number;
  cluster_mean_rating: number;
  top_representative_books: BookItem[];
}

interface ClusteringMetrics {
  n_readers: number;
  n_features: number;
  n_clusters: number;
  inertia: number;
  n_iter: number;
  cluster_sizes: Record<string, number>;
  explained_variance_ratio: number;
  algorithm_used: string;
}

interface TimingBreakdown {
  total_seconds: number;
  ingest_seconds: number;
  catalogue_seconds: number;
  feature_extraction_seconds: number;
  clustering_seconds: number;
  source: string;
  cache_hit: boolean;
}

interface AnalysisResponse {
  status: string;
  model_id: string;
  dataset_version: string;
  summary: {
    total_catalogue_books: number;
    usable_readers: number;
    usable_books: number;
    n_clusters: number;
    inertia: number;
    iterations: number;
    algorithm: string;
  };
  metrics: ClusteringMetrics;
  clusters: ClusterSummary[];
  sample_recommendations: RecommendationResult;
  timing: TimingBreakdown;
}

interface RecommendationResult {
  reader_id: string;
  is_cold_start: boolean;
  assigned_cluster: number;
  already_rated_count: number;
  cluster_size: number;
  recommendations: BookItem[];
  message?: string;
}

interface DbHealth {
  status: string;
  connected: boolean;
  message?: string;
  cluster_host: string;
  database_name?: string;
  server_version?: string;
  latency_ms?: number | null;
  collections?: Record<string, number>;
}

interface DatasetStatus {
  database_connected: boolean;
  cluster_host: string;
  collections: Record<string, number>;
  latest_dataset?: any;
  storage_estimate?: {
    atlas_m0_limit_mb: number;
    books_collection_mb: number;
    raw_ratings_total_mb: number;
    preprocessed_preferences_mb: number;
    recommended_architecture: string;
  };
}

interface AnalysisRunItem {
  run_id: string;
  type?: string;
  dataset_version: string;
  parameters: {
    n_clusters?: number;
    rating_threshold?: number;
    algorithm?: string;
  };
  metrics?: {
    inertia?: number;
    usable_readers?: number;
    usable_books?: number;
  };
  total_seconds: number;
  source: string;
  cache_hit: boolean;
  status: string;
  created_at: string;
}

interface OfflineEvaluation {
  eval_readers_sampled: number;
  k: number;
  precision_at_k: number;
  recall_at_k: number;
  hit_rate_at_k: number;
}

type Stage = 'idle' | 'uploading' | 'features' | 'clustering' | 'complete';

export default function App() {
  const [books, setBooks] = useState<File | null>(null);
  const [ratings, setRatings] = useState<File | null>(null);
  const [nClusters, setNClusters] = useState<number>(5);
  const [threshold, setThreshold] = useState<number>(3.0);
  const [randomState, setRandomState] = useState<number>(42);
  const [algorithm, setAlgorithm] = useState<string>('minibatch_kmeans');

  // Clustering state
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [selectedClusterId, setSelectedClusterId] = useState<number>(0);

  // Personalized Reader Recommendation state
  const [searchReaderId, setSearchReaderId] = useState<string>('1');
  const [recommendationResult, setRecommendationResult] = useState<RecommendationResult | null>(null);
  const [recLoading, setRecLoading] = useState<boolean>(false);
  const [recError, setRecError] = useState<string>('');

  // Offline Evaluation state
  const [evalResult, setEvalResult] = useState<OfflineEvaluation | null>(null);
  const [evalLoading, setEvalLoading] = useState<boolean>(false);

  // Database and system status
  const [dbHealth, setDbHealth] = useState<DbHealth | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [historyRuns, setHistoryRuns] = useState<AnalysisRunItem[]>([]);
  const [showDbDetails, setShowDbDetails] = useState<boolean>(false);

  // Execution & Progress state
  const [stage, setStage] = useState<Stage>('idle');
  const [stageDescription, setStageDescription] = useState<string>('');
  const [uploadPercent, setUploadPercent] = useState<number>(0);
  const [uploadBytesText, setUploadBytesText] = useState<string>('');
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [error, setError] = useState<string>('');

  const timerRef = useRef<number | null>(null);
  const activeXhrRef = useRef<XMLHttpRequest | null>(null);
  const isSubmittingRef = useRef<boolean>(false);

  const busy = stage !== 'idle' && stage !== 'complete';

  useEffect(() => {
    refreshDatabaseStatus();
    return () => {
      isSubmittingRef.current = false;
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      if (activeXhrRef.current) {
        activeXhrRef.current.abort();
      }
    };
  }, []);

  const refreshDatabaseStatus = async () => {
    try {
      const [hRes, dRes, histRes] = await Promise.all([
        fetch(`${API}/db-health`).then((r) => r.json()).catch(() => null),
        fetch(`${API}/dataset/status`).then((r) => r.json()).catch(() => null),
        fetch(`${API}/history?limit=8`).then((r) => r.json()).catch(() => ({ runs: [] })),
      ]);
      if (hRes) setDbHealth(hRes);
      if (dRes) setDatasetStatus(dRes);
      if (histRes?.runs) setHistoryRuns(histRes.runs);
    } catch {
      // Graceful offline fallback
    }
  };

  const handleFile = (file: File | null, type: 'books' | 'ratings') => {
    setError('');
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError(`Only CSV files (.csv) are accepted. Selected: '${file.name}'.`);
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      const mb = (file.size / (1024 * 1024)).toFixed(1);
      setError(`File '${file.name}' (${mb} MiB) exceeds the 100 MiB upload limit.`);
      return;
    }

    if (type === 'books') setBooks(file);
    if (type === 'ratings') setRatings(file);
  };

  const handleClearCache = async () => {
    try {
      await fetch(`${API}/cache`, { method: 'DELETE' });
      setData(null);
      setRecommendationResult(null);
      setEvalResult(null);
      refreshDatabaseStatus();
    } catch {
      // Ignored
    }
  };

  const startClustering = () => {
    if (isSubmittingRef.current || busy) return;
    if (!books || !ratings) {
      setError('Please select both books.csv and ratings.csv files to proceed.');
      return;
    }

    isSubmittingRef.current = true;
    setError('');
    setStage('uploading');
    setStageDescription('Streaming CSV files to server...');
    setUploadPercent(0);
    setUploadBytesText('0%');
    setElapsedSeconds(0);
    setEvalResult(null);

    const startTime = Date.now();
    if (timerRef.current !== null) clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      setElapsedSeconds(Math.round((Date.now() - startTime) / 1000));
    }, 500);

    const formData = new FormData();
    formData.append('books_file', books);
    formData.append('ratings_file', ratings);
    formData.append('n_clusters', String(nClusters));
    formData.append('rating_threshold', String(threshold));
    formData.append('random_state', String(randomState));
    formData.append('algorithm', algorithm);

    const xhr = new XMLHttpRequest();
    activeXhrRef.current = xhr;

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        setUploadPercent(pct);
        const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
        const totalMb = (e.total / (1024 * 1024)).toFixed(1);
        setUploadBytesText(`${pct}% (${loadedMb} / ${totalMb} MB)`);
        if (pct >= 100) {
          setStage('features');
          setStageDescription('Extracting sparse reader rating features & TruncatedSVD...');
        }
      }
    };

    xhr.onload = () => {
      isSubmittingRef.current = false;
      if (timerRef.current !== null) clearInterval(timerRef.current);
      activeXhrRef.current = null;

      if (xhr.status === 200) {
        try {
          const res: AnalysisResponse = JSON.parse(xhr.responseText);
          setData(res);
          setSelectedClusterId(0);
          if (res.sample_recommendations) {
            setRecommendationResult(res.sample_recommendations);
            setSearchReaderId(res.sample_recommendations.reader_id);
          }
          setStage('complete');
          setStageDescription('Clustering completed successfully!');
          refreshDatabaseStatus();
        } catch (e: any) {
          setError(`Could not parse analysis response: ${e.message}`);
          setStage('idle');
        }
      } else {
        try {
          const errRes = JSON.parse(xhr.responseText);
          setError(errRes.detail || `Server returned error status ${xhr.status}.`);
        } catch {
          setError(`Request failed with status ${xhr.status} (${xhr.statusText}).`);
        }
        setStage('idle');
      }
    };

    xhr.onerror = () => {
      isSubmittingRef.current = false;
      if (timerRef.current !== null) clearInterval(timerRef.current);
      activeXhrRef.current = null;
      setError('Network connection error: Failed to reach backend server. Please verify backend is running.');
      setStage('idle');
    };

    xhr.onabort = () => {
      isSubmittingRef.current = false;
      if (timerRef.current !== null) clearInterval(timerRef.current);
      activeXhrRef.current = null;
      setStage('idle');
    };

    xhr.open('POST', `${API}/analyze`);
    xhr.send(formData);
  };

  const fetchRecommendationsForReader = async (rId: string) => {
    if (!rId.trim()) return;
    setRecLoading(true);
    setRecError('');
    try {
      const res = await fetch(`${API}/recommend/${encodeURIComponent(rId.trim())}?n=8`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Failed with status ${res.status}`);
      }
      const recs: RecommendationResult = await res.json();
      setRecommendationResult(recs);
    } catch (e: any) {
      setRecError(e.message || 'Could not retrieve recommendations.');
    } finally {
      setRecLoading(false);
    }
  };

  const runOfflineEvaluation = async () => {
    setEvalLoading(true);
    try {
      const res = await fetch(`${API}/evaluate?test_ratio=0.20&k=10`);
      if (res.ok) {
        const evalJson: OfflineEvaluation = await res.json();
        setEvalResult(evalJson);
      }
    } catch {
      // Ignored
    } finally {
      setEvalLoading(false);
    }
  };

  const selectedCluster = data?.clusters.find((c) => c.cluster_id === selectedClusterId) || data?.clusters[0];

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside>
        <h2>e.<span>library</span></h2>
        <small>K-MEANS CLUSTERING RECS</small>
        <p>RECSYS PLATFORM</p>
        <div className="active">Reader Clustering</div>
        <div className="aside-bottom">
          <p>
            <b>Mathematical K-Means</b> groups readers by rating histories. Candidate books are ranked via cluster-specific Bayesian damped scores.
          </p>
          <div style={{ marginTop: '12px', fontSize: '10px' }}>
            Atlas DB: <b>{dbHealth?.connected ? 'Online' : 'Offline'}</b>
          </div>
        </div>
      </aside>

      <main>
        {/* Header */}
        <header>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <span>K-MEANS ENGINE (SCIKIT-LEARN)</span>
            <div className={`db-pill ${dbHealth?.connected ? 'connected' : 'disconnected'}`}>
              <span className={`db-dot ${dbHealth?.connected ? 'connected' : 'disconnected'}`} />
              MongoDB Atlas: {dbHealth?.connected ? `Connected (${dbHealth.latency_ms?.toFixed(0) || 35}ms)` : 'Offline'}
            </div>
          </div>
          <div>
            API: <span>{API}</span>
          </div>
        </header>

        {/* Hero */}
        <section className="hero">
          <div>
            <small>UNSUPERVISED COLLABORATIVE FILTERING</small>
            <h1>
              K-Means <i>Reader Clustering</i> & Recommendations
            </h1>
            <p>
              Cluster readers based on high-dimensional book preferences minimizing within-cluster variance.
              Generate personalized recommendations using genuine Bayesian damped rating distributions while filtering out books the reader has already experienced.
            </p>
          </div>
          <div className="books-art">
            📚<sup>K</sup>
          </div>
        </section>

        {/* Database Status Banner */}
        <div className="dataset-banner">
          <div>
            <b>MongoDB Atlas Status:</b> {dbHealth?.connected ? `Cluster Host: ${dbHealth.cluster_host}` : 'Running with in-memory caching'}
            {datasetStatus?.collections && (
              <span style={{ marginLeft: '10px' }}>
                | Stored Books: <b>{datasetStatus.collections.books || 0}</b> | Models: <b>{datasetStatus.collections.cluster_models_cache || 0}</b>
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setShowDbDetails(!showDbDetails)}
              style={{ background: 'transparent', border: '1px solid #c2ccbf', borderRadius: '4px', padding: '3px 8px', fontSize: '10px', cursor: 'pointer' }}
            >
              {showDbDetails ? 'Hide Details' : 'View Collections'}
            </button>
            <button
              onClick={handleClearCache}
              style={{ background: 'transparent', border: '1px solid #d99a9a', color: '#8f2e2e', borderRadius: '4px', padding: '3px 8px', fontSize: '10px', cursor: 'pointer' }}
            >
              Clear Cache
            </button>
          </div>
        </div>

        {showDbDetails && datasetStatus?.storage_estimate && (
          <div style={{ background: '#fffefa', border: '1px solid #e1ded3', borderRadius: '8px', padding: '14px', marginBottom: '18px', fontSize: '11px' }}>
            <b>Atlas Storage & Quota Diagnostics:</b>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginTop: '8px' }}>
              <div>Free Quota: <b>{datasetStatus.storage_estimate.atlas_m0_limit_mb} MB</b></div>
              <div>Books Collection: <b>{datasetStatus.storage_estimate.books_collection_mb} MB</b></div>
              <div>Reader Preferences: <b>{datasetStatus.storage_estimate.preprocessed_preferences_mb} MB</b></div>
              <div>Ratings Raw Estimate: <b>{datasetStatus.storage_estimate.raw_ratings_total_mb} MB</b></div>
            </div>
            <div style={{ marginTop: '8px', color: '#576b5a' }}>
              ℹ️ {datasetStatus.storage_estimate.recommended_architecture}
            </div>
          </div>
        )}

        {/* Main Card: Dataset Upload & Configuration */}
        <section className="card">
          <small>STEP 1: INGESTION & HYPERPARAMETERS</small>
          <h2>Dataset Ingestion & Clustering Settings</h2>
          <p>
            Upload <code>books.csv</code> and <code>ratings.csv</code>. The streaming ingestion pipeline extracts sparse reader rating vectors without dense matrix memory blowup.
          </p>

          <div className="files">
            <label>
              <b>Books Catalogue (.csv)</b>
              <span>{books ? `${books.name} (${(books.size / 1024).toFixed(0)} KB)` : 'books.csv (10k books)'}</span>
              <input type="file" accept=".csv" disabled={busy} onChange={(e) => handleFile(e.target.files?.[0] || null, 'books')} />
            </label>
            <label>
              <b>Reader Ratings (.csv)</b>
              <span>{ratings ? `${ratings.name} (${(ratings.size / (1024 * 1024)).toFixed(1)} MB)` : 'ratings.csv (~6M ratings)'}</span>
              <input type="file" accept=".csv" disabled={busy} onChange={(e) => handleFile(e.target.files?.[0] || null, 'ratings')} />
            </label>
          </div>

          <details open>
            <summary>K-Means Hyperparameters & Feature Configuration</summary>
            <div className="settings">
              <label>
                Number of Clusters (K): <b>{nClusters}</b>
                <input
                  type="range"
                  min="2"
                  max="20"
                  step="1"
                  value={nClusters}
                  disabled={busy}
                  onChange={(e) => setNClusters(Number(e.target.value))}
                />
              </label>

              <label>
                Positive Rating Threshold: <b>{threshold.toFixed(1)} ★</b>
                <input
                  type="range"
                  min="1.0"
                  max="5.0"
                  step="0.5"
                  value={threshold}
                  disabled={busy}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
              </label>

              <label>
                Random Seed: <b>{randomState}</b>
                <input
                  type="number"
                  min="0"
                  max="9999"
                  value={randomState}
                  disabled={busy}
                  onChange={(e) => setRandomState(Number(e.target.value))}
                  style={{ width: '100%', padding: '6px', marginTop: '8px', border: '1px solid #d2d6cd', borderRadius: '4px' }}
                />
              </label>

              <label>
                Algorithm:
                <select
                  value={algorithm}
                  disabled={busy}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  style={{ width: '100%', padding: '7px', marginTop: '8px', border: '1px solid #d2d6cd', borderRadius: '4px', background: 'white' }}
                >
                  <option value="minibatch_kmeans">MiniBatchKMeans (Fast, Low RAM)</option>
                  <option value="kmeans">KMeans (Exact Lloyd)</option>
                </select>
              </label>
            </div>
          </details>

          {/* Progress Tracker */}
          {busy && (
            <div className="progress-container">
              <div className="progress-header">
                <span>{stageDescription}</span>
                <span className="instant-tag">ELAPSED: {elapsedSeconds}s</span>
              </div>
              <div className="progress-track">
                {stage === 'uploading' ? (
                  <div className="progress-fill" style={{ width: `${uploadPercent}%` }} />
                ) : (
                  <div className="progress-indeterminate" />
                )}
              </div>
              <div className="progress-detail">
                <span>Stage: {stage.toUpperCase()}</span>
                <span>{stage === 'uploading' ? uploadBytesText : 'Processing in Python memory...'}</span>
              </div>
            </div>
          )}

          {error && <div className="error">{error}</div>}

          <div className="action">
            <span>
              {books && ratings ? `${books.name} + ${ratings.name}` : 'Ready to cluster reader profiles.'}
            </span>
            <div className="action-buttons">
              <button disabled={busy || isSubmittingRef.current || !books || !ratings} onClick={startClustering}>
                {busy || isSubmittingRef.current ? 'Clustering in progress...' : 'Execute K-Means Reader Clustering'}
              </button>
            </div>
          </div>
        </section>

        {/* Results Section */}
        {data && (
          <>
            {/* Timing & Source Badge */}
            <div className="timing-badge">
              <span className={`stage-tag ${data.timing.cache_hit ? 'cache' : 'mining'}`}>
                {data.timing.source.toUpperCase()}
              </span>
              <span>Total Execution: <b>{data.timing.total_seconds}s</b></span>
              <span>Feature Extraction: {data.timing.feature_extraction_seconds}s</span>
              <span>Clustering: {data.timing.clustering_seconds}s</span>
              {data.timing.cache_hit && <span className="instant-tag">⚡ ATLAS CACHE HIT</span>}
            </div>

            {/* Metrics Dashboard */}
            <section className="metrics">
              <div>
                <small>Usable Readers</small>
                <b>{data.summary.usable_readers.toLocaleString()}</b>
              </div>
              <div>
                <small>Feature Books</small>
                <b>{data.summary.usable_books.toLocaleString()}</b>
              </div>
              <div>
                <small>Clusters (K)</small>
                <b>{data.summary.n_clusters}</b>
              </div>
              <div>
                <small>Mathematical Inertia (J)</small>
                <b>{data.summary.inertia.toFixed(1)}</b>
              </div>
            </section>

            {/* Cluster Explorer */}
            <section className="card" style={{ marginTop: '24px' }}>
              <small>STEP 2: EXPLORE READER CLUSTERS</small>
              <h2>Reader Cluster Explorer</h2>
              <p>
                Readers are grouped into {data.clusters.length} distinct behavioral clusters. Click a cluster card to inspect its size and representative signature books.
              </p>

              <div className="cluster-grid">
                {data.clusters.map((c) => (
                  <div
                    key={c.cluster_id}
                    className={`cluster-card ${selectedClusterId === c.cluster_id ? 'active-cluster' : ''}`}
                    onClick={() => setSelectedClusterId(c.cluster_id)}
                  >
                    <h4>{c.name}</h4>
                    <small>{c.size.toLocaleString()} Readers ({c.percentage}%)</small>
                    <div className="cluster-metric">
                      <span>Mean Rating:</span>
                      <b>{c.cluster_mean_rating.toFixed(2)} ★</b>
                    </div>
                  </div>
                ))}
              </div>

              {selectedCluster && (
                <div style={{ marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ margin: 0, fontSize: '15px' }}>
                      Signature Books Favored by {selectedCluster.name} ({selectedCluster.size.toLocaleString()} Readers)
                    </h3>
                    <span style={{ fontSize: '11px', color: '#687869' }}>
                      Ranked by Bayesian Damped Score (m = 5.0)
                    </span>
                  </div>

                  <div className="book-grid">
                    {selectedCluster.top_representative_books.map((b) => (
                      <div key={b.book_id} className="book-card">
                        {b.image_url ? (
                          <img src={b.image_url} alt={b.title} className="book-cover" />
                        ) : (
                          <div className="book-cover-placeholder">📖</div>
                        )}
                        <div className="book-details">
                          <div>
                            <div className="book-title" title={b.title}>{b.title}</div>
                            <div className="book-author">{b.authors}</div>
                          </div>
                          <div>
                            <div className="book-badge">
                              ★ {b.cluster_avg_rating} ({b.cluster_ratings_count} ratings)
                            </div>
                            <div className="book-reason">Score: {b.bayesian_score}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* Personalized Reader Recommendation Engine */}
            <section className="card" style={{ marginTop: '24px' }}>
              <small>STEP 3: PERSONALIZED RECOMMENDATION</small>
              <h2>Personalized Book Recommendations for Readers</h2>
              <p>
                Enter a Reader ID to identify their cluster, retrieve peer preferences, and exclude books they have already rated.
              </p>

              <div className="reader-search-box">
                <input
                  type="text"
                  placeholder="Enter Reader ID (e.g. 1, 2, 10)..."
                  value={searchReaderId}
                  onChange={(e) => setSearchReaderId(e.target.value)}
                />
                <button
                  disabled={recLoading || !searchReaderId.trim()}
                  onClick={() => fetchRecommendationsForReader(searchReaderId)}
                >
                  {recLoading ? 'Retrieving...' : 'Generate Personalized Recommendations'}
                </button>
              </div>

              <div className="reader-chips">
                <span style={{ fontSize: '10px', color: '#7a8276' }}>Quick Test IDs:</span>
                {['1', '2', '5', '10', '25', '999999'].map((id) => (
                  <span
                    key={id}
                    className="reader-chip"
                    onClick={() => {
                      setSearchReaderId(id);
                      fetchRecommendationsForReader(id);
                    }}
                  >
                    {id === '999999' ? 'Unknown (Cold-Start)' : `Reader #${id}`}
                  </span>
                ))}
              </div>

              {recError && <div className="error">{recError}</div>}

              {recommendationResult && (
                <div style={{ marginTop: '20px' }}>
                  {recommendationResult.is_cold_start ? (
                    <div className="cold-start-banner">
                      <span>❄️</span>
                      <div>
                        <b>Cold-Start Reader Detected:</b> {recommendationResult.message || 'Showing community-wide Bayesian favorites.'}
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '12px', color: '#27382b', margin: '10px 0' }}>
                      <span className="stage-tag prep">CLUSTER {recommendationResult.assigned_cluster}</span>
                      <span>
                        Reader <b>#{recommendationResult.reader_id}</b> belongs to <b>Cluster {recommendationResult.assigned_cluster}</b>.
                        Filtered out <b>{recommendationResult.already_rated_count}</b> previously rated books.
                      </span>
                    </div>
                  )}

                  <div className="book-grid">
                    {recommendationResult.recommendations.map((b) => (
                      <div key={b.book_id} className="book-card">
                        {b.image_url ? (
                          <img src={b.image_url} alt={b.title} className="book-cover" />
                        ) : (
                          <div className="book-cover-placeholder">📖</div>
                        )}
                        <div className="book-details">
                          <div>
                            <div className="book-title" title={b.title}>{b.title}</div>
                            <div className="book-author">{b.authors}</div>
                          </div>
                          <div>
                            <div className="book-badge">
                              ★ {b.cluster_avg_rating} ({b.cluster_ratings_count} cluster reviews)
                            </div>
                            <div className="book-reason">{b.reason}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* Offline Evaluation Section */}
            <section className="card" style={{ marginTop: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <small>STEP 4: MODEL EVALUATION</small>
                  <h2 style={{ margin: '4px 0' }}>Offline Evaluation on Held-Out Ratings</h2>
                  <p style={{ margin: '4px 0' }}>
                    Evaluates recommendation accuracy using an 80/20 train-test split on active readers.
                  </p>
                </div>
                <button
                  className="btn-secondary"
                  disabled={evalLoading}
                  onClick={runOfflineEvaluation}
                  style={{ padding: '8px 14px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer' }}
                >
                  {evalLoading ? 'Evaluating...' : 'Run Offline Evaluation'}
                </button>
              </div>

              {evalResult && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginTop: '16px' }}>
                  <div style={{ background: '#f5f7f2', padding: '12px', borderRadius: '8px' }}>
                    <small style={{ color: '#667064' }}>Readers Evaluated</small>
                    <b style={{ display: 'block', fontSize: '18px', marginTop: '4px' }}>{evalResult.eval_readers_sampled}</b>
                  </div>
                  <div style={{ background: '#f5f7f2', padding: '12px', borderRadius: '8px' }}>
                    <small style={{ color: '#667064' }}>Precision@{evalResult.k}</small>
                    <b style={{ display: 'block', fontSize: '18px', marginTop: '4px' }}>{(evalResult.precision_at_k * 100).toFixed(1)}%</b>
                  </div>
                  <div style={{ background: '#f5f7f2', padding: '12px', borderRadius: '8px' }}>
                    <small style={{ color: '#667064' }}>Recall@{evalResult.k}</small>
                    <b style={{ display: 'block', fontSize: '18px', marginTop: '4px' }}>{(evalResult.recall_at_k * 100).toFixed(1)}%</b>
                  </div>
                  <div style={{ background: '#f5f7f2', padding: '12px', borderRadius: '8px' }}>
                    <small style={{ color: '#667064' }}>Hit Rate@{evalResult.k}</small>
                    <b style={{ display: 'block', fontSize: '18px', marginTop: '4px' }}>{(evalResult.hit_rate_at_k * 100).toFixed(1)}%</b>
                  </div>
                </div>
              )}
            </section>
          </>
        )}

        {/* MongoDB Audit Log & Analysis History */}
        {historyRuns.length > 0 && (
          <section className="card history-wrap">
            <small>AUDIT TRAIL</small>
            <h2>Recent Clustering Analyses (MongoDB Atlas)</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>RUN ID</th>
                    <th>DATASET VERSION</th>
                    <th>CLUSTERS (K)</th>
                    <th>INERTIA</th>
                    <th>DURATION</th>
                    <th>SOURCE</th>
                    <th>TIMESTAMP</th>
                  </tr>
                </thead>
                <tbody>
                  {historyRuns.map((r) => (
                    <tr key={r.run_id}>
                      <td><code>{r.run_id.slice(0, 8)}</code></td>
                      <td><code>{r.dataset_version}</code></td>
                      <td>{r.parameters.n_clusters || 5}</td>
                      <td>{r.metrics?.inertia ? r.metrics.inertia.toFixed(1) : '-'}</td>
                      <td>{r.total_seconds}s</td>
                      <td>
                        <span className={r.cache_hit ? 'history-badge-cache' : 'history-badge-run'}>
                          {r.source}
                        </span>
                      </td>
                      <td>{new Date(r.created_at).toLocaleTimeString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer>
          <span>E-Library Recommendation System &copy; 2026</span>
          <span>FastAPI &bull; Scikit-Learn K-Means &bull; React &bull; MongoDB Atlas</span>
        </footer>
      </main>
    </div>
  );
}
