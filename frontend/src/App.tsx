import { useState } from 'react';

const API = (import.meta as any).env.VITE_API_URL || 'http://localhost:8000';
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MiB in bytes

interface AssociationRule {
  antecedent_title: string;
  consequent_title: string;
  confidence: number;
  lift: number;
  antecedent: string;
  consequent: string;
}

interface MiningSummary {
  books: number;
  readers: number;
  ratings: number;
  books_mined: number;
  readers_mined: number;
  rules: number;
  max_item_support?: number;
}

interface AnalysisResponse {
  summary: MiningSummary;
  rules: AssociationRule[];
  message?: string | null;
  weka_output: string;
}

export default function App() {
  const [books, setBooks] = useState<File | null>(null);
  const [ratings, setRatings] = useState<File | null>(null);
  const [threshold, setThreshold] = useState<number>(4.0);
  const [support, setSupport] = useState<number>(0.05);
  const [confidence, setConfidence] = useState<number>(0.5);
  const [limit, setLimit] = useState<number>(20);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [data, setData] = useState<AnalysisResponse | null>(null);

  function handleFileSelect(file: File | null, type: 'books' | 'ratings') {
    setError('');
    if (!file) {
      if (type === 'books') setBooks(null);
      else setRatings(null);
      return;
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError(`'${file.name}' is not a CSV file. Please upload .csv files.`);
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      setError(`'${file.name}' exceeds the 100 MiB upload limit (${sizeMb} MiB). Maximum allowed size is 100 MiB.`);
      if (type === 'books') setBooks(null);
      else setRatings(null);
      return;
    }

    if (type === 'books') setBooks(file);
    else setRatings(file);
  }

  function formatFileSize(bytes: number): string {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  async function analyze() {
    setError('');
    setData(null);

    if (!books || !ratings) {
      setError('Please upload both books.csv and ratings.csv before running analysis.');
      return;
    }

    if (books.size > MAX_FILE_SIZE || ratings.size > MAX_FILE_SIZE) {
      setError('Each CSV file must be under 100 MiB.');
      return;
    }

    setBusy(true);
    const formData = new FormData();
    formData.append('books_file', books);
    formData.append('ratings_file', ratings);
    formData.append('rating_threshold', String(threshold));
    formData.append('min_support', String(support));
    formData.append('min_confidence', String(confidence));
    formData.append('num_rules', String(limit));

    try {
      const response = await fetch(`${API}/analyze`, {
        method: 'POST',
        body: formData,
      });

      const json = await response.json();
      if (!response.ok) {
        throw new Error(json.detail || 'Analysis request failed.');
      }
      setData(json);
    } catch (e: any) {
      setError(e.message || 'Could not connect to the backend server. Please verify it is running.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="layout">
      <aside>
        <h2>
          folio<span>.</span>
        </h2>
        <small>DATA-DRIVEN LIBRARY</small>
        <p>WORKSPACE</p>
        <b className="active">▧ &nbsp; Book discovery</b>
        <div className="aside-bottom">
          ✦ Powered by WEKA Apriori
          <p>Turn reader preferences into book discoveries.</p>
        </div>
      </aside>

      <main>
        <header>
          Workspace / Book discovery <span>● ASSOCIATION MINING</span>
        </header>

        <section className="hero">
          <div>
            <small>✦ INTELLIGENT BOOK DISCOVERY</small>
            <h1>
              Find the next book<br />
              <i>worth opening.</i>
            </h1>
            <p>
              Upload library data, discover frequent reading patterns, and turn real associations into recommendations.
            </p>
          </div>
          <div className="books-art">
            ▤<sup>✦</sup>
          </div>
        </section>

        <section className="card">
          <small>01 / DATASET</small>
          <h2>Bring your library data</h2>
          <p>Upload the book catalogue and reader ratings as CSV files (up to 100 MiB each).</p>

          <div className="files">
            <label>
              ▤ &nbsp; <b>Book catalogue</b>
              <span>
                {books ? `${books.name} (${formatFileSize(books.size)})` : 'books.csv · book_id, title (max 100 MiB)'}
              </span>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => handleFileSelect(e.target.files?.[0] || null, 'books')}
              />
            </label>

            <label>
              ▦ &nbsp; <b>Reader ratings</b>
              <span>
                {ratings ? `${ratings.name} (${formatFileSize(ratings.size)})` : 'ratings.csv · user_id, book_id, rating (max 100 MiB)'}
              </span>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => handleFileSelect(e.target.files?.[0] || null, 'ratings')}
              />
            </label>
          </div>

          <details>
            <summary>Mining settings</summary>
            <div className="settings">
              <label>
                Positive rating threshold: {threshold.toFixed(1)}
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="0.5"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                />
              </label>

              <label>
                Minimum support: {support.toFixed(2)}
                <input
                  type="range"
                  min="0.01"
                  max="0.50"
                  step="0.01"
                  value={support}
                  onChange={(e) => setSupport(parseFloat(e.target.value))}
                />
              </label>

              <label>
                Minimum confidence: {confidence.toFixed(2)}
                <input
                  type="range"
                  min="0.20"
                  max="1.00"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                />
              </label>

              <label>
                Maximum rules: {limit}
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value, 10))}
                />
              </label>
            </div>
          </details>

          <div className="action">
            <span>ⓘ Unrated books are unknown, not dislikes.</span>
            <button onClick={analyze} disabled={busy}>
              {busy ? 'Running WEKA…' : 'Run Apriori analysis →'}
            </button>
          </div>

          {error && <div className="error">{error}</div>}
        </section>

        {data && (
          <>
            <div className="metrics">
              {Object.entries(data.summary).map(([k, v]) => (
                <div key={k}>
                  <small>{k.replace(/_/g, ' ')}</small>
                  <b>{String(v)}</b>
                </div>
              ))}
            </div>

            <section className="card">
              <small>02 / RESULTS</small>
              <h2>Discovered associations</h2>
              <p>Positive rules returned by WEKA Apriori.</p>

              {data.message && (
                <div className="advisory">
                  <b>✦ Mining Advice & Diagnostics</b>
                  {data.message}
                </div>
              )}

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>IF READER LIKES</th>
                      <th>RECOMMEND</th>
                      <th>CONFIDENCE</th>
                      <th>LIFT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rules.map((r, i) => (
                      <tr key={i}>
                        <td>{r.antecedent_title}</td>
                        <td>{r.consequent_title}</td>
                        <td>{(r.confidence * 100).toFixed(0)}%</td>
                        <td>{r.lift.toFixed(2)}×</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {!data.rules.length && !data.message && (
                  <p>No positive rules returned. Try lowering support or confidence.</p>
                )}

                <details>
                  <summary>Raw WEKA output</summary>
                  <pre>{data.weka_output}</pre>
                </details>
              </div>
            </section>
          </>
        )}

        {!data && (
          <section className="empty">
            <h3>Your next discovery starts with data.</h3>
            <p>Upload both CSV files, adjust mining settings, and run Apriori to reveal book associations.</p>
            01 Upload data　→　02 Mine patterns　→　03 Explore books
          </section>
        )}

        <footer>
          folio. · E-Library Recommendation System <span>Data Mining · WEKA Apriori</span>
        </footer>
      </main>
    </div>
  );
}