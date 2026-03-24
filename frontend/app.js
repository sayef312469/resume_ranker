const { useState, useEffect, useRef } = React;

// ─── API HELPERS ────────────────────────────────────────
const api = {
  async ping() {
    try {
      const r = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
      return r.ok;
    } catch { return false; }
  },
  async predictCategory(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`${API_BASE}/predict-category`, { method:'POST', body:fd });
    if (!r.ok) throw new Error(`Server error ${r.status}`);
    return r.json();
  },
  async scoreCV(file, jdText) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('jd_text', jdText);
    const r = await fetch(`${API_BASE}/score-cv`, { method:'POST', body:fd });
    if (!r.ok) throw new Error(`Server error ${r.status}`);
    return r.json();
  },
  async rankCVs(files, jdText) {
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    fd.append('jd_text', jdText);
    const r = await fetch(`${API_BASE}/rank-cvs`, { method:'POST', body:fd });
    if (!r.ok) throw new Error(`Server error ${r.status}`);
    return r.json();
  }
};

// ─── UTILS ──────────────────────────────────────────────
const scoreBadgeClass = s => s >= 70 ? 'sb-high' : s >= 45 ? 'sb-mid' : 'sb-low';
const scoreStars = s => {
  if (s >= 80) return '⭐⭐⭐⭐⭐';
  if (s >= 65) return '⭐⭐⭐⭐';
  if (s >= 50) return '⭐⭐⭐';
  if (s >= 35) return '⭐⭐';
  return '⭐';
};

// ─── UPLOAD ZONE ────────────────────────────────────────
function UploadZone({ file, onFile, onClear, multi=false, files=[], onFiles }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef();

  const handleDrop = e => {
    e.preventDefault(); setDrag(false);
    if (multi) onFiles([...e.dataTransfer.files]);
    else onFile(e.dataTransfer.files[0]);
  };

  if (!multi && file) return (
    <div className="file-chip">
      <span>📎</span>
      <span className="file-chip-name">{file.name}</span>
      <span style={{fontSize:'0.75rem',color:'var(--text-muted)'}}>
        {(file.size/1024).toFixed(0)} KB
      </span>
      <button className="chip-remove" onClick={onClear}>✕</button>
    </div>
  );

  return (
    <div>
      <div
        className={`upload-zone${drag?' drag':''}`}
        onDragOver={e=>{e.preventDefault();setDrag(true)}}
        onDragLeave={()=>setDrag(false)}
        onDrop={handleDrop}
        onClick={()=>ref.current.click()}
      >
        <input
          ref={ref} type="file" accept=".pdf,.docx,.txt"
          multiple={multi} style={{display:'none'}}
          onChange={e=>{
            if(multi) onFiles([...e.target.files]);
            else onFile(e.target.files[0]);
          }}
        />
        <span className="upload-ico">{multi?'📁':'📄'}</span>
        <div className="upload-title">{multi?'Drop multiple CVs here':'Drop your CV here'}</div>
        <div className="upload-sub">{multi?'PDF, DOCX or TXT · Multiple files':'PDF, DOCX or TXT · Max 10MB'}</div>
      </div>

      {multi && files.length > 0 && (
        <div className="file-list">
          {files.map((f,i) => (
            <div className="file-item" key={i} style={{animationDelay:`${i*0.05}s`}}>
              <span>📎</span>
              <span className="file-item-n">{f.name}</span>
              <span style={{fontSize:'0.75rem',color:'var(--text-muted)'}}>
                {(f.size/1024).toFixed(0)} KB
              </span>
              <button className="chip-remove"
                onClick={()=>onFiles(files.filter((_,j)=>j!==i))}>✕</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── SPINNER ────────────────────────────────────────────
function Spinner({ text }) {
  return (
    <div className="spinner-wrap">
      <div className="spinner"/>
      <div className="spinner-txt">{text||'Analyzing with AI...'}</div>
    </div>
  );
}

// ─── TAB 1: PREDICT CATEGORY ────────────────────────────
function PredictTab() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const reset = () => { setFile(null); setResult(null); setError(''); };

  const run = async () => {
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await api.predictCategory(file);
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch(e) {
      setError(e.message||'Failed. Is the backend running?');
    } finally { setLoading(false); }
  };

  return (
    <div>
      <div className="card" style={{animation:'fadeUp 0.5s ease both'}}>
        <div className="card-head">
          <span className="card-icon">🎯</span>
          <span className="card-title">CV Category Predictor</span>
        </div>
        <p className="card-sub">
          Upload your CV and our SVM classifier (trained on fine-tuned SBERT embeddings)
          will predict the top job categories you are best suited for — across 20 professional domains.
        </p>
        <UploadZone file={file} onFile={setFile} onClear={()=>setFile(null)}/>
        <button
          className="btn btn-primary"
          onClick={run}
          disabled={!file||loading}
          style={{marginTop:'18px'}}
        >
          {loading?'⏳ Analyzing...':'🎯 Predict My Best Categories'}
        </button>
      </div>

      {loading && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <Spinner text="Running SVM classifier on SBERT embeddings..."/>
        </div>
      )}

      {error && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <div className="alert alert-err">⚠️ {error}</div>
          <div className="info-box">
            💡 Make sure the backend is running: <code style={{background:'rgba(255,255,255,0.1)',padding:'2px 6px',borderRadius:'4px'}}>python backend/main.py</code>
          </div>
        </div>
      )}

      {result && !loading && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <div className="res-hdr">
            <span className="res-title">📊 Prediction Results</span>
            <button className="btn btn-ghost" onClick={reset}>↩ Reset</button>
          </div>

          <div className="info-box">
            ℹ️ Analyzed <b style={{color:'var(--text)'}}>{result.cv_length?.toLocaleString()} chars</b> · 
            Best match: <b style={{color:'var(--accent)'}}>{result.best_match?.replace(/-/g,' ')}</b> · 
            Confidence: <b style={{color:'var(--accent)'}}>{result.confidence?.toFixed(1)}%</b>
          </div>

          <div className="cat-grid">
            {result.predictions?.map((p,i) => (
              <div
                key={i}
                className={`cat-card${i===0?' top':''}`}
                style={{animationDelay:`${i*0.07}s`}}
              >
                <div className="cat-label">
                  <span className="cat-name">{i===0&&'🏆 '}{p.category?.replace(/-/g,' ')}</span>
                  <span className="cat-pct">{p.confidence?.toFixed(1)}%</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{width:`${p.confidence}%`}}/>
                </div>
                <div className="cat-level">{p.match_level}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── TAB 2: SCORE CV ────────────────────────────────────
function ScoreTab() {
  const [file, setFile] = useState(null);
  const [jd, setJd] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const reset = () => { setFile(null); setJd(''); setResult(null); setError(''); };
  const canRun = file && jd.trim().length > 20 && !loading;

  const run = async () => {
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await api.scoreCV(file, jd);
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch(e) {
      setError(e.message||'Failed. Is the backend running?');
    } finally { setLoading(false); }
  };

  const s = result?.score;
  const g = result?.gap_analysis;

  return (
    <div>
      <div className="card" style={{animation:'fadeUp 0.5s ease both'}}>
        <div className="card-head">
          <span className="card-icon">📊</span>
          <span className="card-title">CV Quality Scorer</span>
        </div>
        <p className="card-sub">
          Upload your CV and paste a Job Description to get a semantic similarity score
          powered by fine-tuned SBERT. Includes keyword gap analysis so you know exactly
          what to improve before submitting.
        </p>

        <UploadZone file={file} onFile={setFile} onClear={()=>setFile(null)}/>

        <div className="divider">then paste the job description</div>

        <div className="form-group">
          <label className="lbl">Job Description</label>
          <textarea
            value={jd}
            onChange={e=>setJd(e.target.value)}
            placeholder="Paste the full job description here...&#10;E.g. We are looking for a Python Developer with experience in Django, REST APIs, PostgreSQL, Docker, Git, CI/CD..."
          />
        </div>

        <button className="btn btn-primary" onClick={run} disabled={!canRun}>
          {loading?'⏳ Scoring...':'📊 Score My CV Against This JD'}
        </button>
      </div>

      {loading && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <Spinner text="Computing semantic similarity with fine-tuned SBERT..."/>
        </div>
      )}

      {error && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <div className="alert alert-err">⚠️ {error}</div>
        </div>
      )}

      {result && s && !loading && (
        <div style={{animation:'fadeIn 0.3s ease'}}>
          <div className="card">
            <div className="res-hdr">
              <span className="res-title">🎯 CV Match Score</span>
              <button className="btn btn-ghost" onClick={reset}>↩ Reset</button>
            </div>
            <div className="score-wrap">
              <div className="score-big">
                {s.score_100?.toFixed(0)}<span className="score-unit">/100</span>
              </div>
              <div className="score-stars">{scoreStars(s.score_100)}</div>
              <div style={{color:'var(--accent)',fontWeight:600,marginBottom:8,position:'relative'}}>
                {s.rating}
              </div>
              <div className="score-fb">{s.feedback}</div>
            </div>
          </div>

          {g && (
            <div className="card" style={{animation:'fadeIn 0.4s ease 0.1s both'}}>
              <div className="card-head">
                <span className="card-icon">🔍</span>
                <span className="card-title">Keyword Gap Analysis</span>
              </div>

              <div className="coverage">
                <div className="coverage-hdr">
                  <span className="coverage-label">JD Keyword Coverage</span>
                  <span className="coverage-pct">{g.coverage_score?.toFixed(0)}%</span>
                </div>
                <div className="coverage-bar">
                  <div className="coverage-fill" style={{width:`${g.coverage_score}%`}}/>
                </div>
                <div className="coverage-hint">{g.suggestion}</div>
              </div>

              <div className="gap-grid">
                <div className="gap-card">
                  <h4>✅ Keywords Found</h4>
                  <div className="pills">
                    {g.keywords_found?.length > 0
                      ? g.keywords_found.map((k,i) => (
                          <span key={i} className="pill pill-ok"
                            style={{animationDelay:`${i*0.04}s`}}>{k}</span>
                        ))
                      : <span style={{color:'var(--text-muted)',fontSize:'0.8rem'}}>
                          No matching keywords detected
                        </span>
                    }
                  </div>
                </div>
                <div className="gap-card">
                  <h4>❌ Missing Keywords</h4>
                  <div className="pills">
                    {g.keywords_missing?.length > 0
                      ? g.keywords_missing.map((k,i) => (
                          <span key={i} className="pill pill-miss"
                            style={{animationDelay:`${i*0.04}s`}}>{k}</span>
                        ))
                      : <span style={{color:'var(--green)',fontSize:'0.8rem'}}>
                          🎉 Great keyword coverage!
                        </span>
                    }
                  </div>
                </div>
              </div>

              {g.keywords_missing?.length > 0 && (
                <div className="info-box">
                  💡 Add the missing keywords above to your CV to significantly
                  boost your match score before submitting.
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── TAB 3: RANK CVs ────────────────────────────────────
function RankTab() {
  const [files, setFiles] = useState([]);
  const [jd, setJd] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const reset = () => { setFiles([]); setJd(''); setResult(null); setError(''); };
  const canRun = files.length >= 2 && jd.trim().length > 20 && !loading;

  const addFiles = newFiles => {
    const merged = [...files];
    newFiles.forEach(f => { if(!merged.find(x=>x.name===f.name)) merged.push(f); });
    setFiles(merged);
  };

  const run = async () => {
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await api.rankCVs(files, jd);
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch(e) {
      setError(e.message||'Failed. Is the backend running?');
    } finally { setLoading(false); }
  };

  return (
    <div>
      <div className="card" style={{animation:'fadeUp 0.5s ease both'}}>
        <div className="card-head">
          <span className="card-icon">🏆</span>
          <span className="card-title">CV Ranker — For Recruiters</span>
        </div>
        <p className="card-sub">
          Upload multiple CVs and paste a Job Description. Our fine-tuned SBERT model
          will rank all candidates by semantic relevance — helping you find the best fit
          in seconds instead of hours.
        </p>

        <UploadZone multi files={files} onFiles={addFiles} onFile={()=>{}}/>

        {files.length > 0 && (
          <div className="info-box" style={{marginTop:'12px'}}>
            📎 {files.length} CV{files.length>1?'s':''} ready.
            {files.length < 2 && <span style={{color:'var(--yellow)'}}> Need at least 2 CVs.</span>}
          </div>
        )}

        <div className="divider">paste the job description</div>

        <div className="form-group">
          <label className="lbl">Job Description</label>
          <textarea
            value={jd}
            onChange={e=>setJd(e.target.value)}
            placeholder="Paste the job description to match against...&#10;E.g. We are hiring a Senior Engineer with Python, AWS, Docker, Kubernetes, microservices experience..."
          />
        </div>

        <button className="btn btn-primary" onClick={run} disabled={!canRun}>
          {loading?`⏳ Ranking ${files.length} CVs...`:`🏆 Rank ${files.length} CV${files.length!==1?'s':''}`}
        </button>
      </div>

      {loading && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <Spinner text={`Ranking ${files.length} CVs with fine-tuned SBERT...`}/>
        </div>
      )}

      {error && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <div className="alert alert-err">⚠️ {error}</div>
        </div>
      )}

      {result && !loading && (
        <div className="card" style={{animation:'fadeIn 0.3s ease'}}>
          <div className="res-hdr">
            <span className="res-title">🏆 Ranked Candidates</span>
            <button className="btn btn-ghost" onClick={reset}>↩ Reset</button>
          </div>

          <div className="info-box">
            🎯 Ranked <b style={{color:'var(--text)'}}>{result.total_cvs} candidates</b> · 
            Top pick: <b style={{color:'var(--accent)'}}>{result.top_candidate}</b>
          </div>

          <div className="rank-wrap">
            <table className="rank-tbl">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Candidate</th>
                  <th>Match Score</th>
                  <th>Rating</th>
                </tr>
              </thead>
              <tbody>
                {result.rankings?.map((r,i) => (
                  <tr
                    key={i}
                    className={i===0?'rank-first':''}
                    style={{animationDelay:`${i*0.06}s`}}
                  >
                    <td>
                      <span className="rank-num">
                        {i===0&&<span style={{marginRight:'3px'}}>👑</span>}
                        #{r.rank}
                      </span>
                    </td>
                    <td><span className="rank-fname">{r.name}</span></td>
                    <td>
                      <span className={`score-badge ${scoreBadgeClass(r.score)}`}>
                        {r.score?.toFixed(1)}%
                      </span>
                    </td>
                    <td style={{color:'var(--text-dim)',fontSize:'0.8rem'}}>{r.rating}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {result.failed_cvs?.length > 0 && (
            <div className="alert alert-err" style={{marginTop:'12px'}}>
              ⚠️ Could not parse: {result.failed_cvs.join(', ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── ROOT APP ───────────────────────────────────────────
function App() {
  const [tab, setTab] = useState('predict');
  const [online, setOnline] = useState(null);

  useEffect(() => {
    api.ping().then(setOnline);
    const t = setInterval(() => api.ping().then(setOnline), 10000);
    return () => clearInterval(t);
  }, []);

  const tabs = [
    { id:'predict', label:'🎯 Predict Category' },
    { id:'score',   label:'📊 Score CV' },
    { id:'rank',    label:'🏆 Rank CVs' },
  ];

  return (
    <>
      {/* Header */}
      <header className="hdr">
        <div className="hdr-inner">
          <div className="logo">
            Resume<span className="logo-iq">IQ</span>
            <div className="logo-dot"/>
          </div>

          <nav className="nav-pills">
            {tabs.map(t => (
              <button
                key={t.id}
                className={`nav-pill${tab===t.id?' active':''}`}
                onClick={()=>setTab(t.id)}
              >{t.label}</button>
            ))}
          </nav>

          <div className="api-status">
            <div className={`status-dot${online===false?' offline':''}`}/>
            {online===null?'Connecting…':online?'API Live':'API Offline'}
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <div className="hero-badge" style={{animation:'fadeUp 0.5s ease both'}}>
          ✦ Fine-tuned SBERT · SVM Classifier · Group-B9
        </div>
        <h1 style={{animation:'fadeUp 0.5s ease 0.1s both'}}>
          AI Resume Intelligence<br/>
          <span className="hero-hl">That Actually Works</span>
        </h1>
        <p style={{animation:'fadeUp 0.5s ease 0.2s both'}}>
          Upload your CV and let our domain fine-tuned transformer model
          analyze, score, and rank it with <b style={{color:'var(--accent)'}}>83.3% accuracy</b> across
          20 real job categories.
        </p>

        <div className="stats" style={{animation:'fadeUp 0.5s ease 0.3s both'}}>
          {[['83.3%','Ranking Accuracy'],['2,095','Resumes'],['137','Real JDs'],['20','Categories']].map(([n,l],i) => (
            <div className="stat" key={i}>
              <div className="stat-n">{n}</div>
              <div className="stat-l">{l}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Offline warning */}
      {online === false && (
        <div style={{maxWidth:'1100px',margin:'0 auto',padding:'0 1.5rem 16px',animation:'fadeIn 0.3s ease'}}>
          <div className="alert alert-err">
            ⚠️ Backend offline. Run <code style={{background:'rgba(255,255,255,0.1)',padding:'2px 7px',borderRadius:'4px',margin:'0 4px'}}>python backend/main.py</code> to start it.
          </div>
        </div>
      )}

      {/* Main panels */}
      <main className="main">
        <div className={`panel${tab==='predict'?' active':''}`}><PredictTab/></div>
        <div className={`panel${tab==='score'?' active':''}`}><ScoreTab/></div>
        <div className={`panel${tab==='rank'?' active':''}`}><RankTab/></div>
      </main>

      {/* Footer */}
      <footer>
        Built by <b>Group-B9</b> · Powered by <b>Fine-tuned SBERT</b> + <b>SVM Classifier</b> ·
        Resume Shortlisting &amp; Ranking with Transformers
      </footer>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
