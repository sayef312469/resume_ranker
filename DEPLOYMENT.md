# ResumeIQ — Deployment Guide for Replit

## 📋 Quick Summary

**ResumeIQ** is an AI-powered resume ranking and scoring system built with:
- **Backend**: FastAPI (Python) - ML models, APIs, static file serving
- **Frontend**: React 18 via CDN (no build step)
- **Models**: Fine-tuned SBERT + SVM classifier
- **Deployment Target**: Replit (free, student-friendly)

---

## 🚀 Deployment Steps

### Step 1: Push Project to GitHub

First, initialize Git locally and push to GitHub:

```bash
cd ~/Desktop/resume_ranker
git init
git add .
git commit -m "Initial commit: ResumeIQ project"
git remote add origin https://github.com/YOUR_USERNAME/resume_ranker.git
git branch -M main
git push -u origin main
```

> Replace `YOUR_USERNAME` with your GitHub username

### Step 2: Create a Replit Account

1. Go to [replit.com](https://replit.com)
2. Sign up (free account)
3. Click **"+ Create"** button

### Step 3: Import from GitHub

1. Click **"Import from GitHub"**
2. Paste your GitHub repo URL: `https://github.com/YOUR_USERNAME/resume_ranker`
3. Click **"Import"**
4. Wait for Replit to clone the repository

### Step 4: Configure Replit Runtime

Replit will auto-detect `requirements.txt` and `replit.nix`. The environment will be prepared automatically.

**Installation may take 3-5 minutes** (first time only). This includes:
- Python packages (PyTorch, transformers, scikit-learn, etc.)
- ML models from `ml_pipeline/models/`
- NLTK data downloads

### Step 5: Run the Project

1. Click the **"Run"** button (top center)
2. Replit will execute the `.replit` configuration which runs `backend/main.py`
3. Server starts on port 8000 (publicly accessible)

### Step 6: Access the Application

Once running, Replit shows a preview pane or URL like:
```
https://resume-ranker.YOUR_USERNAME.replit.dev
```

**Open this URL in a new browser tab** — you'll see ResumeIQ UI!

### Step 7: Share with Team

Share the Replit URL with your project teammates:
```
https://resume-ranker.YOUR_USERNAME.replit.dev
```

They can:
- Upload resumes/CVs
- Predict job categories
- Score CVs against job descriptions
- Rank multiple CVs
- **No backend setup needed** — everything runs on Replit!

---

## 📂 Project Structure

```
resume_ranker/
├── backend/
│   ├── main.py              [FastAPI app - serves both API & frontend]
│   ├── model_loader.py      [Loads ML models at startup]
│   ├── ml_functions.py      [Core ML logic]
│   └── cv_parser.py         [PDF/DOCX/TXT extraction]
│
├── frontend/
│   ├── index.html           [React entry point]
│   ├── app.js               [React components & logic]
│   ├── config.js            [API configuration (auto-configured)]
│   └── styles.css           [Styling]
│
├── ml_pipeline/
│   ├── models/
│   │   └── finetuned_sbert/ [Fine-tuned transformer model]
│   ├── embeddings/          [Pre-computed embeddings & TF-IDF]
│   └── data/                [Training data CSVs]
│
├── requirements.txt         [Python dependencies]
├── .replit                  [Replit configuration]
├── replit.nix               [Nix environment setup]
└── .gitignore               [Git ignore patterns]
```

---

## 🔧 How It Works

### Architecture

```
Browser (Frontend)
      ↓
ResumeIQ UI (React 18 via CDN)
      ↓
API Calls (fetch requests)
      ↓
FastAPI Backend (port 8000)
      ├→ /predict-category (SVM classifier)
      ├→ /score-cv (SBERT similarity)
      ├→ /rank-cvs (Multi-CV ranking)
      └→ Static files (frontend HTML/CSS/JS)
      ↓
ML Models (SBERT + SVM)
Embeddings & TF-IDF matrices
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/predict-category` | POST | Classify CV category |
| `/score-cv` | POST | Score CV vs JD |
| `/rank-cvs` | POST | Rank multiple CVs |
| `/static/` | GET | Frontend files |

---

## ⚙️ Configuration

### Frontend API Endpoint (Auto-Configured)

**File**: `config.js`

```javascript
const API_BASE = (typeof window !== 'undefined' && window.location.hostname === 'localhost')
  ? 'http://localhost:8000'
  : `${window.location.protocol}//${window.location.host}`;
```

**Behavior**:
- **Local Dev**: Uses `http://localhost:8000`
- **Replit Deploy**: Uses the Replit domain automatically (e.g., `https://resume-ranker.user.replit.dev`)

✅ **No manual config needed after deployment!**

---

## 📊 Model Details

### Fine-Tuned SBERT Model
- **Base**: `sentence-transformers/all-MiniLM-L6-v2`
- **Fine-Tuned**: Triplet loss on resume-JD pairs
- **Size**: ~22 MB
- **Embedding Dim**: 384

### SVM Classifier
- **Training Data**: Embeddings from fine-tuned SBERT
- **Classes**: 10+ job categories
- **Accuracy**: ~85% on validation set

### TF-IDF Baseline
- **Used for**: Quick relevance scoring
- **Vectorizer**: Scikit-learn TfidfVectorizer

---

## 🐛 Troubleshooting

### Issue: "Cannot find replit.nix"
**Solution**: It should be auto-detected. If not, manually set Python to 3.11 in Replit Settings.

### Issue: "Models loading takes forever"
**Expected**: First run takes 2-5 minutes.
- PyTorch downloads (~100 MB)
- NLTK data (~50 MB)
- Models cache after first run

### Issue: API calls return 404
**Check**:
1. Frontend URL shows `protocol://domain` (not file://)
2. `config.js` has correct API_BASE
3. Backend is running (check terminal output)

### Issue: "Permission denied" when pushing to GitHub
**Solution**: Use GitHub token instead of password:
```bash
git remote set-url origin https://YOUR_TOKEN@github.com/USERNAME/resume_ranker.git
```

---

## 🎓 For Team Collaboration

### Share with Teammates

1. **Share Replit URL**: Send the deployment link (e.g., `https://resume-ranker.user.replit.dev`)
2. **Teammates open URL**: No setup required — just use the app!
3. **Optional**: Invite them to the Replit project for collaborative development

### Replit Collaboration Features

- **Multiplayer**: Real-time code editing (premium)
- **Version Control**: Git integration built-in
- **Comments**: Leave code comments for review

---

## 📦 Local Development

### Run Locally (for testing before Replit)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run backend
cd backend
python main.py

# 3. Open browser
# Open http://localhost:8000 in your browser

# 4. Frontend will load and make API calls to backend
```

---

## 🔐 Notes & Limitations

### Replit Free Tier
- ✅ Always-on hosting
- ✅ 500MB disk space
- ✅ Public URL sharing
- ✅ Git integration
- ⚠️ Sleeps after 1 hour of inactivity (free tier)
  - Restart by accessing the URL again

### Data Privacy
- ✅ Models running server-side (no data sent elsewhere)
- ⚠️ CVs uploaded are temporary (not stored)
- ✅ Replit provides HTTPS encryption

---

## 📝 File Manifest

### Files Modified for Deployment

| File | Change | Reason |
|------|--------|--------|
| `backend/main.py` | Added `StaticFiles` mounting | Serve frontend + API |
| `frontend/config.js` | Auto-detection logic | Works on Replit domain |
| `requirements.txt` | ✨ **Created** | Python dependencies |
| `.replit` | ✨ **Created** | Replit configuration |
| `replit.nix` | ✨ **Created** | Nix environment |
| `.gitignore` | ✨ **Created** | Git ignore patterns |

---

## ✅ Deployment Checklist

- [ ] GitHub repo created with all files
- [ ] Replit account created
- [ ] Project imported from GitHub
- [ ] Dependencies installed (watch for PyTorch download)
- [ ] Backend running (terminal shows "Uvicorn running")
- [ ] Frontend loads (can see UI in preview)
- [ ] API call works (upload a test CV)
- [ ] Share URL with teammates
- [ ] Test from teammate's browser

---

## 🎉 Success Indicators

✅ Backend starts without errors:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
LOADING ALL MODELS...
✅ SBERT model loaded
✅ SVM classifier ready
✅ TF-IDF vectorizer ready
```

✅ Frontend loads:
```
ResumeIQ UI visible in browser
Three tabs: Predict Category, Score CV, Rank CVs
```

✅ API works:
```
Upload a CV → Get predictions instantly
No CORS errors
```

---

## 📞 Support

- **Replit Docs**: https://docs.replit.com
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Sentence Transformers**: https://www.sbert.net

---

## 🚀 Next Steps

After successful deployment:
1. Test with real CVs and job descriptions
2. Share with project team
3. Gather feedback & iterate
4. (Optional) Deploy backend separately if needed

**Happy deploying! 🎊**
