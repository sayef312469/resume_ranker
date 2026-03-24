# ResumeIQ - AI-Powered Resume Ranking System

![ResumeIQ](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)
![React](https://img.shields.io/badge/React-18+-blue)

## 🎯 Project Overview

**ResumeIQ** is an intelligent resume ranking and evaluation system that uses AI/ML to:

✅ **Classify Resume Categories** - Predict job categories from CVs using a fine-tuned SBERT + SVM classifier
✅ **Score Resumes** - Calculate similarity scores between CVs and job descriptions
✅ **Rank Multiple CVs** - Rank candidate resumes against a job posting

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 2GB+ RAM (for ML models)
- pip

### Local Setup

```bash
# Clone and navigate
git clone https://github.com/yourusername/resume_ranker.git
cd resume_ranker

# Install dependencies
pip install -r requirements.txt

# Run backend
cd backend
python main.py

# Open browser
# Navigate to http://localhost:8000
```

Server starts on `http://localhost:8000`

---

## 🎓 Deployment

### Deploy to Replit (Recommended for Students)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed step-by-step instructions.

**Quick recap**:
1. Push code to GitHub
2. Create Replit account
3. Import from GitHub
4. Click "Run"
5. Share deployment URL with team

---

## 📋 Features

### 1. CV Category Prediction
Upload a resume → Get top 5 predicted job categories with confidence scores
- Uses fine-tuned SBERT embeddings
- SVM classifier trained on 10+ job categories
- ~85% accuracy

### 2. CV Scoring
Compare a resume against a job description → Get similarity score (0-100)
- Cosine similarity of embeddings
- Takes keywords, experience, skills into account
- Real-time scoring

### 3. Multi-CV Ranking
Upload multiple resumes + a job description → Get ranked list of candidates
- Scores all CVs
- Ranks by relevance
- Shows top candidate

---

## 🏗️ Architecture

```
┌─────────────────────┐
│   React Frontend    │  (served at /)
│  (Browser-based)    │
└──────────┬──────────┘
           │ fetch API calls
           ↓
┌─────────────────────┐
│   FastAPI Backend   │  (port 8000)
│  - /predict-category│
│  - /score-cv        │
│  - /rank-cvs        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  ML Models          │
│ - SBERT (fine-tuned)│
│ - SVM Classifier    │
│ - TF-IDF Vectorizer │
└─────────────────────┘
```

---

## 📂 Project Structure

```
resume_ranker/
├── backend/                    # FastAPI Application
│   ├── main.py                # Entry point
│   ├── model_loader.py        # ML model initialization
│   ├── ml_functions.py        # Core ML logic
│   ├── cv_parser.py           # PDF/DOCX/TXT extraction
│   └── test_parser.py         # Unit tests
│
├── frontend/                   # React Application
│   ├── index.html             # HTML entry point
│   ├── app.js                 # React components
│   ├── config.js              # API configuration
│   └── styles.css             # Styling
│
├── ml_pipeline/               # ML Training & Data
│   ├── models/                # Pre-trained models
│   ├── embeddings/            # Pre-computed embeddings
│   ├── data/                  # Training datasets
│   └── {1-9c}_*.py           # Training scripts
│
├── requirements.txt           # Python dependencies
├── .replit                    # Replit configuration
├── DEPLOYMENT.md              # Deployment guide
└── README.md                  # This file
```

---

## 🔧 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check, model status |
| `/predict-category` | POST | Classify CV category |
| `/score-cv` | POST | Score CV vs JD |
| `/rank-cvs` | POST | Rank multiple CVs |

### Request/Response Examples

**Predict Category**
```bash
curl -X POST http://localhost:8000/predict-category \
  -F "file=@resume.pdf"
```

Response:
```json
{
  "status": "success",
  "cv_length": 1500,
  "predictions": [
    {"category": "Software Engineer", "confidence": 0.87},
    {"category": "Data Scientist", "confidence": 0.12}
  ],
  "best_match": "Software Engineer",
  "confidence": 0.87
}
```

---

## 🤖 ML Models

### Fine-Tuned SBERT
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Fine-Tuning**: Triplet loss on resume-JD pairs
- **Embedding Dimension**: 384
- **Size**: ~22 MB

### SVM Classifier
- **Training Data**: SBERT embeddings
- **Classes**: 10+ job categories
- **Kernel**: RBF
- **Accuracy**: ~85%

### TF-IDF Baseline
- **Used for**: Quick relevance scoring
- **Vectorizer**: Scikit-learn TfidfVectorizer

---

## 📊 Performance

| Model | Task | Accuracy | Speed |
|-------|------|----------|-------|
| SBERT + SVM | Category Prediction | 85% | <500ms |
| SBERT Similarity | CV Scoring | - | <200ms |
| SBERT Ranking | Multi-CV Ranking | - | <1s (10 CVs) |

---

## 🛠️ Development

### Running Tests

```bash
cd backend
python test_parser.py
```

### Training Models (ML Pipeline)

The ML pipeline scripts are for reference and retraining:

```bash
# Data preprocessing
python ml_pipeline/2_data_preprocessing.py

# Creating embeddings
python ml_pipeline/3_embedding.py

# Fine-tuning SBERT
python ml_pipeline/9b_finetune_sbert.py

# Evaluation
python ml_pipeline/5_evaluation.py
```

### Environment Variables

Create a `.env` file (optional):
```
API_BASE=http://localhost:8000
```

---

## 📦 Dependencies

### Backend
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `sentence-transformers` - SBERT models
- `scikit-learn` - SVM classifier
- `pandas` - Data processing
- `PyPDF2`, `python-docx` - File parsing

### Frontend
- React 18 (CDN)
- Babel standalone (JSX)
- No npm build step required

---

## 🐛 Troubleshooting

### Models take long to load
**Expected behavior**: First run ~2-5 minutes for model downloads
- PyTorch (~100MB)
- Transformers (~50MB)
- Models cache after first run

### CORS errors
**Solution**: Check `config.js` has correct API_BASE

### Out of Memory
**Issue**: Batch ranking too many CVs
**Solution**: Limit to <50 CVs per request

---

## 🎓 For Students

This project is designed for **team collaboration**:

✅ Deploy to Replit for free (no credit card required)
✅ Share deployment link with teammates instantly
✅ No local setup needed to use the app
✅ Work on code together with Replit's multiplayer feature

See [DEPLOYMENT.md](DEPLOYMENT.md) for student-friendly deployment guide.

---

## 📝 Citation

If you use ResumeIQ in your project, please cite:

```bibtex
@project{resumeiq2024,
  title={ResumeIQ: AI-Powered Resume Ranking System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/resume_ranker}
}
```

---

## 📄 License

This project is open source. See LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Questions

- **Issues**: Create a GitHub issue
- **Discussions**: Use GitHub Discussions for Q&A
- **Email**: your.email@example.com

---

## 🎉 Acknowledgments

- **SBERT**: Sentence transformers for efficient embeddings
- **Scikit-learn**: Machine learning toolkit
- **FastAPI**: Modern web framework
- **React**: User interface library

---

**Built with ❤️ for AI-powered recruitment**

Last Updated: March 2024
Status: Production Ready ✅
