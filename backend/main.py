# ============================================================
# main.py
# FastAPI backend — all API endpoints
# ============================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sys
import os

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# LOAD ALL MODELS AT STARTUP
# ============================================================

from model_loader import (
    sbert_model,
    svm_model,
    le,
    tfidf_vectorizer,
    resume_df,
    jd_df,
    cat_to_jds,
    clean_text
)

from ml_functions import (
    predict_cv_category,
    score_cv_against_jd,
    rank_multiple_cvs,
    analyze_cv_gaps
)

from cv_parser import extract_text_from_file

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title       = "Resume Ranking API",
    description = "ML-powered resume ranking and scoring system",
    version     = "1.0.0"
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)


# ============================================================
# ENDPOINT 1 — HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():
    return {
        "status"    : "running",
        "message"   : "Resume Ranking API is live",
        "models"    : {
            "sbert"     : "fine-tuned ✅",
            "svm"       : "ready ✅",
            "tfidf"     : "ready ✅"
        },
        "categories": list(le.classes_)
    }


# ============================================================
# ENDPOINT 2 — PREDICT CV CATEGORY
# Job seeker uploads CV → get best matching job categories
# ============================================================

@app.post("/predict-category")
async def predict_category(
    file: UploadFile = File(...)
):
    try:
        # Extract text from uploaded CV
        cv_text = await extract_text_from_file(file)

        if not cv_text or len(cv_text.strip()) < 50:
            return JSONResponse(
                status_code = 400,
                content     = {
                    "error": "CV text too short or empty"
                }
            )

        # Predict category
        predictions = predict_cv_category(
            cv_text       = cv_text,
            sbert_model   = sbert_model,
            svm_model     = svm_model,
            label_encoder = le,
            top_n         = 5
        )

        return {
            "status"     : "success",
            "cv_length"  : len(cv_text),
            "predictions": predictions,
            "best_match" : predictions[0]['category'],
            "confidence" : predictions[0]['confidence']
        }

    except Exception as e:
        return JSONResponse(
            status_code = 500,
            content     = {"error": str(e)}
        )


# ============================================================
# ENDPOINT 3 — SCORE CV AGAINST JD
# Job seeker uploads CV + enters JD → get match score
# ============================================================

class ScoreRequest(BaseModel):
    jd_text : str
    cv_text : Optional[str] = None


@app.post("/score-cv")
async def score_cv(
    file   : UploadFile = File(...),
    jd_text: str        = Form(...)
):
    try:
        # Extract CV text
        cv_text = await extract_text_from_file(file)

        if not cv_text or len(cv_text.strip()) < 50:
            return JSONResponse(
                status_code = 400,
                content     = {"error": "CV text too short"}
            )

        if not jd_text or len(jd_text.strip()) < 20:
            return JSONResponse(
                status_code = 400,
                content     = {"error": "JD text too short"}
            )

        # Score CV against JD
        score_result = score_cv_against_jd(
            cv_text     = cv_text,
            jd_text     = jd_text,
            sbert_model = sbert_model,
            clean_fn    = clean_text
        )

        # Gap analysis
        gap_result = analyze_cv_gaps(
            cv_text          = clean_text(cv_text),
            jd_text          = clean_text(jd_text),
            tfidf_vectorizer = tfidf_vectorizer
        )

        return {
            "status"      : "success",
            "score"       : score_result,
            "gap_analysis": gap_result
        }

    except Exception as e:
        return JSONResponse(
            status_code = 500,
            content     = {"error": str(e)}
        )


# ============================================================
# ENDPOINT 4 — RANK MULTIPLE CVs
# Recruiter uploads multiple CVs + JD → get ranked list
# ============================================================

@app.post("/rank-cvs")
async def rank_cvs(
    files  : List[UploadFile] = File(...),
    jd_text: str              = Form(...)
):
    try:
        if len(files) < 2:
            return JSONResponse(
                status_code = 400,
                content     = {
                    "error": "Please upload at least 2 CVs"
                }
            )

        if len(files) > 50:
            return JSONResponse(
                status_code = 400,
                content     = {
                    "error": "Maximum 50 CVs allowed"
                }
            )

        # Extract text from all CVs
        cv_texts = []
        cv_names = []
        failed   = []

        for file in files:
            try:
                text = await extract_text_from_file(file)
                if text and len(text.strip()) > 50:
                    cv_texts.append(text)
                    cv_names.append(file.filename)
                else:
                    failed.append(file.filename)
            except Exception:
                failed.append(file.filename)

        if len(cv_texts) < 2:
            return JSONResponse(
                status_code = 400,
                content     = {
                    "error": "Not enough valid CVs extracted"
                }
            )

        # Rank CVs
        rankings = rank_multiple_cvs(
            cv_texts    = cv_texts,
            cv_names    = cv_names,
            jd_text     = jd_text,
            sbert_model = sbert_model,
            clean_fn    = clean_text
        )

        return {
            "status"        : "success",
            "total_cvs"     : len(cv_texts),
            "failed_cvs"    : failed,
            "jd_preview"    : jd_text[:100] + "...",
            "rankings"      : rankings,
            "top_candidate" : rankings[0]['name']
        }

    except Exception as e:
        return JSONResponse(
            status_code = 500,
            content     = {"error": str(e)}
        )


# ============================================================
# ENDPOINT 5 — GET ALL CATEGORIES
# Returns all available job categories
# ============================================================

@app.get("/categories")
def get_categories():
    return {
        "status"    : "success",
        "categories": list(le.classes_),
        "total"     : len(le.classes_)
    }


# ============================================================
# ENDPOINT 6 — GET JDs FOR A CATEGORY
# Returns available JDs for a specific category
# ============================================================

@app.get("/jds/{category}")
def get_jds_for_category(category: str):
    category = category.upper()
    if category not in cat_to_jds:
        return JSONResponse(
            status_code = 404,
            content     = {
                "error"     : f"Category {category} not found",
                "available" : list(cat_to_jds.keys())
            }
        )
    return {
        "status"  : "success",
        "category": category,
        "jds"     : cat_to_jds[category],
        "count"   : len(cat_to_jds[category])
    }


# ============================================================
# CATCH-ALL: Serve frontend for any unmatched route (SPA)
# ============================================================

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve index.html for client-side routing, or static files"""
    frontend_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend"
    )
    
    # Check if it's a static file that exists
    file_path = os.path.join(frontend_dir, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Otherwise, serve index.html for React Router
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            return HTMLResponse(f.read())
    
    return JSONResponse(status_code=404, content={"error": "Not found"})


# ============================================================
# SERVE FRONTEND STATIC FILES (MUST BE LAST)
# ============================================================

# Mount static files AFTER all API routes so routes take priority
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir), name="frontend")


# ============================================================
# RUN THE API
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False
    )