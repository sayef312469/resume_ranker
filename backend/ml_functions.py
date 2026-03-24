# ============================================================
# ml_functions.py
# Core ML functions wrapped for API use
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import re

# ============================================================
# FUNCTION 1 — PREDICT CV CATEGORY
# Uses SVM classifier on SBERT embeddings
# ============================================================

def predict_cv_category(
    cv_text        : str,
    sbert_model,
    svm_model,
    label_encoder,
    top_n          : int = 5
) -> List[Dict]:
    """
    Predicts the best job categories for a CV.
    Returns top N categories with confidence scores.
    """
    from model_loader import clean_text

    # Clean and embed CV text
    cleaned   = clean_text(cv_text)
    embedding = sbert_model.encode(
        [cleaned], convert_to_numpy=True
    )

    # Get probability scores for all categories
    probs      = svm_model.predict_proba(embedding)[0]
    categories = label_encoder.classes_

    # Sort by probability descending
    sorted_idx = np.argsort(probs)[::-1][:top_n]

    results = []
    for idx in sorted_idx:
        results.append({
            "category"   : categories[idx],
            "confidence" : round(float(probs[idx]) * 100, 2),
            "match_level": get_match_level(probs[idx])
        })

    return results


def get_match_level(score: float) -> str:
    """Convert score to human readable match level"""
    if score >= 0.7:
        return "Excellent Match 🟢"
    elif score >= 0.4:
        return "Good Match 🟡"
    elif score >= 0.2:
        return "Fair Match 🟠"
    else:
        return "Low Match 🔴"


# ============================================================
# FUNCTION 2 — SCORE CV AGAINST A JD
# Uses SBERT-FT cosine similarity
# ============================================================

def score_cv_against_jd(
    cv_text     : str,
    jd_text     : str,
    sbert_model,
    clean_fn
) -> Dict:
    """
    Scores a CV against a specific JD.
    Returns similarity score and rating.
    """
    # Clean both texts
    cleaned_cv = clean_fn(cv_text)
    cleaned_jd = clean_fn(jd_text)

    # Embed both
    cv_emb = sbert_model.encode(
        [cleaned_cv], convert_to_numpy=True
    )
    jd_emb = sbert_model.encode(
        [cleaned_jd], convert_to_numpy=True
    )

    # Cosine similarity
    score = float(
        cosine_similarity(cv_emb, jd_emb)[0][0]
    )

    # Scale to 0-100
    scaled_score = round(score * 100, 2)

    return {
        "raw_score"   : round(score, 4),
        "score_100"   : scaled_score,
        "rating"      : get_score_rating(scaled_score),
        "feedback"    : get_score_feedback(scaled_score)
    }


def get_score_rating(score: float) -> str:
    if score >= 80:
        return "Excellent ⭐⭐⭐⭐⭐"
    elif score >= 65:
        return "Very Good ⭐⭐⭐⭐"
    elif score >= 50:
        return "Good ⭐⭐⭐"
    elif score >= 35:
        return "Fair ⭐⭐"
    else:
        return "Needs Improvement ⭐"


def get_score_feedback(score: float) -> str:
    if score >= 80:
        return "Your CV is an excellent match for this role!"
    elif score >= 65:
        return "Your CV is a strong match. Minor improvements possible."
    elif score >= 50:
        return "Your CV is a decent match. Consider adding more relevant skills."
    elif score >= 35:
        return "Your CV partially matches. Significant skill gaps detected."
    else:
        return "Your CV needs major improvements for this role."


# ============================================================
# FUNCTION 3 — RANK MULTIPLE CVs AGAINST A JD
# Recruiter feature
# ============================================================

def rank_multiple_cvs(
    cv_texts    : List[str],
    cv_names    : List[str],
    jd_text     : str,
    sbert_model,
    clean_fn
) -> List[Dict]:
    """
    Ranks multiple CVs against a single JD.
    Returns ranked list with scores.
    """
    cleaned_jd  = clean_fn(jd_text)
    jd_emb      = sbert_model.encode(
        [cleaned_jd], convert_to_numpy=True
    )

    results = []
    for i, (cv_text, cv_name) in enumerate(
        zip(cv_texts, cv_names)
    ):
        cleaned_cv = clean_fn(cv_text)
        cv_emb     = sbert_model.encode(
            [cleaned_cv], convert_to_numpy=True
        )
        score      = float(
            cosine_similarity(jd_emb, cv_emb)[0][0]
        )
        results.append({
            "rank"      : 0,
            "name"      : cv_name,
            "score"     : round(score * 100, 2),
            "rating"    : get_score_rating(score * 100),
            "raw_score" : round(score, 4)
        })

    # Sort by score descending and assign ranks
    results = sorted(
        results, key=lambda x: x['score'], reverse=True
    )
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return results


# ============================================================
# FUNCTION 4 — GAP ANALYSIS
# Finds missing keywords between CV and JD
# ============================================================

def analyze_cv_gaps(
    cv_text         : str,
    jd_text         : str,
    tfidf_vectorizer,
    top_n           : int = 10
) -> Dict:
    """
    Finds keywords in JD that are missing from CV.
    Gives actionable improvement suggestions.
    """
    # Get TF-IDF feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Transform both texts
    cv_vec  = tfidf_vectorizer.transform([cv_text])
    jd_vec  = tfidf_vectorizer.transform([jd_text])

    # Get top keywords from JD
    jd_scores    = jd_vec.toarray()[0]
    top_jd_idx   = np.argsort(jd_scores)[::-1][:top_n*2]
    top_jd_terms = [
        feature_names[i]
        for i in top_jd_idx
        if jd_scores[i] > 0
    ]

    # Check which JD keywords are in CV
    cv_scores    = cv_vec.toarray()[0]
    cv_terms_set = set([
        feature_names[i]
        for i in range(len(cv_scores))
        if cv_scores[i] > 0
    ])

    present  = []
    missing  = []

    for term in top_jd_terms[:top_n]:
        if term in cv_terms_set:
            present.append(term)
        else:
            missing.append(term)

    # Coverage score
    coverage = len(present) / top_n * 100

    return {
        "keywords_found"  : present,
        "keywords_missing": missing,
        "coverage_score"  : round(coverage, 1),
        "suggestion"      : get_gap_suggestion(coverage)
    }


def get_gap_suggestion(coverage: float) -> str:
    if coverage >= 80:
        return "Great keyword coverage! Your CV aligns well."
    elif coverage >= 60:
        return "Good coverage. Add a few more relevant keywords."
    elif coverage >= 40:
        return "Moderate coverage. Consider tailoring CV to this JD."
    else:
        return "Low coverage. Significantly tailor your CV to this JD."