# ============================================================
# model_loader.py
# Loads all ML models once at startup
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import scipy.sparse as sp

# ============================================================
# PATHS — adjust if your folder structure is different
# ============================================================

BASE_DIR       = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR       = os.path.join(BASE_DIR, "ml_pipeline", "data")
EMB_DIR        = os.path.join(BASE_DIR, "ml_pipeline", "embeddings")
MODEL_DIR      = os.path.join(BASE_DIR, "ml_pipeline", "models")

print("="*55)
print("LOADING ALL MODELS...")
print("="*55)

# ============================================================
# 1. LOAD FINE-TUNED SBERT MODEL
# ============================================================

print("⏳ Loading Fine-tuned SBERT...")
SBERT_PATH   = os.path.join(
    MODEL_DIR, "finetuned_sbert"
)
sbert_model  = SentenceTransformer(SBERT_PATH)
print("✅ Fine-tuned SBERT loaded")

# ============================================================
# 2. LOAD RESUME DATASET
# ============================================================

print("⏳ Loading resume dataset...")
resume_df = pd.read_csv(
    os.path.join(DATA_DIR, "cleaned_resumes.csv")
)
jd_df     = pd.read_csv(
    os.path.join(DATA_DIR, "cleaned_jds.csv")
)
print(f"✅ Resumes loaded     : {len(resume_df)}")
print(f"✅ JDs loaded         : {len(jd_df)}")

# ============================================================
# 3. LOAD PRECOMPUTED EMBEDDINGS
# ============================================================

print("⏳ Loading precomputed embeddings...")
sbert_resume_emb = np.load(
    os.path.join(EMB_DIR, "finetuned_resume_embeddings.npy")
)
sbert_jd_emb     = np.load(
    os.path.join(EMB_DIR, "finetuned_jd_embeddings.npy")
)
print(f"✅ Resume embeddings  : {sbert_resume_emb.shape}")
print(f"✅ JD embeddings      : {sbert_jd_emb.shape}")

# ============================================================
# 4. LOAD TF-IDF MATRIX
# ============================================================

print("⏳ Loading TF-IDF matrix...")
tfidf_resume_matrix = sp.load_npz(
    os.path.join(EMB_DIR, "tfidf_resume_matrix.npz")
)
tfidf_jd_matrix     = sp.load_npz(
    os.path.join(EMB_DIR, "tfidf_jd_matrix.npz")
)
print(f"✅ TF-IDF matrices loaded")

# ============================================================
# 5. RETRAIN SVM + LABEL ENCODER
# We retrain on load since we didn't save the model
# Takes ~30 seconds at startup
# ============================================================

print("⏳ Training SVM classifier...")

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

le          = LabelEncoder()
y           = le.fit_transform(resume_df['Category'].values)
X           = sbert_resume_emb

svm_model   = SVC(
    kernel      = 'rbf',
    C           = 1.0,
    random_state= 42,
    probability = True
)
svm_model.fit(X, y)

print(f"✅ SVM trained on {len(X)} resumes")
print(f"✅ Categories : {list(le.classes_)}")

# ============================================================
# 6. LOAD TF-IDF VECTORIZER
# Refit on existing data since we didn't save it
# ============================================================

print("⏳ Fitting TF-IDF vectorizer...")

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

stop_words = set(stopwords.words('english'))
stemmer    = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text   = text.lower()
    text   = re.sub(r'http\S+|www\S+', '', text)
    text   = re.sub(r'\S+@\S+', '', text)
    text   = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer

all_text = (
    resume_df['Cleaned_Resume'].tolist() +
    jd_df['Cleaned_JD'].tolist()
)

tfidf_vectorizer = TfidfVectorizer(
    max_features = 10000,
    ngram_range  = (1, 2),
    min_df       = 2,
    max_df       = 0.95
)
tfidf_vectorizer.fit(all_text)
print(f"✅ TF-IDF vectorizer fitted")


# ============================================================
# 7. CATEGORY TO JD MAPPING
# Maps category names to their JD texts
# Used for gap analysis
# ============================================================

cat_to_jds = {}
for cat in jd_df['Category'].unique():
    cat_to_jds[cat] = jd_df[
        jd_df['Category'] == cat
    ]['Job_Description'].tolist()

print(f"✅ Category JD map built : {len(cat_to_jds)} categories")


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "="*55)
print("ALL MODELS LOADED SUCCESSFULLY")
print("="*55)
print(f"✅ Fine-tuned SBERT    : ready")
print(f"✅ SVM classifier      : ready")
print(f"✅ TF-IDF vectorizer   : ready")
print(f"✅ Resume embeddings   : {sbert_resume_emb.shape}")
print(f"✅ JD embeddings       : {sbert_jd_emb.shape}")
print(f"✅ Categories          : {len(le.classes_)}")
print("="*55)