# ============================================================
# STEP 3: EMBEDDING WITH BERT AND SBERT
# ============================================================

import pandas as pd
import numpy as np
import torch
import warnings
import os
import time
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

# ============================================================
# 3A. LOAD CLEANED DATA
# ============================================================

resume_df = pd.read_csv("data/cleaned_resumes.csv")
jd_df     = pd.read_csv("data/cleaned_jds.csv")

print("✅ Cleaned data loaded")
print(f"   Resumes : {resume_df.shape}")
print(f"   JDs     : {jd_df.shape}")

# ============================================================
# 3B. DEVICE CHECK
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✅ Running on : {device.upper()}")


# ============================================================
# 3C. SBERT EMBEDDING
# ============================================================

print("\n" + "="*55)
print("LOADING SBERT MODEL...")
print("="*55)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SBERT model loaded : all-MiniLM-L6-v2")

# --- Embed Resumes ---
print(f"\n⏳ Generating SBERT embeddings for {len(resume_df)} resumes...")
start = time.time()

sbert_resume_embeddings = sbert_model.encode(
    resume_df['Cleaned_Resume'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=device
)

elapsed = time.time() - start
print(f"✅ SBERT resume embeddings done in {elapsed:.1f}s")
print(f"   Shape : {sbert_resume_embeddings.shape}")

# --- Embed JDs ---
print(f"\n⏳ Generating SBERT embeddings for {len(jd_df)} JDs...")
start = time.time()

sbert_jd_embeddings = sbert_model.encode(
    jd_df['Cleaned_JD'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=device
)

elapsed = time.time() - start
print(f"✅ SBERT JD embeddings done in {elapsed:.1f}s")
print(f"   Shape : {sbert_jd_embeddings.shape}")
# Expected: (137, 384)


# ============================================================
# 3D. BERT EMBEDDING
# ============================================================

print("\n" + "="*55)
print("LOADING BERT MODEL...")
print("="*55)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model     = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
bert_model.to(device)
print("✅ BERT model loaded : bert-base-uncased")

def get_bert_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(768)
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# --- Embed Resumes ---
print(f"\n⏳ Generating BERT embeddings for {len(resume_df)} resumes...")
print("   (may take a few mins on GPU...)")

start = time.time()
bert_resume_embeddings = []

for i, text in enumerate(resume_df['Cleaned_Resume'].tolist()):
    emb = get_bert_embedding(text)
    bert_resume_embeddings.append(emb)
    if (i + 1) % 200 == 0:
        print(f"   Processed {i+1}/{len(resume_df)} resumes...")

bert_resume_embeddings = np.array(bert_resume_embeddings)
elapsed = time.time() - start
print(f"✅ BERT resume embeddings done in {elapsed:.1f}s")
print(f"   Shape : {bert_resume_embeddings.shape}")
# Expected: (2095, 768)

# --- Embed JDs ---
print(f"\n⏳ Generating BERT embeddings for {len(jd_df)} JDs...")
start = time.time()

bert_jd_embeddings = np.array([
    get_bert_embedding(text)
    for text in jd_df['Cleaned_JD'].tolist()
])

elapsed = time.time() - start
print(f"✅ BERT JD embeddings done in {elapsed:.1f}s")
print(f"   Shape : {bert_jd_embeddings.shape}")
# Expected: (137, 768)


# ============================================================
# 3E. SAVE ALL EMBEDDINGS
# ============================================================

os.makedirs("embeddings", exist_ok=True)

np.save("embeddings/sbert_resume_embeddings.npy", sbert_resume_embeddings)
np.save("embeddings/sbert_jd_embeddings.npy",     sbert_jd_embeddings)
np.save("embeddings/bert_resume_embeddings.npy",  bert_resume_embeddings)
np.save("embeddings/bert_jd_embeddings.npy",      bert_jd_embeddings)

print("\n✅ All embeddings saved:")
print("   → embeddings/sbert_resume_embeddings.npy")
print("   → embeddings/sbert_jd_embeddings.npy")
print("   → embeddings/bert_resume_embeddings.npy")
print("   → embeddings/bert_jd_embeddings.npy")


# ============================================================
# 3F. VERIFICATION
# ============================================================

print("\n" + "="*55)
print("VERIFICATION — Reloading saved embeddings")
print("="*55)

s_res = np.load("embeddings/sbert_resume_embeddings.npy")
s_jd  = np.load("embeddings/sbert_jd_embeddings.npy")
b_res = np.load("embeddings/bert_resume_embeddings.npy")
b_jd  = np.load("embeddings/bert_jd_embeddings.npy")

print(f"SBERT resume embeddings : {s_res.shape}")
print(f"SBERT JD embeddings     : {s_jd.shape}")
print(f"BERT  resume embeddings : {b_res.shape}")
print(f"BERT  JD embeddings     : {b_jd.shape}")


# ============================================================
# 3G. FINAL SUMMARY
# ============================================================

print("\n" + "="*55)
print("EMBEDDING SUMMARY")
print("="*55)
print(f"✅ SBERT dimensions       : 384")
print(f"✅ BERT  dimensions       : 768")
print(f"✅ Total resumes embedded : {s_res.shape[0]}")
print(f"✅ Total JDs embedded     : {s_jd.shape[0]}")
print(f"✅ Device used            : {device.upper()}")
print("="*55)
print("\n🎉 Step 3 Complete — Ready for Step 4: Ranking!")