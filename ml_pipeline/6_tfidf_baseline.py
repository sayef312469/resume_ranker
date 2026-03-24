# ============================================================
# STEP 6: TF-IDF + COSINE SIMILARITY BASELINE
# Classical ML baseline to compare against BERT and SBERT
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

# ============================================================
# 6A. LOAD CLEANED DATA
# ============================================================

resume_df = pd.read_csv("data/cleaned_resumes.csv")
jd_df     = pd.read_csv("data/cleaned_jds.csv")

print("="*55)
print("STEP 6: TF-IDF BASELINE")
print("="*55)
print(f"✅ Resumes loaded : {len(resume_df)}")
print(f"✅ JDs loaded     : {len(jd_df)}")


# ============================================================
# 6B. BUILD TF-IDF MATRIX
# Fit vectorizer on ALL text (resumes + JDs combined)
# This gives a shared vocabulary space
# ============================================================

print("\n⏳ Building TF-IDF matrix...")

# Combine resumes and JDs for fitting
all_text = (
    resume_df['Cleaned_Resume'].tolist() +
    jd_df['Cleaned_JD'].tolist()
)

# TF-IDF Vectorizer
# max_features=10000 keeps top 10k terms — enough for resumes
tfidf_vectorizer = TfidfVectorizer(
    max_features = 10000,
    ngram_range  = (1, 2),   # unigrams and bigrams
    min_df       = 2,        # ignore terms appearing < 2 times
    max_df       = 0.95      # ignore terms in > 95% of docs
)

tfidf_vectorizer.fit(all_text)

# Transform resumes and JDs separately
resume_tfidf = tfidf_vectorizer.transform(
    resume_df['Cleaned_Resume'].tolist()
)
jd_tfidf = tfidf_vectorizer.transform(
    jd_df['Cleaned_JD'].tolist()
)

print(f"✅ TF-IDF matrix built")
print(f"   Resume matrix shape : {resume_tfidf.shape}")
print(f"   JD matrix shape     : {jd_tfidf.shape}")
print(f"   Vocabulary size     : {len(tfidf_vectorizer.vocabulary_)}")


# ============================================================
# 6C. RANKING FUNCTION — same logic as SBERT/BERT
# ============================================================

def rank_resumes_tfidf(jd_index, resume_tfidf, jd_tfidf,
                       resume_df, top_n=10):
    """
    Rank resumes against a JD using TF-IDF cosine similarity
    """
    jd_vector = jd_tfidf[jd_index]
    scores    = cosine_similarity(jd_vector, resume_tfidf)[0]

    results               = resume_df.copy()
    results['TFIDF_Score'] = scores

    results = results.sort_values('TFIDF_Score', ascending=False)
    return results.head(top_n)[
        ['Resume_ID', 'Category', 'Resume', 'TFIDF_Score']
    ].reset_index(drop=True)


# ============================================================
# 6D. RUN RANKING FOR ALL 137 JDs
# ============================================================

print("\n⏳ Ranking resumes with TF-IDF for all 137 JDs...")

all_tfidf_results  = {}
tfidf_accuracies   = []

for i, row in jd_df.iterrows():
    jd_category = row['Category']
    jd_title    = row['Job_Title']
    jd_number   = row['JD_Number']

    tfidf_top = rank_resumes_tfidf(
        jd_index     = i,
        resume_tfidf = resume_tfidf,
        jd_tfidf     = jd_tfidf,
        resume_df    = resume_df,
        top_n        = 10
    )

    key = f"{jd_category}_JD{jd_number}"
    all_tfidf_results[key] = tfidf_top

    correct  = (tfidf_top['Category'] == jd_category).sum()
    accuracy = correct / 10 * 100

    tfidf_accuracies.append({
        'Category'  : jd_category,
        'Job_Title' : jd_title,
        'JD_Number' : jd_number,
        'Correct'   : correct,
        'Accuracy'  : accuracy
    })

print("✅ TF-IDF ranking complete")


# ============================================================
# 6E. ACCURACY PER CATEGORY
# ============================================================

tfidf_acc_df  = pd.DataFrame(tfidf_accuracies)
tfidf_cat_acc = tfidf_acc_df.groupby('Category')['Accuracy'].mean()

# Load SBERT and BERT results for comparison
sbert_acc_df  = pd.read_csv("results/sbert_jd_accuracy.csv")
bert_acc_df   = pd.read_csv("results/bert_jd_accuracy.csv")
sbert_cat_acc = sbert_acc_df.groupby('Category')['Accuracy'].mean()
bert_cat_acc  = bert_acc_df.groupby('Category')['Accuracy'].mean()

print("\n" + "="*70)
print("ACCURACY COMPARISON — TF-IDF vs BERT vs SBERT")
print("="*70)
print(f"{'Category':<25} {'TF-IDF':>10} {'BERT':>10} "
      f"{'SBERT':>10} {'Winner':>10}")
print("-"*65)

for cat in sorted(tfidf_cat_acc.index):
    t_acc  = tfidf_cat_acc[cat]
    b_acc  = bert_cat_acc.get(cat, 0)
    s_acc  = sbert_cat_acc.get(cat, 0)

    best   = max(t_acc, b_acc, s_acc)
    if best == s_acc:
        winner = "SBERT 🔵"
    elif best == b_acc:
        winner = "BERT 🟡"
    else:
        winner = "TF-IDF 🟢"

    print(f"{cat:<25} {t_acc:>9.1f}% {b_acc:>9.1f}% "
          f"{s_acc:>9.1f}% {winner:>10}")

print("-"*65)
print(f"{'OVERALL AVERAGE':<25} "
      f"{tfidf_cat_acc.mean():>9.1f}% "
      f"{bert_cat_acc.mean():>9.1f}% "
      f"{sbert_cat_acc.mean():>9.1f}%")


# ============================================================
# 6F. OVERALL SUMMARY
# ============================================================

overall_tfidf = tfidf_cat_acc.mean()
overall_bert  = bert_cat_acc.mean()
overall_sbert = sbert_cat_acc.mean()

print("\n" + "="*55)
print("OVERALL RESULTS")
print("="*55)
print(f"✅ TF-IDF avg accuracy : {overall_tfidf:.1f}%")
print(f"✅ BERT  avg accuracy  : {overall_bert:.1f}%")
print(f"✅ SBERT avg accuracy  : {overall_sbert:.1f}%")
print(f"\n📊 Improvement over TF-IDF:")
print(f"   BERT  : +{overall_bert  - overall_tfidf:.1f}%")
print(f"   SBERT : +{overall_sbert - overall_tfidf:.1f}%")
print(f"\n✅ Best model : SBERT" if overall_sbert > overall_bert
      else f"\n✅ Best model : BERT")


# ============================================================
# 6G. SAVE RESULTS
# ============================================================

# Save all TF-IDF rankings
all_tfidf_df = pd.concat([
    df.assign(JD_Key=key)
    for key, df in all_tfidf_results.items()
])
all_tfidf_df.to_csv("results/tfidf_rankings.csv",    index=False)
tfidf_acc_df.to_csv("results/tfidf_jd_accuracy.csv", index=False)

print("\n✅ Results saved:")
print("   → results/tfidf_rankings.csv")
print("   → results/tfidf_jd_accuracy.csv")


# ============================================================
# 6H. SAVE TF-IDF EMBEDDINGS FOR STEP 8
# ============================================================

import scipy.sparse as sp
sp.save_npz("embeddings/tfidf_resume_matrix.npz", resume_tfidf)
sp.save_npz("embeddings/tfidf_jd_matrix.npz",     jd_tfidf)

print("   → embeddings/tfidf_resume_matrix.npz")
print("   → embeddings/tfidf_jd_matrix.npz")
print("\n🎉 Step 6 Complete — Ready for Step 7: Classifier!")