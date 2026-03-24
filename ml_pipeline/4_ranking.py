# ============================================================
# STEP 4: RANKING WITH COSINE SIMILARITY
# Updated for 2095 resumes, 137 JDs, 20 categories
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
warnings.filterwarnings("ignore")

# ============================================================
# 4A. LOAD DATA AND EMBEDDINGS
# ============================================================

resume_df = pd.read_csv("data/cleaned_resumes.csv")
jd_df     = pd.read_csv("data/cleaned_jds.csv")

sbert_resume_emb = np.load("embeddings/sbert_resume_embeddings.npy")
sbert_jd_emb     = np.load("embeddings/sbert_jd_embeddings.npy")
bert_resume_emb  = np.load("embeddings/bert_resume_embeddings.npy")
bert_jd_emb      = np.load("embeddings/bert_jd_embeddings.npy")

print("✅ Data and embeddings loaded")
print(f"   Resumes          : {len(resume_df)}")
print(f"   JDs              : {len(jd_df)}")
print(f"   SBERT resume emb : {sbert_resume_emb.shape}")
print(f"   BERT  resume emb : {bert_resume_emb.shape}")


# ============================================================
# 4B. CORE RANKING FUNCTION
# ============================================================

def rank_resumes(jd_index, embeddings_resumes, embeddings_jds,
                 resume_df, top_n=10, model_name="SBERT"):
    """
    Rank all resumes against a specific JD using cosine similarity.
    Returns top N ranked resumes with scores.
    """
    jd_embedding = embeddings_jds[jd_index].reshape(1, -1)
    scores       = cosine_similarity(jd_embedding, embeddings_resumes)[0]

    results                      = resume_df.copy()
    results[f'{model_name}_Score'] = scores

    results = results.sort_values(
        f'{model_name}_Score', ascending=False
    )

    return results.head(top_n)[
        ['Resume_ID', 'Category', 'Resume', f'{model_name}_Score']
    ].reset_index(drop=True)


# ============================================================
# 4C. RUN RANKING FOR ALL 137 JDs — BOTH MODELS
# ============================================================

os.makedirs("results", exist_ok=True)

all_sbert_results = {}
all_bert_results  = {}

# Store accuracy per JD for detailed analysis
sbert_jd_accuracies = []
bert_jd_accuracies  = []

print("\n" + "="*60)
print("RANKING RESULTS — TOP 10 PER JD")
print("="*60)

for i, row in jd_df.iterrows():
    jd_category = row['Category']
    jd_title    = row['Job_Title']
    jd_number   = row['JD_Number']

    # --- SBERT Ranking ---
    sbert_top = rank_resumes(
        jd_index           = i,
        embeddings_resumes = sbert_resume_emb,
        embeddings_jds     = sbert_jd_emb,
        resume_df          = resume_df,
        top_n              = 10,
        model_name         = "SBERT"
    )

    # --- BERT Ranking ---
    bert_top = rank_resumes(
        jd_index           = i,
        embeddings_resumes = bert_resume_emb,
        embeddings_jds     = bert_jd_emb,
        resume_df          = resume_df,
        top_n              = 10,
        model_name         = "BERT"
    )

    # Store results
    key = f"{jd_category}_JD{jd_number}"
    all_sbert_results[key] = sbert_top
    all_bert_results[key]  = bert_top

    # Accuracy for this JD
    sbert_correct = (sbert_top['Category'] == jd_category).sum()
    bert_correct  = (bert_top['Category']  == jd_category).sum()

    sbert_jd_accuracies.append({
        'Category'  : jd_category,
        'Job_Title' : jd_title,
        'JD_Number' : jd_number,
        'Correct'   : sbert_correct,
        'Accuracy'  : sbert_correct / 10 * 100
    })

    bert_jd_accuracies.append({
        'Category'  : jd_category,
        'Job_Title' : jd_title,
        'JD_Number' : jd_number,
        'Correct'   : bert_correct,
        'Accuracy'  : bert_correct / 10 * 100
    })

    # Print every JD result
    print(f"\n{'='*60}")
    print(f"📋 JD #{i+1} | {jd_category} | {jd_title} (JD {jd_number})")
    print(f"{'='*60}")

    print(f"\n🔵 SBERT Top 10:")
    print(f"{'Rank':<6}{'Resume_ID':<12}{'Category':<28}{'Score':<10}")
    print("-"*56)
    for rank, (_, r) in enumerate(sbert_top.iterrows(), 1):
        match = "✅" if r['Category'] == jd_category else "  "
        print(f"{rank:<6}{int(r['Resume_ID']):<12}"
              f"{r['Category']:<28}{r['SBERT_Score']:.4f}  {match}")

    print(f"\n🟡 BERT Top 10:")
    print(f"{'Rank':<6}{'Resume_ID':<12}{'Category':<28}{'Score':<10}")
    print("-"*56)
    for rank, (_, r) in enumerate(bert_top.iterrows(), 1):
        match = "✅" if r['Category'] == jd_category else "  "
        print(f"{rank:<6}{int(r['Resume_ID']):<12}"
              f"{r['Category']:<28}{r['BERT_Score']:.4f}  {match}")

    print(f"\n📊 SBERT: {sbert_correct}/10 correct | "
          f"BERT: {bert_correct}/10 correct")


# ============================================================
# 4D. ACCURACY PER CATEGORY
# Average accuracy across all JDs within same category
# ============================================================

sbert_acc_df = pd.DataFrame(sbert_jd_accuracies)
bert_acc_df  = pd.DataFrame(bert_jd_accuracies)

# Group by category — average accuracy across all JDs in that category
sbert_cat_acc = sbert_acc_df.groupby('Category')['Accuracy'].mean()
bert_cat_acc  = bert_acc_df.groupby('Category')['Accuracy'].mean()

print("\n" + "="*65)
print("ACCURACY PER CATEGORY — Avg across all JDs in that category")
print("="*65)
print(f"{'Category':<25} {'SBERT':>10} {'BERT':>10} {'Winner':>10}")
print("-"*55)

for cat in sorted(sbert_cat_acc.index):
    s_acc  = sbert_cat_acc[cat]
    b_acc  = bert_cat_acc[cat]
    winner = "SBERT 🔵" if s_acc > b_acc else "BERT 🟡"
    print(f"{cat:<25} {s_acc:>9.1f}% {b_acc:>9.1f}% {winner:>10}")

print("-"*55)
print(f"{'OVERALL AVERAGE':<25} "
      f"{sbert_cat_acc.mean():>9.1f}% "
      f"{bert_cat_acc.mean():>9.1f}%")


# ============================================================
# 4E. DETAILED JD LEVEL ACCURACY TABLE
# Shows accuracy for each individual JD
# ============================================================

print("\n" + "="*70)
print("DETAILED JD-LEVEL ACCURACY")
print("="*70)
print(f"{'Category':<25}{'Job Title':<30}{'JD#':<6}"
      f"{'SBERT':>8}{'BERT':>8}")
print("-"*70)

for idx in range(len(sbert_acc_df)):
    s_row = sbert_acc_df.iloc[idx]
    b_row = bert_acc_df.iloc[idx]
    print(f"{s_row['Category']:<25}"
          f"{s_row['Job_Title'][:28]:<30}"
          f"{int(s_row['JD_Number']):<6}"
          f"{s_row['Accuracy']:>7.0f}%"
          f"{b_row['Accuracy']:>7.0f}%")


# ============================================================
# 4F. OVERALL WINNER STATS
# ============================================================

overall_sbert = sbert_cat_acc.mean()
overall_bert  = bert_cat_acc.mean()

print("\n" + "="*55)
print("OVERALL RESULTS")
print("="*55)
print(f"✅ SBERT avg accuracy : {overall_sbert:.1f}%")
print(f"✅ BERT  avg accuracy : {overall_bert:.1f}%")
print(f"✅ SBERT improvement  : "
      f"+{overall_sbert - overall_bert:.1f}% over BERT")
print(f"✅ Winner             : "
      f"{'SBERT' if overall_sbert > overall_bert else 'BERT'}")
print(f"✅ Categories ranked  : {jd_df['Category'].nunique()}")
print(f"✅ Total JDs ranked   : {len(jd_df)}")
print(f"✅ Total resumes      : {len(resume_df)}")
print("="*55)


# ============================================================
# 4G. SAVE ALL RESULTS
# ============================================================

# Combine all SBERT results
all_sbert_df = pd.concat([
    df.assign(JD_Key=key)
    for key, df in all_sbert_results.items()
])

all_bert_df = pd.concat([
    df.assign(JD_Key=key)
    for key, df in all_bert_results.items()
])

all_sbert_df.to_csv("results/sbert_rankings.csv",       index=False)
all_bert_df.to_csv("results/bert_rankings.csv",         index=False)
sbert_acc_df.to_csv("results/sbert_jd_accuracy.csv",    index=False)
bert_acc_df.to_csv("results/bert_jd_accuracy.csv",      index=False)

print("\n✅ Results saved:")
print("   → results/sbert_rankings.csv")
print("   → results/bert_rankings.csv")
print("   → results/sbert_jd_accuracy.csv")
print("   → results/bert_jd_accuracy.csv")
print("\n🎉 Step 4 Complete — Ready for Step 5: Evaluation!")