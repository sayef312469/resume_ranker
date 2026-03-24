# ============================================================
# STEP 9C: COMPARE BASE SBERT vs FINE-TUNED SBERT
# Full ranking + metrics comparison
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

# ============================================================
# 9C-1. LOAD DATA AND ALL EMBEDDINGS
# ============================================================

resume_df = pd.read_csv("data/cleaned_resumes.csv")
jd_df     = pd.read_csv("data/cleaned_jds.csv")

# Load all embeddings
sbert_resume_emb = np.load(
    "embeddings/sbert_resume_embeddings.npy")
sbert_jd_emb     = np.load(
    "embeddings/sbert_jd_embeddings.npy")
ft_resume_emb    = np.load(
    "embeddings/finetuned_resume_embeddings.npy")
ft_jd_emb        = np.load(
    "embeddings/finetuned_jd_embeddings.npy")
bert_resume_emb  = np.load(
    "embeddings/bert_resume_embeddings.npy")
bert_jd_emb      = np.load(
    "embeddings/bert_jd_embeddings.npy")

import scipy.sparse as sp
tfidf_resume = sp.load_npz(
    "embeddings/tfidf_resume_matrix.npz")
tfidf_jd     = sp.load_npz(
    "embeddings/tfidf_jd_matrix.npz")

print("="*60)
print("STEP 9C: BASE vs FINE-TUNED SBERT COMPARISON")
print("="*60)
print(f"✅ Resumes          : {len(resume_df)}")
print(f"✅ JDs              : {len(jd_df)}")
print(f"✅ All embeddings loaded")


# ============================================================
# 9C-2. RANKING FUNCTION
# ============================================================

def rank_resumes(jd_idx, resume_emb, jd_emb,
                 resume_df, top_n=10,
                 is_sparse=False):
    if is_sparse:
        jd_vec = jd_emb[jd_idx]
        scores = cosine_similarity(jd_vec, resume_emb)[0]
    else:
        jd_vec = jd_emb[jd_idx].reshape(1, -1)
        scores = cosine_similarity(jd_vec, resume_emb)[0]

    results          = resume_df.copy()
    results['Score'] = scores
    results          = results.sort_values(
        'Score', ascending=False
    )
    return results.head(top_n)[
        ['Resume_ID', 'Category', 'Score']
    ].reset_index(drop=True)


# ============================================================
# 9C-3. METRIC FUNCTIONS
# ============================================================

category_counts = resume_df['Category'].value_counts().to_dict()

def precision_at_k(ranked_cats, true_cat, k):
    return sum(1 for c in ranked_cats[:k]
               if c == true_cat) / k

def ndcg_at_k(ranked_cats, true_cat, k, total_rel):
    dcg  = sum(
        1/np.log2(r+2)
        for r, c in enumerate(ranked_cats[:k])
        if c == true_cat
    )
    idcg = sum(
        1/np.log2(r+2)
        for r in range(min(k, total_rel))
    )
    return dcg/idcg if idcg > 0 else 0

def accuracy_at_k(ranked_cats, true_cat, k):
    return sum(1 for c in ranked_cats[:k]
               if c == true_cat) / k * 100


# ============================================================
# 9C-4. RUN ALL 5 MODELS ON ALL 137 JDs
# ============================================================

print("\n⏳ Running all 5 models on 137 JDs...")

models_config = {
    'BERT'         : (bert_resume_emb,  bert_jd_emb,  False),
    'TF-IDF'       : (tfidf_resume,     tfidf_jd,     True),
    'SBERT'        : (sbert_resume_emb, sbert_jd_emb, False),
    'SBERT-FT'     : (ft_resume_emb,    ft_jd_emb,    False),
}

all_records = []

for model_name, (res_emb, jd_emb, is_sp) in models_config.items():
    print(f"   ⏳ Running {model_name}...")

    for i, jd_row in jd_df.iterrows():
        jd_cat   = jd_row['Category']
        jd_num   = jd_row['JD_Number']
        total_rel = category_counts.get(jd_cat, 1)

        top10 = rank_resumes(
            i, res_emb, jd_emb, resume_df,
            top_n=10, is_sparse=is_sp
        )
        ranked_cats = top10['Category'].tolist()

        for k in [5, 10]:
            all_records.append({
                'Model'    : model_name,
                'Category' : jd_cat,
                'JD_Num'   : jd_num,
                'K'        : k,
                'P@K'      : precision_at_k(
                    ranked_cats, jd_cat, k),
                'NDCG@K'   : ndcg_at_k(
                    ranked_cats, jd_cat, k, total_rel),
                'Accuracy' : accuracy_at_k(
                    ranked_cats, jd_cat, k)
            })

    print(f"   ✅ {model_name} done")

metrics_df = pd.DataFrame(all_records)
print("✅ All models evaluated")


# ============================================================
# 9C-5. OVERALL SUMMARY TABLE
# ============================================================

summary = metrics_df.groupby(
    ['Model', 'K']
)[['P@K', 'NDCG@K', 'Accuracy']].mean().round(4)

print("\n" + "="*65)
print("FINAL COMPARISON — ALL MODELS")
print("="*65)

for k in [5, 10]:
    print(f"\n{'─'*65}")
    print(f"  K = {k}")
    print(f"{'─'*65}")
    print(f"  {'Model':<15} {'P@K':>10} "
          f"{'NDCG@K':>10} {'Accuracy':>10}")
    print(f"  {'-'*45}")

    for model in ['BERT', 'TF-IDF', 'SBERT', 'SBERT-FT']:
        row  = summary.loc[(model, k)]
        flag = " ⭐" if model == 'SBERT-FT' else ""
        print(f"  {model:<15} {row['P@K']:>10.4f} "
              f"{row['NDCG@K']:>10.4f} "
              f"{row['Accuracy']:>9.1f}%{flag}")


# ============================================================
# 9C-6. IMPROVEMENT ANALYSIS
# ============================================================

print("\n" + "="*65)
print("IMPROVEMENT OVER BASE SBERT @ K=10")
print("="*65)

base_p    = summary.loc[('SBERT',    10)]['P@K']
base_ndcg = summary.loc[('SBERT',    10)]['NDCG@K']
base_acc  = summary.loc[('SBERT',    10)]['Accuracy']
ft_p      = summary.loc[('SBERT-FT', 10)]['P@K']
ft_ndcg   = summary.loc[('SBERT-FT', 10)]['NDCG@K']
ft_acc    = summary.loc[('SBERT-FT', 10)]['Accuracy']

print(f"\n  {'Metric':<15} {'Base SBERT':>12} "
      f"{'Fine-tuned':>12} {'Change':>10}")
print(f"  {'-'*50}")
print(f"  {'P@10':<15} {base_p:>12.4f} "
      f"{ft_p:>12.4f} {ft_p-base_p:>+9.4f}")
print(f"  {'NDCG@10':<15} {base_ndcg:>12.4f} "
      f"{ft_ndcg:>12.4f} {ft_ndcg-base_ndcg:>+9.4f}")
print(f"  {'Accuracy':<15} {base_acc:>11.1f}% "
      f"{ft_acc:>11.1f}% {ft_acc-base_acc:>+9.1f}%")


# ============================================================
# 9C-7. PER CATEGORY COMPARISON — SBERT vs SBERT-FT
# ============================================================

print("\n" + "="*65)
print("PER CATEGORY — SBERT vs SBERT-FT @ K=10")
print("="*65)
print(f"{'Category':<25} {'Base SBERT':>12} "
      f"{'Fine-tuned':>12} {'Change':>10} {'Better?':>8}")
print("-"*65)

k10 = metrics_df[metrics_df['K'] == 10]

sbert_cat = k10[k10['Model'] == 'SBERT'].groupby(
    'Category')['Accuracy'].mean()
ft_cat    = k10[k10['Model'] == 'SBERT-FT'].groupby(
    'Category')['Accuracy'].mean()

improved  = 0
degraded  = 0

for cat in sorted(sbert_cat.index):
    s_acc  = sbert_cat[cat]
    f_acc  = ft_cat.get(cat, 0)
    change = f_acc - s_acc

    if change > 0:
        status = "✅ Better"
        improved += 1
    elif change < 0:
        status = "❌ Worse"
        degraded += 1
    else:
        status = "➡️  Same"

    print(f"{cat:<25} {s_acc:>11.1f}% "
          f"{f_acc:>11.1f}% {change:>+9.1f}% {status:>8}")

print("-"*65)
print(f"Improved : {improved} categories")
print(f"Degraded : {degraded} categories")


# ============================================================
# 9C-8. CHART 1 — Full Model Comparison Bar Chart
# ============================================================

print("\n⏳ Generating comparison charts...")

models     = ['BERT', 'TF-IDF', 'SBERT', 'SBERT-FT']
labels     = ['BERT', 'TF-IDF', 'SBERT', 'SBERT\nFine-tuned']
colors     = ['tomato', 'steelblue', 'seagreen', 'darkorchid']
metrics_list = ['P@K', 'NDCG@K', 'Accuracy']
metric_labels = ['Precision@10', 'NDCG@10', 'Accuracy@10 (%)']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax_idx, (metric, label) in enumerate(
    zip(metrics_list, metric_labels)
):
    ax   = axes[ax_idx]
    vals = [summary.loc[(m, 10)][metric] for m in models]

    # Scale accuracy to 0-1 for consistency
    if metric == 'Accuracy':
        vals = [v/100 for v in vals]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                  edgecolor='gray')

    # Highlight fine-tuned bar
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2.5)

    for bar, val in zip(bars, vals):
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width()/2,
                bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha='center', fontsize=10, fontweight='bold'
        )

    ax.set_title(label, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('Score', fontsize=10)

plt.suptitle(
    'Complete Model Comparison @ K=10\n'
    'BERT vs TF-IDF vs SBERT vs Fine-tuned SBERT',
    fontsize=14
)
plt.tight_layout()
plt.savefig("results/figure8_full_comparison.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 8 saved → results/figure8_full_comparison.png")


# ============================================================
# 9C-9. CHART 2 — SBERT vs SBERT-FT Per Category
# ============================================================

print("⏳ Generating per category comparison...")

categories  = sorted(sbert_cat.index)
base_vals   = [sbert_cat[c] for c in categories]
ft_vals     = [ft_cat.get(c, 0) for c in categories]

x     = np.arange(len(categories))
width = 0.35

fig2, ax2 = plt.subplots(figsize=(18, 7))

bars1 = ax2.bar(x - width/2, base_vals, width,
                label='Base SBERT',
                color='seagreen', alpha=0.85)
bars2 = ax2.bar(x + width/2, ft_vals, width,
                label='Fine-tuned SBERT',
                color='darkorchid', alpha=0.85)

ax2.set_xlabel('Category', fontsize=11)
ax2.set_ylabel('Accuracy@10 (%)', fontsize=11)
ax2.set_title(
    'Base SBERT vs Fine-tuned SBERT — Per Category Accuracy\n'
    '(Fine-tuning on 3699 Resume-JD Pairs)',
    fontsize=13
)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=30,
                    ha='right', fontsize=9)
ax2.set_ylim(0, 115)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/figure9_sbert_vs_finetuned.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 9 saved → results/figure9_sbert_vs_finetuned.png")


# ============================================================
# 9C-10. SAVE RESULTS
# ============================================================

metrics_df.to_csv("results/all_models_metrics.csv", index=False)
summary.to_csv("results/final_summary.csv")

print("\n✅ Results saved:")
print("   → results/all_models_metrics.csv")
print("   → results/final_summary.csv")


# ============================================================
# 9C-11. COMPLETE FINAL SUMMARY
# ============================================================

print("\n" + "="*65)
print("🏆 COMPLETE PROJECT FINAL SUMMARY")
print("="*65)
print(f"""
📦 DATASET
   Resumes         : 2095 (20 categories)
   JDs             : 137  (real world)
   Fine-tune pairs : 3699 training + 411 validation

🧠 MODELS COMPARED
   1. BERT          (raw deep learning)
   2. TF-IDF        (classical ML baseline)
   3. SBERT         (transfer learning)
   4. SBERT-FT      (fine-tuned transfer learning)

📊 FINAL RANKING METRICS @ K=10
   Model          P@10     NDCG@10   Accuracy
   ------         ----     -------   --------
   BERT         {summary.loc[('BERT',     10)]['P@K']:.4f}    {summary.loc[('BERT',     10)]['NDCG@K']:.4f}    {summary.loc[('BERT',     10)]['Accuracy']:.1f}%
   TF-IDF       {summary.loc[('TF-IDF',   10)]['P@K']:.4f}    {summary.loc[('TF-IDF',   10)]['NDCG@K']:.4f}    {summary.loc[('TF-IDF',   10)]['Accuracy']:.1f}%
   SBERT        {summary.loc[('SBERT',    10)]['P@K']:.4f}    {summary.loc[('SBERT',    10)]['NDCG@K']:.4f}    {summary.loc[('SBERT',    10)]['Accuracy']:.1f}%
   SBERT-FT     {summary.loc[('SBERT-FT', 10)]['P@K']:.4f}    {summary.loc[('SBERT-FT', 10)]['NDCG@K']:.4f}    {summary.loc[('SBERT-FT', 10)]['Accuracy']:.1f}%

🤖 CLASSIFIER (SVM on SBERT embeddings)
   Accuracy        : 66.11%
   Train/Test split: 80/20 stratified

📈 STS CORRELATION (Table 1)
   SBERT avg       : 0.803053
   BERT  avg       : 0.212505

📊 FINE-TUNING IMPROVEMENT
   Validation score: 0.3521 → 0.8453 (+0.4931)
   Ranking P@10    : {base_p:.4f} → {ft_p:.4f} ({ft_p-base_p:+.4f})
   NDCG@10         : {base_ndcg:.4f} → {ft_ndcg:.4f} ({ft_ndcg-base_ndcg:+.4f})

🏆 FINAL ORDER
   SBERT-FT > SBERT > TF-IDF > BERT
""")
print("="*65)
print("🎉 PROJECT 100% COMPLETE!")
print("="*65)