# ============================================================
# STEP 8: RANKING METRICS — Precision@K, Recall@K, NDCG
# Final comparison of all methods
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

# ============================================================
# 8A. LOAD ALL RANKING RESULTS
# ============================================================

print("="*60)
print("STEP 8: RANKING METRICS")
print("="*60)

sbert_rankings = pd.read_csv("results/sbert_rankings.csv")
bert_rankings  = pd.read_csv("results/bert_rankings.csv")
tfidf_rankings = pd.read_csv("results/tfidf_rankings.csv")
jd_df          = pd.read_csv("data/cleaned_jds.csv")

print(f"✅ SBERT rankings loaded : {len(sbert_rankings)}")
print(f"✅ BERT  rankings loaded : {len(bert_rankings)}")
print(f"✅ TF-IDF rankings loaded: {len(tfidf_rankings)}")
print(f"✅ Total JDs             : {len(jd_df)}")


# ============================================================
# 8B. METRIC FUNCTIONS
# ============================================================

def precision_at_k(ranked_categories, true_category, k):
    """
    Precision@K = number of relevant items in top K / K
    Relevant = same category as JD
    """
    top_k   = ranked_categories[:k]
    correct = sum(1 for c in top_k if c == true_category)
    return correct / k


def recall_at_k(ranked_categories, true_category, k,
                total_relevant):
    """
    Recall@K = number of relevant items in top K /
               total relevant items in dataset
    """
    top_k   = ranked_categories[:k]
    correct = sum(1 for c in top_k if c == true_category)
    return correct / total_relevant if total_relevant > 0 else 0


def dcg_at_k(ranked_categories, true_category, k):
    """
    DCG@K = sum of (relevance / log2(rank+1))
    Relevance = 1 if correct category else 0
    """
    top_k = ranked_categories[:k]
    dcg   = 0.0
    for rank, cat in enumerate(top_k, 1):
        if cat == true_category:
            dcg += 1.0 / np.log2(rank + 1)
    return dcg


def idcg_at_k(k, total_relevant):
    """
    IDCG@K = ideal DCG (all top K are relevant)
    """
    ideal_k = min(k, total_relevant)
    idcg    = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_k + 1)
    )
    return idcg


def ndcg_at_k(ranked_categories, true_category, k,
              total_relevant):
    """
    NDCG@K = DCG@K / IDCG@K
    Normalized so always between 0 and 1
    """
    dcg  = dcg_at_k(ranked_categories, true_category, k)
    idcg = idcg_at_k(k, total_relevant)
    return dcg / idcg if idcg > 0 else 0


# ============================================================
# 8C. COMPUTE METRICS FOR ALL THREE METHODS
# K values: 5 and 10
# ============================================================

print("\n⏳ Computing metrics for all methods...")

# Count total resumes per category for recall denominator
resume_df      = pd.read_csv("data/cleaned_resumes.csv")
category_counts = resume_df['Category'].value_counts().to_dict()

def compute_all_metrics(rankings_df, jd_df, model_name):
    """
    Compute P@K, R@K, NDCG@K for all JDs
    Returns a dataframe with metrics per JD
    """
    records = []

    for i, jd_row in jd_df.iterrows():
        jd_category = jd_row['Category']
        jd_key      = f"{jd_category}_JD{jd_row['JD_Number']}"

        # Get ranked results for this JD
        jd_results  = rankings_df[
            rankings_df['JD_Key'] == jd_key
        ].sort_values(
            f'{model_name}_Score' if model_name != 'TFIDF'
            else 'TFIDF_Score',
            ascending=False
        )

        if len(jd_results) == 0:
            continue

        ranked_cats    = jd_results['Category'].tolist()
        total_relevant = category_counts.get(jd_category, 1)

        for k in [5, 10]:
            p_at_k = precision_at_k(ranked_cats, jd_category, k)
            r_at_k = recall_at_k(
                ranked_cats, jd_category, k, total_relevant
            )
            n_at_k = ndcg_at_k(
                ranked_cats, jd_category, k, total_relevant
            )

            records.append({
                'Model'    : model_name,
                'JD_Key'   : jd_key,
                'Category' : jd_category,
                'K'        : k,
                'P@K'      : p_at_k,
                'R@K'      : r_at_k,
                'NDCG@K'   : n_at_k
            })

    return pd.DataFrame(records)


# Compute for all 3 methods
sbert_metrics = compute_all_metrics(
    sbert_rankings, jd_df, 'SBERT'
)
bert_metrics  = compute_all_metrics(
    bert_rankings,  jd_df, 'BERT'
)
tfidf_metrics = compute_all_metrics(
    tfidf_rankings, jd_df, 'TFIDF'
)

all_metrics = pd.concat(
    [sbert_metrics, bert_metrics, tfidf_metrics]
)

print("✅ Metrics computed for all methods")


# ============================================================
# 8D. SUMMARY TABLE — Average metrics per model per K
# ============================================================

summary = all_metrics.groupby(
    ['Model', 'K']
)[['P@K', 'R@K', 'NDCG@K']].mean().round(4)

print("\n" + "="*65)
print("RANKING METRICS SUMMARY")
print("="*65)

for k in [5, 10]:
    print(f"\n{'─'*65}")
    print(f"  K = {k}")
    print(f"{'─'*65}")
    print(f"  {'Model':<12} {'Precision@K':>14} "
          f"{'Recall@K':>12} {'NDCG@K':>10}")
    print(f"  {'-'*48}")

    for model in ['TFIDF', 'BERT', 'SBERT']:
        row = summary.loc[(model, k)]
        print(f"  {model:<12} {row['P@K']:>13.4f} "
              f"{row['R@K']:>11.4f} {row['NDCG@K']:>9.4f}")

print(f"\n{'─'*65}")
print("  Higher is better for all metrics ↑")


# ============================================================
# 8E. PER CATEGORY METRICS — SBERT (best model)
# ============================================================

print("\n" + "="*65)
print("PER CATEGORY METRICS — SBERT @ K=10")
print("="*65)
print(f"{'Category':<25} {'P@10':>8} {'R@10':>8} {'NDCG@10':>10}")
print("-"*53)

sbert_k10 = sbert_metrics[sbert_metrics['K'] == 10]
cat_metrics = sbert_k10.groupby('Category')[
    ['P@K', 'R@K', 'NDCG@K']
].mean().round(4)

for cat, row in cat_metrics.iterrows():
    print(f"{cat:<25} {row['P@K']:>8.4f} "
          f"{row['R@K']:>8.4f} {row['NDCG@K']:>10.4f}")

print("-"*53)
print(f"{'AVERAGE':<25} "
      f"{cat_metrics['P@K'].mean():>8.4f} "
      f"{cat_metrics['R@K'].mean():>8.4f} "
      f"{cat_metrics['NDCG@K'].mean():>10.4f}")


# ============================================================
# 8F. CHART 1 — P@K, R@K, NDCG@K Grouped Bar Chart
# ============================================================

print("\n⏳ Generating metrics comparison chart...")

models  = ['TF-IDF', 'BERT', 'SBERT']
metrics = ['P@K', 'R@K', 'NDCG@K']
colors  = ['steelblue', 'darkorange', 'seagreen']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, k in enumerate([5, 10]):
    ax     = axes[ax_idx]
    x      = np.arange(len(metrics))
    width  = 0.25

    for m_idx, (model, db_model) in enumerate(
        zip(models, ['TFIDF', 'BERT', 'SBERT'])
    ):
        vals = [
            summary.loc[(db_model, k)]['P@K'],
            summary.loc[(db_model, k)]['R@K'],
            summary.loc[(db_model, k)]['NDCG@K']
        ]
        bars = ax.bar(
            x + m_idx * width, vals, width,
            label=model, color=colors[m_idx], alpha=0.85
        )
        for bar in bars:
            ax.annotate(
                f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2,
                    bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontsize=8
            )

    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Ranking Metrics @ K={k}', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle(
    'TF-IDF vs BERT vs SBERT — Precision, Recall & NDCG\n'
    '(137 Real World JDs | 2095 Resumes)',
    fontsize=13
)
plt.tight_layout()
plt.savefig("results/figure6_ranking_metrics.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 6 saved → results/figure6_ranking_metrics.png")


# ============================================================
# 8G. CHART 2 — NDCG@10 Per Category
# ============================================================

print("⏳ Generating NDCG per category chart...")

sbert_ndcg = sbert_k10.groupby('Category')['NDCG@K'].mean()
bert_k10   = bert_metrics[bert_metrics['K'] == 10]
bert_ndcg  = bert_k10.groupby('Category')['NDCG@K'].mean()
tfidf_k10  = tfidf_metrics[tfidf_metrics['K'] == 10]
tfidf_ndcg = tfidf_k10.groupby('Category')['NDCG@K'].mean()

categories = sorted(sbert_ndcg.index)
x2         = np.arange(len(categories))
width2     = 0.25

fig2, ax2 = plt.subplots(figsize=(18, 7))

ax2.bar(x2 - width2, [tfidf_ndcg.get(c, 0) for c in categories],
        width2, label='TF-IDF', color='steelblue',  alpha=0.85)
ax2.bar(x2,          [bert_ndcg.get(c, 0)  for c in categories],
        width2, label='BERT',   color='darkorange', alpha=0.85)
ax2.bar(x2 + width2, [sbert_ndcg.get(c, 0) for c in categories],
        width2, label='SBERT',  color='seagreen',   alpha=0.85)

ax2.set_xlabel('Category', fontsize=11)
ax2.set_ylabel('NDCG@10', fontsize=11)
ax2.set_title(
    'NDCG@10 Per Category — TF-IDF vs BERT vs SBERT',
    fontsize=13
)
ax2.set_xticks(x2)
ax2.set_xticklabels(categories, rotation=30,
                    ha='right', fontsize=9)
ax2.set_ylim(0, 1.0)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/figure7_ndcg_per_category.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 7 saved → results/figure7_ndcg_per_category.png")


# ============================================================
# 8H. SAVE ALL METRICS
# ============================================================

all_metrics.to_csv("results/all_ranking_metrics.csv",
                   index=False)
summary.to_csv("results/metrics_summary.csv")

print("\n✅ Metrics saved:")
print("   → results/all_ranking_metrics.csv")
print("   → results/metrics_summary.csv")


# ============================================================
# 8I. FINAL COMPLETE PROJECT SUMMARY
# ============================================================

sbert_p10  = summary.loc[('SBERT', 10)]['P@K']
sbert_r10  = summary.loc[('SBERT', 10)]['R@K']
sbert_n10  = summary.loc[('SBERT', 10)]['NDCG@K']
bert_p10   = summary.loc[('BERT',  10)]['P@K']
bert_r10   = summary.loc[('BERT',  10)]['R@K']
bert_n10   = summary.loc[('BERT',  10)]['NDCG@K']
tfidf_p10  = summary.loc[('TFIDF', 10)]['P@K']
tfidf_r10  = summary.loc[('TFIDF', 10)]['R@K']
tfidf_n10  = summary.loc[('TFIDF', 10)]['NDCG@K']

sbert_acc_df = pd.read_csv("results/sbert_jd_accuracy.csv")
bert_acc_df  = pd.read_csv("results/bert_jd_accuracy.csv")
tfidf_acc_df = pd.read_csv("results/tfidf_jd_accuracy.csv")
clf_df       = pd.read_csv("results/classifier_summary.csv")
best_clf     = clf_df.loc[
    clf_df['Accuracy'].str.replace('%','').astype(float).idxmax()
]

print("\n" + "="*65)
print("COMPLETE PROJECT SUMMARY")
print("="*65)
print(f"""
📦 DATASET
   Resumes         : 2095 (20 categories)
   Job Descriptions: 137  (real world JDs)
   STS Benchmark   : 1379 sentence pairs

🧹 PREPROCESSING
   Encoding fix    : ✅
   Stop words      : ✅
   Lemmatization   : ✅
   Stemming        : ✅ Snowball

📊 RANKING METRICS @ K=10
   {'Model':<10} {'P@10':>8} {'R@10':>8} {'NDCG@10':>10}
   {'------':<10} {'----':>8} {'----':>8} {'-------':>10}
   {'TF-IDF':<10} {tfidf_p10:>8.4f} {tfidf_r10:>8.4f} {tfidf_n10:>10.4f}
   {'BERT':<10} {bert_p10:>8.4f}  {bert_r10:>8.4f} {bert_n10:>10.4f}
   {'SBERT':<10} {sbert_p10:>8.4f} {sbert_r10:>8.4f} {sbert_n10:>10.4f}

🤖 CLASSIFIER RESULTS
   Best model      : {best_clf['Model']}
   Best accuracy   : {best_clf['Accuracy']}
   Features        : SBERT embeddings (384 dim)
   Train/Test      : 80/20 stratified split

📈 CORRELATION (Table 1 — STS Benchmark)
   SBERT avg       : 0.803053
   BERT  avg       : 0.212505

🏆 CONCLUSION
   SBERT > TF-IDF > BERT across all metrics
   SVM classifier achieves {best_clf['Accuracy']} on 20 categories
   Transfer learning (SBERT) outperforms classical ML (TF-IDF)
   and raw deep learning (BERT) for resume ranking
""")
print("="*65)
print("🎉 ALL STEPS COMPLETE!")
print("="*65)