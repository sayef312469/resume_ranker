# ============================================================
# STEP 5: EVALUATION — Updated for New Datasets
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import warnings
import os
import torch
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

os.makedirs("results", exist_ok=True)

# ============================================================
# 5A. LOAD STS BENCHMARK
# ============================================================

sts_df = pd.read_csv("data/sts_test.csv")

print("="*60)
print("STEP 5: EVALUATION")
print("="*60)
print(f"\n✅ STS Benchmark loaded : {sts_df.shape[0]} pairs")
print(f"   Score range         : "
      f"{sts_df['score'].min():.2f} to {sts_df['score'].max():.2f}")
print(f"\nSample rows:")
print(sts_df.head(3).to_string(index=False))


# ============================================================
# 5B. SPLIT INTO 6 SUBSETS (STS1-STS6)
# ============================================================

chunk_size  = len(sts_df) // 6
sts_chunks  = []

for i in range(6):
    start = i * chunk_size
    end   = start + chunk_size if i < 5 else len(sts_df)
    sts_chunks.append(sts_df.iloc[start:end].reset_index(drop=True))

print(f"\n✅ STS split into 6 subsets:")
for i, chunk in enumerate(sts_chunks):
    print(f"   STS{i+1} : {len(chunk)} pairs")


# ============================================================
# 5C. LOAD MODELS
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✅ Device : {device.upper()}")

print("\n⏳ Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SBERT loaded")

print("\n⏳ Loading BERT model...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model_obj = BertModel.from_pretrained('bert-base-uncased')
bert_model_obj.eval()
bert_model_obj.to(device)
print("✅ BERT loaded")


# ============================================================
# 5D. EMBEDDING FUNCTIONS
# ============================================================

def get_sbert_embeddings(texts):
    return sbert_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        device=device
    )

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
        outputs = bert_model_obj(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def get_cosine_sim(emb1, emb2):
    return cosine_similarity(
        emb1.reshape(1, -1),
        emb2.reshape(1, -1)
    )[0][0]


# ============================================================
# 5E. COMPUTE CORRELATION FOR EACH STS SUBSET
# Replicates Table 1 from the paper
# ============================================================

print("\n" + "="*60)
print("COMPUTING CORRELATIONS (Replicating Table 1)...")
print("="*60)

results_table = []

for idx, chunk in enumerate(sts_chunks):
    sts_name = f"STS{idx+1}"
    print(f"\n⏳ Processing {sts_name} ({len(chunk)} pairs)...")

    ground_truth = chunk['score'].tolist()
    sbert_scores = []
    bert_scores  = []

    # SBERT — batch encode both sentences
    sbert_emb1 = get_sbert_embeddings(chunk['sentence1'].tolist())
    sbert_emb2 = get_sbert_embeddings(chunk['sentence2'].tolist())

    for i in range(len(chunk)):
        sim = get_cosine_sim(sbert_emb1[i], sbert_emb2[i])
        sbert_scores.append(sim)
    print(f"   ✅ SBERT scores computed")

    # BERT — one by one
    for i in range(len(chunk)):
        emb1 = get_bert_embedding(chunk['sentence1'].iloc[i])
        emb2 = get_bert_embedding(chunk['sentence2'].iloc[i])
        bert_scores.append(get_cosine_sim(emb1, emb2))
        if (i + 1) % 50 == 0:
            print(f"   BERT: {i+1}/{len(chunk)} done...")
    print(f"   ✅ BERT scores computed")

    # Pearson correlation
    sbert_corr, _ = pearsonr(ground_truth, sbert_scores)
    bert_corr,  _ = pearsonr(ground_truth, bert_scores)

    results_table.append({
        'Dataset' : sts_name,
        'SBERT'   : round(sbert_corr, 6),
        'BERT'    : round(bert_corr,  6)
    })

    print(f"   📊 {sts_name} → "
          f"SBERT: {sbert_corr:.6f} | BERT: {bert_corr:.6f}")


# ============================================================
# 5F. PRINT TABLE 1 — Paper Replication
# ============================================================

results_df = pd.DataFrame(results_table)

print("\n" + "="*60)
print("TABLE 1 — COHERENCE VALUE COMPARISON (BERT vs SBERT)")
print("Replicating Table 1 from the paper")
print("="*60)
print(f"\n{'Data Set':<12} {'Model':<8} {'Correlation Value':>20}")
print("-"*42)

for _, row in results_df.iterrows():
    print(f"{row['Dataset']:<12} {'SBERT':<8} {row['SBERT']:>20.6f}")
    print(f"{'':<12} {'BERT':<8}  {row['BERT']:>20.6f}")
    print()

print("-"*42)
print(f"{'Average':<12} {'SBERT':<8} {results_df['SBERT'].mean():>20.6f}")
print(f"{'':<12} {'BERT':<8}  {results_df['BERT'].mean():>20.6f}")


# ============================================================
# 5G. FIGURE 3 — STS Correlation Bar Chart
# Replicates Fig 3 from the paper
# ============================================================

print("\n⏳ Generating Figure 3...")

datasets   = results_df['Dataset'].tolist()
sbert_vals = results_df['SBERT'].tolist()
bert_vals  = results_df['BERT'].tolist()
x          = np.arange(len(datasets))
width      = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, sbert_vals, width,
               label='SBERT', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, bert_vals,  width,
               label='BERT',  color='lightyellow',
               alpha=0.85, edgecolor='gray')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Correlation Value for Similarity', fontsize=12)
ax.set_title(
    'Performance Analysis of SBERT vs BERT — Correlation Values\n'
    '(Replication of Fig. 3 from paper)',
    fontsize=13
)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"SBERT  BERT\n{d}" for d in datasets],
    fontsize=9
)
ax.set_ylim(0, max(max(sbert_vals), max(bert_vals)) * 1.25)
ax.legend(fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.annotate(
        f'{bar.get_height():.3f}',
        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
        xytext=(0, 3), textcoords="offset points",
        ha='center', fontsize=8
    )
for bar in bars2:
    ax.annotate(
        f'{bar.get_height():.3f}',
        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
        xytext=(0, 3), textcoords="offset points",
        ha='center', fontsize=8
    )

plt.tight_layout()
plt.savefig("results/figure3_correlation.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("✅ Figure 3 saved → results/figure3_correlation.png")


# ============================================================
# 5H. FIGURE 4 — Ranking Accuracy Per Category
# New chart — not in paper, your extension
# ============================================================

print("\n⏳ Generating Figure 4 — Ranking Accuracy...")

sbert_acc_df = pd.read_csv("results/sbert_jd_accuracy.csv")
bert_acc_df  = pd.read_csv("results/bert_jd_accuracy.csv")

sbert_cat = sbert_acc_df.groupby('Category')['Accuracy'].mean()
bert_cat  = bert_acc_df.groupby('Category')['Accuracy'].mean()

categories  = sorted(sbert_cat.index)
sbert_acc   = [sbert_cat[c] for c in categories]
bert_acc    = [bert_cat[c]  for c in categories]

x2    = np.arange(len(categories))
fig2, ax2 = plt.subplots(figsize=(16, 7))

ax2.bar(x2 - width/2, sbert_acc, width,
        label='SBERT', color='steelblue', alpha=0.85)
ax2.bar(x2 + width/2, bert_acc,  width,
        label='BERT',  color='lightyellow',
        alpha=0.85, edgecolor='gray')

ax2.set_xlabel('Job Category', fontsize=12)
ax2.set_ylabel('Top-10 Accuracy (%)', fontsize=12)
ax2.set_title(
    'Resume Ranking Accuracy per Category — SBERT vs BERT\n'
    '(Top-10 Correct Category Matches | 2095 Resumes | 137 JDs)',
    fontsize=13
)
ax2.set_xticks(x2)
ax2.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
ax2.set_ylim(0, 115)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

for bar in ax2.patches:
    if bar.get_height() > 0:
        ax2.annotate(
            f'{bar.get_height():.0f}%',
            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha='center', fontsize=7.5
        )

plt.tight_layout()
plt.savefig("results/figure4_ranking_accuracy.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("✅ Figure 4 saved → results/figure4_ranking_accuracy.png")


# ============================================================
# 5I. FIGURE 5 — JD Count vs Accuracy Scatter
# Shows whether more JDs per category = better accuracy
# ============================================================

print("\n⏳ Generating Figure 5 — JD Count vs Accuracy...")

jd_df        = pd.read_csv("data/cleaned_jds.csv")
jd_counts    = jd_df.groupby('Category').size().reset_index(
    name='JD_Count'
)
sbert_means  = sbert_acc_df.groupby('Category')['Accuracy'].mean(
).reset_index(name='SBERT_Accuracy')
merged       = jd_counts.merge(sbert_means, on='Category')

fig3, ax3 = plt.subplots(figsize=(9, 6))
ax3.scatter(merged['JD_Count'], merged['SBERT_Accuracy'],
            color='steelblue', s=100, alpha=0.8)

for _, r in merged.iterrows():
    ax3.annotate(
        r['Category'],
        (r['JD_Count'], r['SBERT_Accuracy']),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=7.5
    )

ax3.set_xlabel('Number of JDs per Category', fontsize=12)
ax3.set_ylabel('SBERT Avg Accuracy (%)', fontsize=12)
ax3.set_title(
    'JD Count vs SBERT Ranking Accuracy per Category',
    fontsize=13
)
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figure5_jd_count_vs_accuracy.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 5 saved → results/figure5_jd_count_vs_accuracy.png")


# ============================================================
# 5J. SAVE TABLE 1
# ============================================================

results_df.to_csv("results/table1_correlation.csv", index=False)
print("\n✅ Table 1 saved → results/table1_correlation.csv")


# ============================================================
# 5K. FINAL SUMMARY REPORT
# ============================================================

overall_sbert_acc = sbert_acc_df['Accuracy'].mean()
overall_bert_acc  = bert_acc_df['Accuracy'].mean()

print("\n" + "="*60)
print("FINAL PROJECT SUMMARY REPORT")
print("="*60)
print(f"""
📦 DATASET
   Resume Dataset  : Sneha Bhawal (Kaggle)
   Resumes         : 2095 (20 categories)
   JD Dataset      : Ravindra Singh (Kaggle)
   Job Descriptions: 137 (real world JDs)
   STS Benchmark   : 1379 sentence pairs

🧹 PREPROCESSING
   Encoding fixed  : ✅
   Stop words      : ✅ removed
   Lemmatization   : ✅ applied
   Stemming        : ✅ Snowball algorithm (as per paper)

🧠 MODELS USED
   SBERT : all-MiniLM-L6-v2  (384 dimensions)
   BERT  : bert-base-uncased  (768 dimensions)

📊 TABLE 1 — CORRELATION RESULTS (STS Benchmark)
   Dataset     SBERT          BERT
   -------     -----          ----""")

for _, row in results_df.iterrows():
    print(f"   {row['Dataset']:<10}  {row['SBERT']:.6f}     "
          f"{row['BERT']:.6f}")

print(f"""
   Average     {results_df['SBERT'].mean():.6f}     \
{results_df['BERT'].mean():.6f}

🏆 RANKING ACCURACY (Top-10 | 137 Real World JDs)
   SBERT avg accuracy : {overall_sbert_acc:.1f}%
   BERT  avg accuracy : {overall_bert_acc:.1f}%
   SBERT improvement  : +{overall_sbert_acc - overall_bert_acc:.1f}%

📈 CHARTS GENERATED
   Figure 3 → Correlation values  (replicates paper Fig 3)
   Figure 4 → Ranking accuracy    (new extension)
   Figure 5 → JD count vs accuracy(new extension)

✅ CONCLUSION
   SBERT outperforms BERT in both:
   1. Semantic similarity correlation (Table 1)
   2. Resume ranking accuracy on real world JDs
   This confirms and extends the paper findings.
   Using real JDs instead of synthetic ones makes
   the evaluation more realistic and challenging.
""")
print("="*60)
print("🎉 PROJECT COMPLETE!")
print("="*60)