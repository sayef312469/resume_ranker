# ============================================================
# STEP 9A: GENERATE TRAINING PAIRS FOR SBERT FINE-TUNING
# Creates (resume, jd, similarity_score) pairs
# Same category = 1.0, Different category = 0.0
# ============================================================

import pandas as pd
import numpy as np
import random
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)
random.seed(42)
np.random.seed(42)

# ============================================================
# 9A-1. LOAD DATA
# ============================================================

resume_df = pd.read_csv("data/cleaned_resumes.csv")
jd_df     = pd.read_csv("data/cleaned_jds.csv")

print("="*55)
print("STEP 9A: GENERATE TRAINING PAIRS")
print("="*55)
print(f"✅ Resumes loaded : {len(resume_df)}")
print(f"✅ JDs loaded     : {len(jd_df)}")
print(f"✅ Categories     : {resume_df['Category'].nunique()}")


# ============================================================
# 9A-2. GROUP BY CATEGORY
# ============================================================

# Group resumes by category
resume_by_cat = {}
for cat in resume_df['Category'].unique():
    resume_by_cat[cat] = resume_df[
        resume_df['Category'] == cat
    ]['Cleaned_Resume'].tolist()

# Group JDs by category
jd_by_cat = {}
for cat in jd_df['Category'].unique():
    jd_by_cat[cat] = jd_df[
        jd_df['Category'] == cat
    ]['Cleaned_JD'].tolist()

categories = list(jd_by_cat.keys())

print(f"\n✅ Categories with JDs : {len(categories)}")
print(f"   {categories}")


# ============================================================
# 9A-3. GENERATE POSITIVE PAIRS
# Resume + JD from SAME category → score = 1.0
# ============================================================

print("\n⏳ Generating positive pairs (same category)...")

positive_pairs = []

for cat in categories:
    resumes = resume_by_cat.get(cat, [])
    jds     = jd_by_cat.get(cat, [])

    if len(resumes) == 0 or len(jds) == 0:
        continue

    # For each JD pair it with up to 15 resumes
    # from the same category
    for jd_text in jds:
        sampled_resumes = random.sample(
            resumes, min(30, len(resumes))
        )
        for resume_text in sampled_resumes:
            positive_pairs.append({
                'resume'   : resume_text,
                'jd'       : jd_text,
                'label'    : 1.0,
                'category' : cat
            })

print(f"✅ Positive pairs generated : {len(positive_pairs)}")


# ============================================================
# 9A-4. GENERATE NEGATIVE PAIRS
# Resume + JD from DIFFERENT category → score = 0.0
# Keep equal to positive pairs for balance
# ============================================================

print("⏳ Generating negative pairs (different category)...")

negative_pairs = []
target_negatives = len(positive_pairs)

while len(negative_pairs) < target_negatives:
    # Pick random resume category
    cat1 = random.choice(categories)
    # Pick different JD category
    cat2 = random.choice([c for c in categories if c != cat1])

    resumes = resume_by_cat.get(cat1, [])
    jds     = jd_by_cat.get(cat2, [])

    if len(resumes) == 0 or len(jds) == 0:
        continue

    resume_text = random.choice(resumes)
    jd_text     = random.choice(jds)

    negative_pairs.append({
        'resume'   : resume_text,
        'jd'       : jd_text,
        'label'    : 0.0,
        'category' : f"{cat1}_neg_{cat2}"
    })

print(f"✅ Negative pairs generated : {len(negative_pairs)}")


# ============================================================
# 9A-5. COMBINE AND SHUFFLE
# ============================================================

all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)
pairs_df  = pd.DataFrame(all_pairs)

print(f"\n✅ Total pairs : {len(pairs_df)}")
print(f"   Positive    : {(pairs_df['label']==1.0).sum()}")
print(f"   Negative    : {(pairs_df['label']==0.0).sum()}")
print(f"   Balance     : "
      f"{(pairs_df['label']==1.0).sum()/len(pairs_df)*100:.1f}% "
      f"positive")


# ============================================================
# 9A-6. TRAIN / VALIDATION SPLIT
# 90% train, 10% validation
# ============================================================

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    pairs_df,
    test_size    = 0.1,
    random_state = 42
)

print(f"\n✅ Train/Val split (90/10):")
print(f"   Training pairs   : {len(train_df)}")
print(f"   Validation pairs : {len(val_df)}")


# ============================================================
# 9A-7. SAVE
# ============================================================

train_df.to_csv("data/finetune_train.csv", index=False)
val_df.to_csv("data/finetune_val.csv",     index=False)
pairs_df.to_csv("data/finetune_all.csv",   index=False)

print("\n✅ Saved:")
print("   → data/finetune_train.csv")
print("   → data/finetune_val.csv")
print("   → data/finetune_all.csv")


# ============================================================
# 9A-8. SAMPLE PREVIEW
# ============================================================

print("\n" + "="*55)
print("SAMPLE PAIRS")
print("="*55)

print("\n🟢 Positive pair sample:")
pos_sample = pairs_df[pairs_df['label'] == 1.0].iloc[0]
print(f"   Category : {pos_sample['category']}")
print(f"   Resume   : {pos_sample['resume'][:100]}...")
print(f"   JD       : {pos_sample['jd'][:100]}...")
print(f"   Label    : {pos_sample['label']}")

print("\n🔴 Negative pair sample:")
neg_sample = pairs_df[pairs_df['label'] == 0.0].iloc[0]
print(f"   Category : {neg_sample['category']}")
print(f"   Resume   : {neg_sample['resume'][:100]}...")
print(f"   JD       : {neg_sample['jd'][:100]}...")
print(f"   Label    : {neg_sample['label']}")


# ============================================================
# 9A-9. FINAL SUMMARY
# ============================================================

print("\n" + "="*55)
print("PAIR GENERATION SUMMARY")
print("="*55)
print(f"✅ Total pairs       : {len(pairs_df)}")
print(f"✅ Training pairs    : {len(train_df)}")
print(f"✅ Validation pairs  : {len(val_df)}")
print(f"✅ Positive ratio    : 50%")
print(f"✅ Negative ratio    : 50%")
print(f"✅ Categories covered: {len(categories)}")
print("="*55)
print("\n🎉 Step 9A Complete — Ready for Step 9B: Fine-tuning!")