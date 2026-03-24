# ============================================================
# STEP 9B: FINE-TUNE SBERT ON RESUME-JD PAIRS
# Uses CosineSimilarityLoss on labeled pairs
# ============================================================

import pandas as pd
import numpy as np
import torch
import os
import time
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)

os.makedirs("models",     exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# ============================================================
# 9B-1. LOAD TRAINING PAIRS
# ============================================================

train_df = pd.read_csv("data/finetune_train.csv")
val_df   = pd.read_csv("data/finetune_val.csv")

print("="*55)
print("STEP 9B: FINE-TUNE SBERT")
print("="*55)
print(f"✅ Training pairs   : {len(train_df)}")
print(f"✅ Validation pairs : {len(val_df)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device           : {device.upper()}")


# ============================================================
# 9B-2. LOAD BASE SBERT MODEL
# ============================================================

print("\n⏳ Loading base SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Base SBERT loaded : all-MiniLM-L6-v2")


# ============================================================
# 9B-3. PREPARE TRAINING DATA
# Convert to InputExample format required by SBERT
# ============================================================

print("\n⏳ Preparing training examples...")

train_examples = [
    InputExample(
        texts  = [row['resume'], row['jd']],
        label  = float(row['label'])
    )
    for _, row in train_df.iterrows()
]

val_examples = [
    InputExample(
        texts  = [row['resume'], row['jd']],
        label  = float(row['label'])
    )
    for _, row in val_df.iterrows()
]

print(f"✅ Training examples   : {len(train_examples)}")
print(f"✅ Validation examples : {len(val_examples)}")

# DataLoader — batches training examples
train_dataloader = DataLoader(
    train_examples,
    shuffle    = True,
    batch_size = 32
)

print(f"✅ Batches per epoch   : {len(train_dataloader)}")


# ============================================================
# 9B-4. DEFINE LOSS FUNCTION
# CosineSimilarityLoss:
# Pushes similar pairs (label=1) closer together
# Pushes dissimilar pairs (label=0) further apart
# ============================================================

train_loss = losses.CosineSimilarityLoss(model)
print(f"\n✅ Loss function : CosineSimilarityLoss")


# ============================================================
# 9B-5. DEFINE EVALUATOR
# ============================================================

val_sentences1 = [ex.texts[0] for ex in val_examples]
val_sentences2 = [ex.texts[1] for ex in val_examples]
val_scores     = [ex.label    for ex in val_examples]

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1 = val_sentences1,
    sentences2 = val_sentences2,
    scores     = val_scores,
    name       = 'resume-jd-val'
)

# Helper function to extract score from dict or float
def extract_score(result):
    if isinstance(result, dict):
        # Try common keys in order
        for key in ['pearson_cosine', 'spearman_cosine',
                    'pearson', 'spearman']:
            if key in result:
                return result[key]
        # If none found just return first value
        return list(result.values())[0]
    return float(result)

# Evaluate base model BEFORE fine-tuning
print("\n⏳ Evaluating BASE model before fine-tuning...")
base_result = evaluator(model)
base_score  = extract_score(base_result)
print(f"✅ Base SBERT validation score : {base_score:.4f}")


# ============================================================
# 9B-6. FINE-TUNE
# ============================================================

NUM_EPOCHS    = 8
WARMUP_STEPS  = int(len(train_dataloader) * NUM_EPOCHS * 0.1)
OUTPUT_PATH   = "models/finetuned_sbert"

print(f"\n{'='*55}")
print(f"STARTING FINE-TUNING")
print(f"{'='*55}")
print(f"   Epochs       : {NUM_EPOCHS}")
print(f"   Batch size   : 32")
print(f"   Warmup steps : {WARMUP_STEPS}")
print(f"   Output path  : {OUTPUT_PATH}")
print(f"{'='*55}\n")

start = time.time()

model.fit(
    train_objectives  = [(train_dataloader, train_loss)],
    evaluator         = evaluator,
    epochs            = NUM_EPOCHS,
    warmup_steps      = WARMUP_STEPS,
    evaluation_steps  = len(train_dataloader),  # eval each epoch
    output_path       = OUTPUT_PATH,
    save_best_model   = True,
    show_progress_bar = True
)

elapsed = time.time() - start
print(f"\n✅ Fine-tuning complete in {elapsed:.1f}s "
      f"({elapsed/60:.1f} mins)")


# ============================================================
# 9B-7. LOAD BEST MODEL AND EVALUATE
# ============================================================

print("\n⏳ Loading best fine-tuned model...")
finetuned_model = SentenceTransformer(OUTPUT_PATH)
print("✅ Fine-tuned model loaded")

print("\n⏳ Evaluating fine-tuned model...")
finetuned_result = evaluator(finetuned_model)
finetuned_score  = extract_score(finetuned_result)

print(f"\n{'='*55}")
print(f"FINE-TUNING RESULTS")
print(f"{'='*55}")
print(f"   Base SBERT score     : {base_score:.4f}")
print(f"   Fine-tuned score     : {finetuned_score:.4f}")
print(f"   Improvement          : "
      f"{finetuned_score - base_score:+.4f}")

if finetuned_score > base_score:
    print(f"   ✅ Fine-tuning HELPED!")
else:
    print(f"   ⚠️  Fine-tuning did not improve — "
          f"base model was better")


# ============================================================
# 9B-8. GENERATE NEW EMBEDDINGS WITH FINE-TUNED MODEL
# ============================================================

resume_df = pd.read_csv("data/cleaned_resumes.csv")
jd_df     = pd.read_csv("data/cleaned_jds.csv")

print(f"\n⏳ Generating embeddings with fine-tuned model...")

# Resume embeddings
ft_resume_emb = finetuned_model.encode(
    resume_df['Cleaned_Resume'].tolist(),
    batch_size        = 32,
    show_progress_bar = True,
    convert_to_numpy  = True,
    device            = device
)

# JD embeddings
ft_jd_emb = finetuned_model.encode(
    jd_df['Cleaned_JD'].tolist(),
    batch_size        = 32,
    show_progress_bar = True,
    convert_to_numpy  = True,
    device            = device
)

print(f"✅ Fine-tuned resume embeddings : {ft_resume_emb.shape}")
print(f"✅ Fine-tuned JD embeddings     : {ft_jd_emb.shape}")

# Save embeddings
np.save("embeddings/finetuned_resume_embeddings.npy", ft_resume_emb)
np.save("embeddings/finetuned_jd_embeddings.npy",     ft_jd_emb)

print("\n✅ Fine-tuned embeddings saved:")
print("   → embeddings/finetuned_resume_embeddings.npy")
print("   → embeddings/finetuned_jd_embeddings.npy")


# ============================================================
# 9B-9. FINAL SUMMARY
# ============================================================

print("\n" + "="*55)
print("FINE-TUNING SUMMARY")
print("="*55)
print(f"✅ Base model          : all-MiniLM-L6-v2")
print(f"✅ Fine-tuned on       : {len(train_examples)} pairs")
print(f"✅ Epochs              : {NUM_EPOCHS}")
print(f"✅ Loss function       : CosineSimilarityLoss")
print(f"✅ Base score          : {base_score:.4f}")
print(f"✅ Fine-tuned score    : {finetuned_score:.4f}")
print(f"✅ Improvement         : "
      f"{finetuned_score - base_score:+.4f}")
print(f"✅ Saved to            : {OUTPUT_PATH}")
print("="*55)
print("\n🎉 Step 9B Complete — Ready for Step 9C: Compare!")