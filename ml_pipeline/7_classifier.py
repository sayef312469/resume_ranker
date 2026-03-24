# ============================================================
# STEP 7: RESUME CLASSIFIER
# Train/Test Split + 4 ML classifiers on SBERT embeddings
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

# ============================================================
# 7A. LOAD DATA AND EMBEDDINGS
# We use SBERT embeddings as features — best quality
# ============================================================

resume_df        = pd.read_csv("data/cleaned_resumes.csv")
sbert_resume_emb = np.load("embeddings/sbert_resume_embeddings.npy")

print("="*55)
print("STEP 7: RESUME CLASSIFIER")
print("="*55)
print(f"✅ Resumes loaded       : {len(resume_df)}")
print(f"✅ SBERT embeddings     : {sbert_resume_emb.shape}")
print(f"✅ Categories           : {resume_df['Category'].nunique()}")
print(f"\nCategory distribution:")
print(resume_df['Category'].value_counts().to_string())


# ============================================================
# 7B. PREPARE FEATURES AND LABELS
# X = SBERT embeddings (384 dimensional vectors)
# y = Category labels (encoded as integers)
# ============================================================

# Encode category labels to integers
le = LabelEncoder()
y  = le.fit_transform(resume_df['Category'].values)
X  = sbert_resume_emb

print(f"\n✅ Features shape : {X.shape}")
print(f"✅ Labels shape   : {y.shape}")
print(f"✅ Classes        : {list(le.classes_)}")


# ============================================================
# 7C. TRAIN / TEST SPLIT
# 80% train, 20% test — stratified to keep category balance
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y      # keeps category proportions equal
)

print(f"\n✅ Train/Test Split (80/20 stratified):")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")


# ============================================================
# 7D. DEFINE 4 CLASSIFIERS
# ============================================================

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter     = 1000,
        random_state = 42,
        C            = 1.0
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators = 200,
        random_state = 42,
        n_jobs       = -1
    ),
    "SVM": SVC(
        kernel       = 'rbf',
        C            = 1.0,
        random_state = 42,
        probability  = True
    ),
    "Naive Bayes": GaussianNB()
}


# ============================================================
# 7E. TRAIN AND EVALUATE ALL 4 CLASSIFIERS
# ============================================================

results       = {}
best_model    = None
best_accuracy = 0
best_name     = ""

print("\n" + "="*55)
print("TRAINING AND EVALUATING CLASSIFIERS...")
print("="*55)

for name, clf in classifiers.items():
    print(f"\n⏳ Training {name}...")

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names = le.classes_,
        output_dict  = True
    )

    results[name] = {
        'model'     : clf,
        'accuracy'  : acc,
        'y_pred'    : y_pred,
        'report'    : report
    }

    print(f"✅ {name} done")
    print(f"   Test Accuracy : {acc*100:.2f}%")
    print(f"   Macro F1      : "
          f"{report['macro avg']['f1-score']*100:.2f}%")
    print(f"   Weighted F1   : "
          f"{report['weighted avg']['f1-score']*100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model    = clf
        best_name     = name

# ============================================================
# 7E2. ENSEMBLE VOTING CLASSIFIER
# Combines all 4 models using soft voting
# ============================================================

from sklearn.ensemble import VotingClassifier

print("\n" + "="*55)
print("TRAINING ENSEMBLE VOTING CLASSIFIER...")
print("="*55)

# Build voting classifier from the 4 trained estimators
# Use soft voting — averages predicted probabilities
voting_clf = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(
            max_iter=1000, random_state=42, C=1.0)),
        ('rf',  RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1)),
        ('svm', SVC(
            kernel='rbf', C=1.0,
            random_state=42, probability=True)),
        ('nb',  GaussianNB())
    ],
    voting='soft'   # uses predicted probabilities
)

print("⏳ Training Voting Classifier (soft voting)...")
print("   (this trains all 4 models again internally)")

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

voting_acc    = accuracy_score(y_test, y_pred_voting)
voting_report = classification_report(
    y_test, y_pred_voting,
    target_names=le.classes_,
    output_dict=True
)

print(f"✅ Voting Classifier done")
print(f"   Test Accuracy : {voting_acc*100:.2f}%")
print(f"   Macro F1      : "
      f"{voting_report['macro avg']['f1-score']*100:.2f}%")
print(f"   Weighted F1   : "
      f"{voting_report['weighted avg']['f1-score']*100:.2f}%")

# Add to results dictionary
results["Voting Ensemble"] = {
    'model'    : voting_clf,
    'accuracy' : voting_acc,
    'y_pred'   : y_pred_voting,
    'report'   : voting_report
}

# Update best model if voting is better
if voting_acc > best_accuracy:
    best_accuracy = voting_acc
    best_model    = voting_clf
    best_name     = "Voting Ensemble"
    print(f"\n🏆 Voting Ensemble is the new best model!")
else:
    print(f"\n   Previous best ({best_name}: "
          f"{best_accuracy*100:.2f}%) still leads")


# ============================================================
# 7F. DETAILED COMPARISON TABLE
# ============================================================

print("\n" + "="*65)
print("CLASSIFIER COMPARISON")
print("="*65)
print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} "
      f"{'Weighted F1':>12}")
print("-"*57)

for name, res in results.items():
    acc  = res['accuracy']
    mf1  = res['report']['macro avg']['f1-score']
    wf1  = res['report']['weighted avg']['f1-score']
    flag = " ⭐" if name == best_name else ""
    print(f"{name:<25} {acc*100:>9.2f}% {mf1*100:>9.2f}% "
          f"{wf1*100:>11.2f}%{flag}")

print("-"*57)
print(f"\n🏆 Best classifier : {best_name} "
      f"({best_accuracy*100:.2f}%)")


# ============================================================
# 7G. DETAILED REPORT FOR BEST MODEL
# ============================================================

print(f"\n{'='*55}")
print(f"DETAILED REPORT — {best_name}")
print(f"{'='*55}")
print(classification_report(
    y_test,
    results[best_name]['y_pred'],
    target_names = le.classes_
))


# ============================================================
# 7H. CONFUSION MATRIX — Best Model
# ============================================================

print("⏳ Generating confusion matrix...")

cm  = confusion_matrix(y_test, results[best_name]['y_pred'])
fig, ax = plt.subplots(figsize=(14, 12))

disp = ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels   = le.classes_
)
disp.plot(ax=ax, cmap='Blues', colorbar=True)

ax.set_title(
    f'Confusion Matrix — {best_name}\n'
    f'(Test Accuracy: {best_accuracy*100:.2f}%)',
    fontsize=14
)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved → results/confusion_matrix.png")


# ============================================================
# 7I. UPDATED ACCURACY BAR CHART — All 5 Models
# ============================================================

print("⏳ Generating classifier comparison chart...")

names  = list(results.keys())
accs   = [results[n]['accuracy'] * 100 for n in names]
colors = ['steelblue', 'seagreen', 'darkorange',
          'mediumpurple', 'crimson']  # crimson for ensemble

fig2, ax2 = plt.subplots(figsize=(11, 5))
bars      = ax2.bar(names, accs, color=colors,
                    alpha=0.85, edgecolor='gray')

for bar, acc in zip(bars, accs):
    ax2.annotate(
        f'{acc:.2f}%',
        xy     = (bar.get_x() + bar.get_width()/2,
                  bar.get_height()),
        xytext = (0, 5), textcoords="offset points",
        ha='center', fontsize=10, fontweight='bold'
    )

ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title(
    'Resume Classifier — Individual vs Ensemble Voting\n'
    '(Trained on SBERT Embeddings | 80/20 Stratified Split)',
    fontsize=13
)
ax2.set_ylim(0, max(accs) * 1.2)
ax2.grid(axis='y', alpha=0.3)

# Highlight ensemble bar with a border
bars[-1].set_edgecolor('black')
bars[-1].set_linewidth(2)

plt.tight_layout()
plt.savefig("results/classifier_comparison.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved → results/classifier_comparison.png")


# ============================================================
# 7J. SAVE RESULTS
# ============================================================

# Save per-category accuracy for best model
cat_report = pd.DataFrame(
    results[best_name]['report']
).transpose().reset_index()
cat_report.columns = ['Category', 'Precision', 'Recall',
                      'F1', 'Support']
cat_report.to_csv("results/classifier_report.csv", index=False)

# Save summary
summary_df = pd.DataFrame([{
    'Model'      : name,
    'Accuracy'   : f"{res['accuracy']*100:.2f}%",
    'Macro_F1'   : f"{res['report']['macro avg']['f1-score']*100:.2f}%",
    'Weighted_F1': f"{res['report']['weighted avg']['f1-score']*100:.2f}%"
} for name, res in results.items()])

summary_df.to_csv("results/classifier_summary.csv", index=False)

print("\n✅ Results saved:")
print("   → results/classifier_report.csv")
print("   → results/classifier_summary.csv")
print("   → results/confusion_matrix.png")
print("   → results/classifier_comparison.png")


# ============================================================
# 7K. FINAL SUMMARY
# ============================================================

print("\n" + "="*55)
print("CLASSIFIER SUMMARY")
print("="*55)
print(f"✅ Features used      : SBERT embeddings (384 dim)")
print(f"✅ Training samples   : {len(X_train)}")
print(f"✅ Testing samples    : {len(X_test)}")
print(f"✅ Categories         : {len(le.classes_)}")
print(f"✅ Models trained     : {len(classifiers)}")
print(f"✅ Best model         : {best_name}")
print(f"✅ Best accuracy      : {best_accuracy*100:.2f}%")
print("="*55)
print("\n🎉 Step 7 Complete — Ready for Step 8: Metrics!")