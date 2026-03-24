# ============================================================
# STEP 2: DATA PREPROCESSING — Updated for New Datasets
# ============================================================

import pandas as pd
import re
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# ============================================================
# 2A. LOAD DATA
# ============================================================

resume_df = pd.read_csv("data/ResumeDataset.csv")
jd_df     = pd.read_csv("data/job_descriptions.csv")

print("="*55)
print("DATA LOADED")
print("="*55)
print(f"Resume shape      : {resume_df.shape}")
print(f"JD shape          : {jd_df.shape}")
print(f"Resume columns    : {resume_df.columns.tolist()}")
print(f"JD columns        : {jd_df.columns.tolist()}")
print(f"\nResume categories : {resume_df['Category'].nunique()}")
print(f"JD categories     : {jd_df['Category'].nunique()}")
print(f"JDs per category  :")
print(jd_df.groupby('Category')['JD_Number'].max().to_string())


# ============================================================
# 2B. FIX ENCODING
# ============================================================

def fix_encoding(text):
    if not isinstance(text, str):
        return ""
    try:
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text

resume_df['Resume']          = resume_df['Resume'].apply(fix_encoding)
jd_df['Job_Description']     = jd_df['Job_Description'].apply(fix_encoding)

print("\n✅ Encoding fixed")


# ============================================================
# 2C. ADD UNIQUE ID TO EACH RESUME ROW
# ============================================================

resume_df.insert(0, 'Resume_ID', range(1, len(resume_df) + 1))
print(f"✅ Resume IDs assigned (1 to {len(resume_df)})")


# ============================================================
# 2D. CLEANING FUNCTION
# Paper steps: lowercase → remove noise → tokenize
#              → stop words → lemmatize → stem
# ============================================================

stop_words = set(stopwords.words('english'))
stemmer    = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters, digits, punctuation
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]

    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Snowball stem (as per paper)
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# ============================================================
# 2E. APPLY CLEANING
# ============================================================

print("\n⏳ Cleaning resume texts...")
resume_df['Cleaned_Resume'] = resume_df['Resume'].apply(clean_text)

print("⏳ Cleaning job descriptions...")
jd_df['Cleaned_JD'] = jd_df['Job_Description'].apply(clean_text)

print("✅ Cleaning done")


# ============================================================
# 2F. DROP GENUINELY EMPTY ROWS ONLY
# No deduplication — keep all 2096 resumes
# ============================================================

before = len(resume_df)
resume_df.dropna(subset=['Cleaned_Resume'], inplace=True)
resume_df = resume_df[resume_df['Cleaned_Resume'].str.strip() != ""]
resume_df.reset_index(drop=True, inplace=True)
after = len(resume_df)

print(f"\n✅ Empty rows dropped  : {before - after}")
print(f"   Resumes remaining  : {after}")


# ============================================================
# 2G. SANITY CHECK — Before vs After
# ============================================================

print("\n" + "="*55)
print("BEFORE vs AFTER — Sample Resume")
print("="*55)
print("\n📄 ORIGINAL (first 300 chars):")
print(resume_df['Resume'].iloc[0][:300])
print("\n🧹 CLEANED (first 300 chars):")
print(resume_df['Cleaned_Resume'].iloc[0][:300])

print("\n" + "="*55)
print("BEFORE vs AFTER — Sample JD")
print("="*55)
print("\n📄 ORIGINAL (first 300 chars):")
print(jd_df['Job_Description'].iloc[0][:300])
print("\n🧹 CLEANED (first 300 chars):")
print(jd_df['Cleaned_JD'].iloc[0][:300])


# ============================================================
# 2H. CATEGORY DISTRIBUTION AFTER CLEANING
# ============================================================

print("\n✅ Resume count per category after cleaning:")
print(resume_df.groupby('Category').size()
      .sort_values(ascending=False).to_string())

print("\n✅ JD count per category after cleaning:")
print(jd_df.groupby('Category').size()
      .sort_values(ascending=False).to_string())


# ============================================================
# 2I. SAVE CLEANED DATA
# ============================================================

resume_df.to_csv("data/cleaned_resumes.csv", index=False)
jd_df.to_csv("data/cleaned_jds.csv",         index=False)

print("\n✅ Saved:")
print("   → data/cleaned_resumes.csv")
print("   → data/cleaned_jds.csv")


# ============================================================
# 2J. FINAL SUMMARY
# ============================================================

print("\n" + "="*55)
print("PREPROCESSING SUMMARY")
print("="*55)
print(f"✅ Total resumes kept     : {len(resume_df)}")
print(f"✅ Total JDs cleaned      : {len(jd_df)}")
print(f"✅ Resume categories      : {resume_df['Category'].nunique()}")
print(f"✅ JD categories          : {jd_df['Category'].nunique()}")
print(f"✅ Avg tokens per resume  : "
      f"{resume_df['Cleaned_Resume'].apply(lambda x: len(x.split())).mean():.0f} words")
print(f"✅ Avg tokens per JD      : "
      f"{jd_df['Cleaned_JD'].apply(lambda x: len(x.split())).mean():.0f} words")
print("="*55)
print("\n🎉 Step 2 Complete — Ready for Step 3: Embedding!")