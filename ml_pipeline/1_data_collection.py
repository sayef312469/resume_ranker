# ============================================================
# STEP 1: DATA COLLECTION — Final Version for New Datasets
# ============================================================

import pandas as pd
from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

# ============================================================
# 1A. LOAD SNEHA'S RESUME DATASET
# ============================================================

resume_df = pd.read_csv("data/Resume.csv")
resume_df = resume_df[['Category', 'Resume_str']].copy()
resume_df.rename(columns={'Resume_str': 'Resume'}, inplace=True)

print("="*50)
print("RESUME DATASET")
print("="*50)
print(f"Shape         : {resume_df.shape}")
print(f"Categories    : {resume_df['Category'].nunique()}")
print(f"Category list :\n{resume_df['Category'].value_counts()}")


# ============================================================
# 1B. UPDATED MAPPING — fixes all 10 missing categories
# ============================================================

# ============================================================
# 1B. UPDATED MAPPING — All 24 categories with broader titles
# ============================================================

category_to_jobtitle = {
    "HR": [
        "HR Manager", "Human Resources Manager", "HR Specialist",
        "Recruitment Manager", "HR Officer", "Talent Acquisition Manager",
        "HR Coordinator", "HR Business Partner", "HR Director",
        "Recruitment Specialist"
    ],
    "INFORMATION-TECHNOLOGY": [
        "Software Engineer", "Network Engineer", "IT Manager",
        "Systems Administrator", "IT Support Specialist",
        "Software Developer", "IT Analyst", "Systems Engineer",
        "IT Consultant", "Network Administrator"
    ],
    "DESIGNER": [
        "UX/UI Designer", "Graphic Designer", "UI Designer",
        "Visual Designer", "Creative Designer", "Web Designer",
        "Interaction Designer", "User Interface Designer",
        "User Experience Designer", "Product Designer"
    ],
    "SALES": [
        "Sales Representative", "Sales Manager", "Sales Executive",
        "Account Manager", "Sales Associate", "Sales Coordinator",
        "Business Sales Manager", "Regional Sales Manager",
        "Sales Analyst", "Sales Director"
    ],
    "FINANCE": [
        "Financial Advisor", "Finance Manager", "Financial Analyst",
        "Finance Officer", "CFO", "Investment Analyst",
        "Financial Controller", "Treasury Analyst",
        "Budget Analyst", "Finance Director"
    ],
    "BANKING": [
        "Loan Officer", "Credit Analyst", "Bank Teller",
        "Branch Manager", "Investment Banker", "Risk Analyst",
        "Banking Analyst", "Mortgage Advisor", "Wealth Manager",
        "Financial Services Manager"
    ],
    "HEALTHCARE": [
        "Registered Nurse", "Medical Assistant", "Healthcare Analyst",
        "Clinical Coordinator", "Health Administrator",
        "Medical Officer", "Patient Care Coordinator",
        "Clinical Manager", "Healthcare Manager",
        "Medical Coordinator"
    ],
    "ENGINEERING": [
        "Mechanical Engineer", "Civil Engineer", "Electrical Engineer",
        "Structural Engineer", "Engineering Manager",
        "Chemical Engineer", "Industrial Engineer",
        "Manufacturing Engineer", "Process Engineer",
        "Project Engineer"
    ],
    "ACCOUNTANT": [
        "Accountant", "Senior Accountant", "Tax Accountant",
        "Accounting Manager", "Junior Accountant",
        "Cost Accountant", "Staff Accountant",
        "Public Accountant", "Financial Accountant",
        "Management Accountant"
    ],
    "DIGITAL-MEDIA": [
        "Digital Marketing Specialist", "Social Media Manager",
        "Content Creator", "Digital Marketing Manager",
        "Social Media Analyst", "SEO Specialist",
        "Content Marketing Manager", "Digital Strategist",
        "Social Media Specialist", "Content Strategist"
    ],
    "CONSULTANT": [
        "Business Analyst", "Strategy Analyst",
        "Management Analyst", "Operations Analyst",
        "IT Business Analyst", "Process Analyst",
        "Business Systems Analyst", "Senior Business Analyst",
        "Data Analyst", "Research Analyst"
    ],
    "TEACHER": [
        "Teacher", "Instructor", "Academic Coordinator",
        "Education Coordinator", "Training Specialist",
        "Corporate Trainer", "Learning Specialist",
        "Training Manager", "Educational Consultant",
        "Curriculum Developer"
    ],
    "ADVOCATE": [
        "Legal Counsel", "Legal Advisor", "Compliance Officer",
        "Legal Assistant", "Contract Manager",
        "Paralegal", "Legal Analyst",
        "Corporate Counsel", "Legal Manager",
        "Regulatory Affairs Manager"
    ],
    "CHEF": [
        "Food Service Manager", "Catering Manager",
        "Restaurant Manager", "Kitchen Manager",
        "Food and Beverage Manager", "Catering Coordinator",
        "Food Production Manager", "Culinary Manager",
        "Food Safety Manager", "Banquet Manager"
    ],
    "FITNESS": [
        "Wellness Manager", "Sports Coach",
        "Health Coach", "Wellness Coordinator",
        "Recreation Manager", "Sports Manager",
        "Athletic Trainer", "Physical Education Teacher",
        "Health Promotion Manager", "Wellness Consultant"
    ],
    "ARTS": [
        "Art Director", "Creative Director",
        "Illustrator", "Animator",
        "Multimedia Designer", "Visual Artist",
        "Motion Graphics Designer", "Creative Manager",
        "Brand Designer", "Concept Artist"
    ],
    "BUSINESS-DEVELOPMENT": [
        "Business Development Manager",
        "Business Development Executive",
        "Partnership Manager", "Growth Manager",
        "Strategic Alliance Manager", "Market Development Manager",
        "New Business Manager", "Commercial Manager",
        "Corporate Development Manager", "Expansion Manager"
    ],
    "AVIATION": [
        "Airport Manager", "Ground Operations Manager",
        "Aviation Safety Officer", "Flight Operations Manager",
        "Airline Operations Manager", "Airport Operations Manager",
        "Air Traffic Manager", "Aviation Manager",
        "Flight Dispatcher", "Airport Coordinator"
    ],
    "AUTOMOBILE": [
        "Automotive Technician", "Fleet Manager",
        "Vehicle Inspector", "Service Advisor",
        "Auto Service Manager", "Automotive Service Manager",
        "Vehicle Fleet Manager", "Automotive Sales Manager",
        "Car Sales Manager", "Dealership Manager"
    ],
    "AGRICULTURE": [
        "Environmental Manager", "Sustainability Manager",
        "Supply Chain Manager", "Logistics Manager",
        "Quality Control Manager", "Operations Manager",
        "Production Manager", "Procurement Manager",
        "Inventory Manager", "Supply Manager"
    ],
    "APPAREL": [
        "Fashion Designer", "Merchandise Planner",
        "Retail Buyer", "Product Manager",
        "Brand Manager", "Marketing Manager",
        "Category Manager", "Retail Manager",
        "Visual Merchandiser", "Buying Manager"
    ],
    "CONSTRUCTION": [
        "Project Manager", "Construction Manager",
        "Site Manager", "Quantity Surveyor",
        "Construction Supervisor", "Civil Project Manager",
        "Building Manager", "Facilities Manager",
        "Infrastructure Manager", "Property Manager"
    ],
    "PUBLIC-RELATIONS": [
        "Communications Manager", "Marketing Manager",
        "Brand Manager", "Public Affairs Manager",
        "Corporate Communications Manager", "Media Manager",
        "Marketing Communications Manager", "Content Manager",
        "Digital Communications Manager", "Social Media Manager"
    ],
    "BPO": [
        "Customer Service Manager", "Call Center Manager",
        "Operations Manager", "Customer Support Manager",
        "BPO Manager", "Contact Center Manager",
        "Customer Experience Manager", "Service Delivery Manager",
        "Customer Operations Manager", "Support Operations Manager"
    ],
}


# ============================================================
# 1C. LOAD JD DATASET — Smart loading, no memory overload
# ============================================================

print("\n⏳ Loading JD dataset in chunks to save memory...")

# All job titles we're looking for across all categories
all_target_titles = [
    title
    for titles in category_to_jobtitle.values()
    for title in titles
]

# Read in chunks of 50,000 rows at a time
# Only keep rows where Job Title matches our targets
# This way we never load 1.6M rows into memory at once

chunk_size  = 50000
kept_chunks = []

for i, chunk in enumerate(pd.read_csv(
    "data/job_descriptions.csv",
    usecols=['Job Title', 'Job Description', 'skills', 'Responsibilities'],
    chunksize=chunk_size
)):
    filtered = chunk[chunk['Job Title'].isin(all_target_titles)]
    if len(filtered) > 0:
        kept_chunks.append(filtered)

    # Progress update every 500k rows
    if (i + 1) % 10 == 0:
        print(f"   Scanned {(i+1)*chunk_size:,} rows...")

jd_raw = pd.concat(kept_chunks).reset_index(drop=True)

print(f"✅ Relevant JD rows kept : {len(jd_raw)} out of 1,615,940")
print(f"   Memory saved         : ~97% reduction 🎉")

# ============================================================
# 1D. BUILD 5 SEPARATE JDs PER CATEGORY = 70 JDs TOTAL
# ============================================================

print("\n⏳ Extracting 10 JDs per category...")

jd_records = []

for category, job_titles in category_to_jobtitle.items():

    mask   = jd_raw['Job Title'].isin(job_titles)
    subset = jd_raw[mask].dropna(subset=['Job Description', 'skills'])

    if len(subset) == 0:
        print(f"   ⚠️  No match found for : {category}")
        continue

    # Take top 10 UNIQUE JDs as separate rows
    top10 = subset.drop_duplicates(subset=['Job Description']).head(10)

    for i, (_, row) in enumerate(top10.iterrows(), 1):
        jd_records.append({
            "Category"       : category,
            "Job_Title"      : row['Job Title'],
            "Job_Description": (
                f"Job Title: {row['Job Title']}. "
                f"Description: {row['Job Description']} "
                f"Skills: {row['skills']} "
                f"Responsibilities: {row['Responsibilities']}"
            ),
            "JD_Number"      : i   # 1 to 10 within each category
        })

    print(f"   ✅ {category:<25} → {len(top10)} JDs extracted")

# Build final JD dataframe
jd_df = pd.DataFrame(jd_records)

print(f"\n✅ Total JDs created    : {len(jd_df)}")
print(f"✅ Categories covered  : {jd_df['Category'].nunique()}")
print(f"\nJD distribution per category:")
print(jd_df.groupby('Category')['JD_Number'].max().to_string())

print(f"\nSample JD row:")
print(jd_df.iloc[0][['Category', 'Job_Title', 'JD_Number']])
print(f"Description (first 300 chars):")
print(jd_df['Job_Description'].iloc[0][:300])


# ============================================================
# 1E. BUILD FINAL JD DATAFRAME
# ============================================================

jd_df = pd.DataFrame(jd_records)

print(f"\n✅ JDs extracted for {len(jd_df)} out of {len(category_to_jobtitle)} categories")
print("\nJD Category list:")
print(jd_df['Category'].tolist())

print("\nSample JD (first 300 chars):")
print(jd_df['Job_Description'].iloc[0][:300])


# ============================================================
# 1F. MAKE SURE BOTH DATASETS HAVE MATCHING CATEGORIES
# Drop resume rows whose category has no matching JD
# ============================================================

valid_categories = jd_df['Category'].tolist()
resume_df        = resume_df[resume_df['Category'].isin(valid_categories)]
resume_df        = resume_df.reset_index(drop=True)

print(f"\n✅ Resumes after category alignment : {len(resume_df)}")
print(f"✅ Categories matched               : {resume_df['Category'].nunique()}")


# ============================================================
# 1G. LOAD STS BENCHMARK
# ============================================================

print("\n⏳ Loading STS Benchmark...")
sts_dataset = load_dataset("sentence-transformers/stsb")
sts_test    = pd.DataFrame(sts_dataset['test'])
sts_test.to_csv("data/sts_test.csv", index=False)
print(f"✅ STS loaded : {sts_test.shape[0]} pairs")


# ============================================================
# 1H. SAVE EVERYTHING
# ============================================================

resume_df.to_csv("data/ResumeDataset.csv", index=False)
jd_df.to_csv("data/job_descriptions.csv",  index=False)

print("\n✅ Files saved:")
print("   → data/ResumeDataset.csv")
print("   → data/job_descriptions.csv")
print("   → data/sts_test.csv")


# ============================================================
# 1I. FINAL SUMMARY
# ============================================================

print("\n" + "="*50)
print("DATA COLLECTION SUMMARY")
print("="*50)
print(f"✅ Resumes          : {len(resume_df)} rows")
print(f"✅ Categories       : {resume_df['Category'].nunique()}")
print(f"✅ JDs created      : {len(jd_df)} (one per category)")
print(f"✅ STS Benchmark    : {sts_test.shape[0]} sentence pairs")
print(f"✅ Avg resume length: {resume_df['Resume'].str.len().mean():.0f} chars")
print("="*50)
print("\n🎉 Step 1 Complete — Ready for Step 2: Preprocessing!")