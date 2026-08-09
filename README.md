# 🔍 Real or Fake Job Posting Prediction using NLP

An NLP and Machine Learning project that predicts whether a job posting is **Real** or **Fake/Fraudulent** based on the textual content of the job posting.

## 📌 Project Overview

Online job platforms can contain fraudulent job advertisements that may mislead job seekers or attempt to collect money and personal information.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to identify patterns in job postings and classify them as either:

- 🟢 Real Job Posting
- 🔴 Fake / Fraudulent Job Posting

The project is implemented using Python in Jupyter Notebook.

## 📊 Dataset

The dataset contains **17,880 job postings** with both textual and metadata information.

The target variable is:

- `fraudulent = 0` → Real job posting
- `fraudulent = 1` → Fake/Fraudulent job posting

The dataset contains **18 columns**, including:

- `job_id`
- `title`
- `location`
- `department`
- `salary_range`
- `company_profile`
- `description`
- `requirements`
- `benefits`
- `telecommuting`
- `has_company_logo`
- `has_questions`
- `employment_type`
- `required_experience`
- `required_education`
- `industry`
- `function`
- `fraudulent`

## 🧠 NLP Approach

The following text fields were combined for NLP processing:

- Job Title
- Company Profile
- Job Description
- Requirements
- Benefits

### Text Preprocessing

The project performs:

1. HTML entity conversion
2. Lowercase conversion
3. URL removal
4. HTML tag removal
5. Special character removal
6. Extra-space removal
7. Stop-word removal
8. Lemmatization

NLTK is used for stop-word removal and lemmatization.

## 🔢 Feature Extraction

The processed text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Configuration:

- Maximum features: `10,000`
- N-gram range: `(1, 2)`
- Minimum document frequency: `2`
- Maximum document frequency: `0.95`

This produces a feature matrix of:

`17,880 × 10,000`

## 🤖 Machine Learning Model

The project uses:

**Logistic Regression**

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

## 📈 Model Performance

The model achieved:

**Test Accuracy: 97.46%**

Classification results:

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Real (0) | 0.97 | 1.00 | 0.99 |
| Fake (1) | 1.00 | 0.43 | 0.60 |

The confusion matrix showed:

- True Real: 3416
- False Fake: 0
- False Real: 91
- True Fake: 69

Although the overall accuracy is high, the relatively lower recall for fraudulent postings shows that detecting fake jobs remains challenging.

## 🧪 New Job Prediction

The project also includes a function that accepts a new job description and predicts whether it is real or fraudulent.

Example result:

```text
Prediction: REAL JOB POSTING
Real probability: 97.81 %
Fake probability: 2.19 %
