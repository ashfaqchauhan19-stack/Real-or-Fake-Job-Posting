"""Professional Gradio deployment for the Real/Fake Job Posting classifier.

Run after creating artifacts/model_bundle.joblib with train_model.py.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import gradio as gr
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

APP_TITLE = "JobGuard AI — Real or Fake Job Detector"
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT", "artifacts/model_bundle.joblib"))


def ensure_nltk_resources() -> bool:
    """Try to make the notebook's NLTK resources available."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    try:
        for resource_path, package in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(package, quiet=True)
                nltk.data.find(resource_path)
        return True
    except (LookupError, OSError):
        return False


def load_model_bundle():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{ARTIFACT_PATH}'. "
            "Run: python train_model.py --data /path/to/fake_job_postings.csv"
        )

    bundle = joblib.load(ARTIFACT_PATH)
    return bundle


BUNDLE = load_model_bundle()
TFIDF = BUNDLE["tfidf"]
CLASSIFIER = BUNDLE["classifier"]
TEXT_COLUMNS = BUNDLE.get(
    "text_columns",
    ["title", "company_profile", "description", "requirements", "benefits"],
)
METRICS = BUNDLE.get("metrics", {})
PREPROCESSING_BACKEND = BUNDLE.get("preprocessing_backend", "nltk_wordnet")

if PREPROCESSING_BACKEND == "nltk_wordnet" and ensure_nltk_resources():
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
else:
    PREPROCESSING_BACKEND = "sklearn_stopwords_no_lemmatization"
    STOP_WORDS = set(ENGLISH_STOP_WORDS)
    LEMMATIZER = None


def clean_text(text: object) -> str:
    """Match the notebook's cleaning logic."""
    text = str(text or "")
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    words = [word for word in text.split() if word not in STOP_WORDS]
    if LEMMATIZER is not None:
        words = [LEMMATIZER.lemmatize(word) for word in words]
    return " ".join(words)


def build_model_input(full_posting: str, title: str, company_profile: str,
                      description: str, requirements: str, benefits: str) -> str:
    """Use a pasted full posting when provided; otherwise mirror notebook fields."""
    if (full_posting or "").strip():
        return full_posting.strip()

    parts = [title, company_profile, description, requirements, benefits]
    return " ".join((part or "").strip() for part in parts if (part or "").strip())


def predict_job(full_posting, title, company_profile, description, requirements, benefits):
    raw_text = build_model_input(
        full_posting,
        title,
        company_profile,
        description,
        requirements,
        benefits,
    )

    if not raw_text:
        empty = "## Add a job posting to begin\n\nPaste the full posting above, or fill the structured fields below."
        return empty, {"REAL JOB POSTING": 0.0, "FAKE / FRAUDULENT JOB POSTING": 0.0}, ""

    cleaned = clean_text(raw_text)
    processed = preprocess_text(cleaned)
    vector = TFIDF.transform([processed])
    prediction = int(CLASSIFIER.predict(vector)[0])
    probability = CLASSIFIER.predict_proba(vector)[0]

    # LogisticRegression was trained on target 0 = real and 1 = fraudulent.
    class_to_probability = {int(cls): float(prob) for cls, prob in zip(CLASSIFIER.classes_, probability)}
    real_prob = class_to_probability.get(0, 0.0)
    fake_prob = class_to_probability.get(1, 0.0)

    if prediction == 1:
        verdict = "FAKE / FRAUDULENT JOB POSTING"
        confidence = fake_prob
        tone = "HIGH RISK"
        icon = "🚨"
        guidance = (
            "The model detected language patterns associated with fraudulent postings. "
            "Do not treat this as proof of fraud; verify the employer independently."
        )
    else:
        verdict = "REAL JOB POSTING"
        confidence = real_prob
        tone = "LOWER RISK"
        icon = "✅"
        guidance = (
            "The model found the posting more consistent with legitimate examples in the training data. "
            "A positive result still does not independently verify the employer."
        )

    verdict_md = f"""## {icon} {verdict}\n\n**Model confidence:** {confidence * 100:.2f}%  \n**Risk assessment:** {tone}\n\n{guidance}"""
    probabilities = {
        "REAL JOB POSTING": real_prob,
        "FAKE / FRAUDULENT JOB POSTING": fake_prob,
    }

    detail = (
        f"**Probability breakdown**\n\n"
        f"- Real: **{real_prob * 100:.2f}%**\n"
        f"- Fraudulent: **{fake_prob * 100:.2f}%**\n\n"
        "_Probabilities reflect the classifier output, not a guarantee of authenticity._"
    )
    return verdict_md, probabilities, detail


def clear_all():
    return "", "", "", "", "", "", "", {"REAL JOB POSTING": 0.0, "FAKE / FRAUDULENT JOB POSTING": 0.0}, ""


example_real = """Software Developer\n\nWe are looking for a talented software developer to join our team. The candidate should have experience with Python, SQL and machine learning. You will work with our engineering team to develop and maintain software applications. Bachelor's degree in computer science preferred."""

example_fake = """URGENT WORK FROM HOME OPPORTUNITY!!!\n\nEarn $5,000 per week working only 2 hours a day! No experience, education, or interview required. We are hiring immediately and there are unlimited vacancies. You must pay a $250 registration and training fee before starting. Send your bank account details, credit card information, and personal identification documents to complete registration. Contact us only through WhatsApp. Limited positions available! ACT NOW to secure your guaranteed income."""

CSS = """
:root {
  --app-radius: 18px;
}
body { background: #f5f7fb; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.hero {
  padding: 28px 30px;
  border-radius: 22px;
  background: linear-gradient(135deg, #101828 0%, #1d2939 55%, #344054 100%);
  color: white;
  margin-bottom: 18px;
}
.hero h1 { margin: 0 0 8px 0; font-size: 34px; line-height: 1.1; }
.hero p { margin: 0; opacity: .88; font-size: 15px; }
.card { border: 1px solid #e4e7ec; border-radius: var(--app-radius); background: white; padding: 16px; }
.section-title { font-size: 17px; font-weight: 700; margin: 4px 0 10px; }
.result-card { min-height: 235px; }
.disclaimer { font-size: 12px; color: #667085; line-height: 1.55; }
.metric { text-align: center; padding: 14px; border-radius: 14px; background: #f8fafc; border: 1px solid #eaecf0; }
footer { display: none !important; }
button { border-radius: 12px !important; }
"""

with gr.Blocks(title=APP_TITLE, fill_width=True) as demo:
    gr.HTML(
        """
        <div class='hero'>
          <h1>🛡️ JobGuard AI</h1>
          <p>Professional NLP screening for suspicious job postings — powered by the same TF‑IDF + Logistic Regression pipeline used in your notebook.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=7, elem_classes="card"):
            gr.Markdown("### 1 · Job posting", elem_classes="section-title")
            full_posting = gr.Textbox(
                label="Paste full job posting",
                placeholder="Paste the complete title, company information, description, requirements, benefits, salary, contact details, and other text here…",
                lines=14,
                max_lines=22,
            )
            gr.Markdown("**Or use structured fields** — these mirror the five text columns in the training notebook.")
            with gr.Accordion("Structured fields", open=False):
                title = gr.Textbox(label="Job title", lines=2)
                company_profile = gr.Textbox(label="Company profile", lines=4)
                description = gr.Textbox(label="Description", lines=6)
                requirements = gr.Textbox(label="Requirements", lines=5)
                benefits = gr.Textbox(label="Benefits", lines=4)

            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze posting", variant="primary", size="lg")
                clear_btn = gr.Button("Reset", variant="secondary", size="lg")

            with gr.Row():
                real_demo = gr.Button("Load realistic example", size="sm")
                fake_demo = gr.Button("Load suspicious example", size="sm")

        with gr.Column(scale=5, elem_classes="card result-card"):
            gr.Markdown("### 2 · Screening result", elem_classes="section-title")
            verdict = gr.Markdown("Enter a posting and click **Analyze posting**.")
            probabilities = gr.Label(
                label="Model probability",
                num_top_classes=2,
            )
            details = gr.Markdown("", elem_classes="disclaimer")

    with gr.Row():
        with gr.Column(elem_classes="card"):
            gr.Markdown("### About this deployment", elem_classes="section-title")
            gr.Markdown(
                "The deployment follows the notebook's model design: five job-related text fields are combined, cleaned, transformed with TF‑IDF (10,000 max features; 1–2 grams), and classified with Logistic Regression. When NLTK resources are available, stopword removal and WordNet lemmatization match the notebook; the bundled app also supports a deterministic offline fallback."
            )
        with gr.Column(elem_classes="card"):
            gr.Markdown("### Safety note", elem_classes="section-title")
            gr.Markdown(
                "This is a decision-support classifier, not an authenticity verifier. The notebook explicitly notes dataset dependence, false predictions, changing fraud patterns, missing information, class imbalance, and the inability to independently verify a company or job posting."
            )

    analyze_inputs = [full_posting, title, company_profile, description, requirements, benefits]
    analyze_outputs = [verdict, probabilities, details]
    analyze_btn.click(predict_job, inputs=analyze_inputs, outputs=analyze_outputs)

    clear_btn.click(
        clear_all,
        inputs=[],
        outputs=[full_posting, title, company_profile, description, requirements, benefits, verdict, probabilities, details],
    )

    real_demo.click(
        lambda: (example_real, "", "", "", "", ""),
        outputs=[full_posting, title, company_profile, description, requirements, benefits],
    )
    fake_demo.click(
        lambda: (example_fake, "", "", "", "", ""),
        outputs=[full_posting, title, company_profile, description, requirements, benefits],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=CSS,
        show_error=True,
    )
"""Professional Gradio deployment for the Real/Fake Job Posting classifier.

Run after creating artifacts/model_bundle.joblib with train_model.py.
"""


import html
import os
import re
from pathlib import Path

import gradio as gr
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

APP_TITLE = "JobGuard AI — Real or Fake Job Detector"
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT", "artifacts/model_bundle.joblib"))


def ensure_nltk_resources() -> bool:
    """Try to make the notebook's NLTK resources available."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    try:
        for resource_path, package in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(package, quiet=True)
                nltk.data.find(resource_path)
        return True
    except (LookupError, OSError):
        return False


def load_model_bundle():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{ARTIFACT_PATH}'. "
            "Run: python train_model.py --data /path/to/fake_job_postings.csv"
        )

    bundle = joblib.load(ARTIFACT_PATH)
    return bundle


BUNDLE = load_model_bundle()
TFIDF = BUNDLE["tfidf"]
CLASSIFIER = BUNDLE["classifier"]
TEXT_COLUMNS = BUNDLE.get(
    "text_columns",
    ["title", "company_profile", "description", "requirements", "benefits"],
)
METRICS = BUNDLE.get("metrics", {})
PREPROCESSING_BACKEND = BUNDLE.get("preprocessing_backend", "nltk_wordnet")

if PREPROCESSING_BACKEND == "nltk_wordnet" and ensure_nltk_resources():
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
else:
    PREPROCESSING_BACKEND = "sklearn_stopwords_no_lemmatization"
    STOP_WORDS = set(ENGLISH_STOP_WORDS)
    LEMMATIZER = None


def clean_text(text: object) -> str:
    """Match the notebook's cleaning logic."""
    text = str(text or "")
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    words = [word for word in text.split() if word not in STOP_WORDS]
    if LEMMATIZER is not None:
        words = [LEMMATIZER.lemmatize(word) for word in words]
    return " ".join(words)


def build_model_input(full_posting: str, title: str, company_profile: str,
                      description: str, requirements: str, benefits: str) -> str:
    """Use a pasted full posting when provided; otherwise mirror notebook fields."""
    if (full_posting or "").strip():
        return full_posting.strip()

    parts = [title, company_profile, description, requirements, benefits]
    return " ".join((part or "").strip() for part in parts if (part or "").strip())


def predict_job(full_posting, title, company_profile, description, requirements, benefits):
    raw_text = build_model_input(
        full_posting,
        title,
        company_profile,
        description,
        requirements,
        benefits,
    )

    if not raw_text:
        empty = "## Add a job posting to begin\n\nPaste the full posting above, or fill the structured fields below."
        return empty, {"REAL JOB POSTING": 0.0, "FAKE / FRAUDULENT JOB POSTING": 0.0}, ""

    cleaned = clean_text(raw_text)
    processed = preprocess_text(cleaned)
    vector = TFIDF.transform([processed])
    prediction = int(CLASSIFIER.predict(vector)[0])
    probability = CLASSIFIER.predict_proba(vector)[0]

    # LogisticRegression was trained on target 0 = real and 1 = fraudulent.
    class_to_probability = {int(cls): float(prob) for cls, prob in zip(CLASSIFIER.classes_, probability)}
    real_prob = class_to_probability.get(0, 0.0)
    fake_prob = class_to_probability.get(1, 0.0)

    if prediction == 1:
        verdict = "FAKE / FRAUDULENT JOB POSTING"
        confidence = fake_prob
        tone = "HIGH RISK"
        icon = "🚨"
        guidance = (
            "The model detected language patterns associated with fraudulent postings. "
            "Do not treat this as proof of fraud; verify the employer independently."
        )
    else:
        verdict = "REAL JOB POSTING"
        confidence = real_prob
        tone = "LOWER RISK"
        icon = "✅"
        guidance = (
            "The model found the posting more consistent with legitimate examples in the training data. "
            "A positive result still does not independently verify the employer."
        )

    verdict_md = f"""## {icon} {verdict}\n\n**Model confidence:** {confidence * 100:.2f}%  \n**Risk assessment:** {tone}\n\n{guidance}"""
    probabilities = {
        "REAL JOB POSTING": real_prob,
        "FAKE / FRAUDULENT JOB POSTING": fake_prob,
    }

    detail = (
        f"**Probability breakdown**\n\n"
        f"- Real: **{real_prob * 100:.2f}%**\n"
        f"- Fraudulent: **{fake_prob * 100:.2f}%**\n\n"
        "_Probabilities reflect the classifier output, not a guarantee of authenticity._"
    )
    return verdict_md, probabilities, detail


def clear_all():
    return "", "", "", "", "", "", "", {"REAL JOB POSTING": 0.0, "FAKE / FRAUDULENT JOB POSTING": 0.0}, ""


example_real = """Software Developer\n\nWe are looking for a talented software developer to join our team. The candidate should have experience with Python, SQL and machine learning. You will work with our engineering team to develop and maintain software applications. Bachelor's degree in computer science preferred."""

example_fake = """URGENT WORK FROM HOME OPPORTUNITY!!!\n\nEarn $5,000 per week working only 2 hours a day! No experience, education, or interview required. We are hiring immediately and there are unlimited vacancies. You must pay a $250 registration and training fee before starting. Send your bank account details, credit card information, and personal identification documents to complete registration. Contact us only through WhatsApp. Limited positions available! ACT NOW to secure your guaranteed income."""

CSS = """
:root {
  --app-radius: 18px;
}
body { background: #f5f7fb; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.hero {
  padding: 28px 30px;
  border-radius: 22px;
  background: linear-gradient(135deg, #101828 0%, #1d2939 55%, #344054 100%);
  color: white;
  margin-bottom: 18px;
}
.hero h1 { margin: 0 0 8px 0; font-size: 34px; line-height: 1.1; }
.hero p { margin: 0; opacity: .88; font-size: 15px; }
.card { border: 1px solid #e4e7ec; border-radius: var(--app-radius); background: white; padding: 16px; }
.section-title { font-size: 17px; font-weight: 700; margin: 4px 0 10px; }
.result-card { min-height: 235px; }
.disclaimer { font-size: 12px; color: #667085; line-height: 1.55; }
.metric { text-align: center; padding: 14px; border-radius: 14px; background: #f8fafc; border: 1px solid #eaecf0; }
footer { display: none !important; }
button { border-radius: 12px !important; }
"""

with gr.Blocks(title=APP_TITLE, fill_width=True) as demo:
    gr.HTML(
        """
        <div class='hero'>
          <h1>🛡️ JobGuard AI</h1>
          <p>Professional NLP screening for suspicious job postings — powered by the same TF‑IDF + Logistic Regression pipeline used in your notebook.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=7, elem_classes="card"):
            gr.Markdown("### 1 · Job posting", elem_classes="section-title")
            full_posting = gr.Textbox(
                label="Paste full job posting",
                placeholder="Paste the complete title, company information, description, requirements, benefits, salary, contact details, and other text here…",
                lines=14,
                max_lines=22,
            )
            gr.Markdown("**Or use structured fields** — these mirror the five text columns in the training notebook.")
            with gr.Accordion("Structured fields", open=False):
                title = gr.Textbox(label="Job title", lines=2)
                company_profile = gr.Textbox(label="Company profile", lines=4)
                description = gr.Textbox(label="Description", lines=6)
                requirements = gr.Textbox(label="Requirements", lines=5)
                benefits = gr.Textbox(label="Benefits", lines=4)

            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze posting", variant="primary", size="lg")
                clear_btn = gr.Button("Reset", variant="secondary", size="lg")

            with gr.Row():
                real_demo = gr.Button("Load realistic example", size="sm")
                fake_demo = gr.Button("Load suspicious example", size="sm")

        with gr.Column(scale=5, elem_classes="card result-card"):
            gr.Markdown("### 2 · Screening result", elem_classes="section-title")
            verdict = gr.Markdown("Enter a posting and click **Analyze posting**.")
            probabilities = gr.Label(
                label="Model probability",
                num_top_classes=2,
            )
            details = gr.Markdown("", elem_classes="disclaimer")

    with gr.Row():
        with gr.Column(elem_classes="card"):
            gr.Markdown("### About this deployment", elem_classes="section-title")
            gr.Markdown(
                "The deployment follows the notebook's model design: five job-related text fields are combined, cleaned, transformed with TF‑IDF (10,000 max features; 1–2 grams), and classified with Logistic Regression. When NLTK resources are available, stopword removal and WordNet lemmatization match the notebook; the bundled app also supports a deterministic offline fallback."
            )
        with gr.Column(elem_classes="card"):
            gr.Markdown("### Safety note", elem_classes="section-title")
            gr.Markdown(
                "This is a decision-support classifier, not an authenticity verifier. The notebook explicitly notes dataset dependence, false predictions, changing fraud patterns, missing information, class imbalance, and the inability to independently verify a company or job posting."
            )

    analyze_inputs = [full_posting, title, company_profile, description, requirements, benefits]
    analyze_outputs = [verdict, probabilities, details]
    analyze_btn.click(predict_job, inputs=analyze_inputs, outputs=analyze_outputs)

    clear_btn.click(
        clear_all,
        inputs=[],
        outputs=[full_posting, title, company_profile, description, requirements, benefits, verdict, probabilities, details],
    )

    real_demo.click(
        lambda: (example_real, "", "", "", "", ""),
        outputs=[full_posting, title, company_profile, description, requirements, benefits],
    )
    fake_demo.click(
        lambda: (example_fake, "", "", "", "", ""),
        outputs=[full_posting, title, company_profile, description, requirements, benefits],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=CSS,
        show_error=True,
    )
