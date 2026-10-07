"""Train the Real/Fake Job Posting classifier used by the Gradio app.

This preserves the preprocessing and model choices in the supplied notebook:
- title + company_profile + description + requirements + benefits
- HTML entity decoding, lowercasing, URL/HTML removal, non-letter removal
- English stopword removal + WordNet lemmatization
- TF-IDF: max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95
- train_test_split(test_size=0.2, random_state=10)
- LogisticRegression()
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
TARGET_COLUMN = "fraudulent"


def ensure_nltk_resources() -> bool:
    """Try to make the notebook's NLTK resources available.

    Returns True when the full NLTK backend is usable. The fallback keeps
    training and inference consistent in offline/container environments.
    """
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


def clean_text(text: object) -> str:
    text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_preprocessor():
    use_nltk = ensure_nltk_resources()
    if use_nltk:
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        def preprocess_text(text: str) -> str:
            words = text.split()
            words = [word for word in words if word not in stop_words]
            words = [lemmatizer.lemmatize(word) for word in words]
            return " ".join(words)

        return preprocess_text, "nltk_wordnet"

    # Offline fallback: deterministic and available with scikit-learn.
    stop_words = set(ENGLISH_STOP_WORDS)

    def preprocess_text(text: str) -> str:
        words = [word for word in text.split() if word not in stop_words]
        return " ".join(words)

    return preprocess_text, "sklearn_stopwords_no_lemmatization"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fake-job classifier.")
    parser.add_argument(
        "--data",
        default="fake_job_postings.csv",
        help="Path to fake_job_postings.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for the saved model bundle",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}\n"
            "Pass the correct CSV path with --data."
        )

    df = pd.read_csv(data_path)
    required_columns = TEXT_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing)
        )

    df[TEXT_COLUMNS] = df[TEXT_COLUMNS].fillna("")
    df["combined_text"] = df[TEXT_COLUMNS].agg(" ".join, axis=1)
    df["clean_text"] = df["combined_text"].apply(clean_text)

    preprocess_text, preprocessing_backend = build_preprocessor()
    df["processed_text"] = df["clean_text"].apply(preprocess_text)

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X = tfidf.fit_transform(df["processed_text"])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=10,
    )

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    bundle = {
        "tfidf": tfidf,
        "classifier": classifier,
        "text_columns": TEXT_COLUMNS,
        "target_column": TARGET_COLUMN,
        "preprocessing_backend": preprocessing_backend,
        "preprocessing": {
            "html_unescape": True,
            "lowercase": True,
            "remove_urls": True,
            "remove_html_tags": True,
            "letters_and_spaces_only": True,
            "english_stopwords": True,
            "wordnet_lemmatization": True,
        },
        "metrics": {
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": matrix,
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "vocabulary_size": int(X.shape[1]),
        },
    }

    artifact_path = output_dir / "model_bundle.joblib"
    joblib.dump(bundle, artifact_path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")

    print("\nTraining complete")
    print(f"Dataset:          {data_path.resolve()}")
    print(f"Rows:             {len(df):,}")
    print(f"Vocabulary size:  {X.shape[1]:,}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Artifacts:        {artifact_path.resolve()}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
