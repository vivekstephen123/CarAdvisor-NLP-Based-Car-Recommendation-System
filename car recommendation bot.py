import streamlit as st
import pandas as pd
import numpy as np
import random
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CarAdvisor — Your Trusted Car Guru",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f5f6fa !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]  { display: none; }
[data-testid="stSidebar"]  { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.hero {
    background: #ffffff;
    border-bottom: 1px solid #e8eaf0;
    padding: 24px 48px 20px;
}
.hero-badge {
    display: inline-block;
    background: #eef0fb; border: 1px solid #c7cbf4;
    color: #4a55c8; font-size: 10px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 3px 12px; border-radius: 20px; margin-bottom: 10px;
}
.hero h1 {
    font-size: 22px !important; font-weight: 700 !important;
    color: #111827 !important; line-height: 1.2 !important;
    letter-spacing: -0.3px !important; margin-bottom: 6px !important;
}
.hero h1 span { color: #4a55c8; }
.hero-sub { font-size: 13px; color: #6b7280; max-width: 560px; line-height: 1.55; }
.hero-stats {
    display: flex; gap: 32px; margin-top: 14px;
    padding-top: 14px; border-top: 1px solid #f0f1f6;
}
.stat-item  { display: flex; flex-direction: column; gap: 2px; }
.stat-num   { font-size: 14px; font-weight: 700; color: #111827; }
.stat-label { font-size: 10px; color: #9ca3af; }

[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stVerticalBlock"] {
    background: #ffffff !important; padding: 20px 22px !important;
    border-right: 1px solid #e8eaf0; min-height: 80vh;
}
[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stVerticalBlock"] {
    background: #f5f6fa !important; padding: 20px 36px !important; min-height: 80vh;
}

.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: #9ca3af;
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid #f0f1f6;
}

[data-testid="stTextInput"] > div > div,
[data-testid="stNumberInput"] > div > div {
    background: #ffffff !important; border: 1px solid #d1d5db !important;
    border-radius: 8px !important; color: #000000 !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input { color: #000000 !important; font-weight: 500 !important; }
[data-testid="stTextInput"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div:focus-within {
    border-color: #4a55c8 !important; box-shadow: 0 0 0 3px rgba(74,85,200,0.08) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label { color: #374151 !important; font-size: 13px !important; font-weight: 500 !important; }

[data-testid="stButton"] > button {
    background: #4a55c8 !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    padding: 13px 24px !important; font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
    width: 100% !important; cursor: pointer !important;
    transition: background 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(74,85,200,0.22) !important; margin-top: 6px !important;
}
[data-testid="stButton"] > button:hover { background: #3b45b0 !important; }

/* Pills */
.intent-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #eef0fb; border: 1px solid #c7cbf4;
    color: #3b45b0; border-radius: 20px;
    padding: 5px 14px; font-size: 12px; font-weight: 500;
    margin-right: 6px; margin-bottom: 6px;
}
.fuel-pill        { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
.budget-pill      { background: #fefce8; border-color: #fde047; color: #713f12; }
.mileage-pill     { background: #f0fdf4; border-color: #86efac; color: #14532d; }
.performance-pill { background: #fff1f2; border-color: #fda4af; color: #881337; }
.safety-pill      { background: #eff6ff; border-color: #93c5fd; color: #1e3a8a; }
.family-pill      { background: #fdf4ff; border-color: #d8b4fe; color: #581c87; }
.comfort-pill     { background: #fff7ed; border-color: #fdba74; color: #7c2d12; }
.off-road-pill    { background: #f0fdf4; border-color: #4ade80; color: #14532d; }
.automatic-pill   { background: #f0f9ff; border-color: #7dd3fc; color: #0c4a6e; }
.first-car-pill   { background: #fdf2f8; border-color: #f0abfc; color: #701a75; }
.multi-badge {
    display: inline-block; background: #fff7ed; border: 1px solid #fed7aa;
    color: #9a3412; border-radius: 6px; padding: 8px 12px;
    font-size: 12px; margin-bottom: 14px; line-height: 1.5;
}

/* Intent weight bars */
.intent-breakdown { margin-bottom: 16px; }
.intent-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 7px; font-size: 12px;
}
.intent-name  { width: 90px; color: #374151; font-weight: 500; text-transform: capitalize; }
.intent-track { flex: 1; background: #e5e7eb; border-radius: 4px; height: 6px; overflow: hidden; }
.intent-fill  { height: 100%; border-radius: 4px; background: #4a55c8; }
.intent-pct   { width: 36px; text-align: right; color: #6b7280; font-weight: 500; }

/* Confidence bar */
.conf-track { background: #e5e7eb; border-radius: 6px; height: 5px; width: 100%; overflow: hidden; margin-top: 5px; }
.conf-fill-high { height:100%; border-radius:6px; background:#4a55c8; }
.conf-fill-low  { height:100%; border-radius:6px; background:#f59e0b; }

/* Results */
.results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.results-title  { font-size: 17px; font-weight: 700; color: #111827; }
.results-count  {
    font-size: 12px; color: #6b7280; background: #fff;
    border: 1px solid #e5e7eb; padding: 3px 12px; border-radius: 20px;
}

/* Car card */
.car-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 20px 22px; margin-bottom: 12px;
    position: relative; transition: box-shadow 0.18s, border-color 0.18s;
}
.car-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.07); border-color: #c7cbf4; }
.card-rank {
    position: absolute; top: 16px; right: 18px;
    font-size: 36px; font-weight: 700; color: #f0f1f6; line-height: 1; user-select: none;
}
.car-brand { font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
.car-name  { font-size: 17px; font-weight: 700; color: #111827; }
.price-label { font-size: 10px; color: #9ca3af; letter-spacing: 1px; text-transform: uppercase; text-align: right; }
.price-tag   { font-size: 16px; font-weight: 700; color: #4a55c8; text-align: right; }

.spec-box { background: #f9fafb; border: 1px solid #f0f1f6; border-radius: 8px; padding: 10px 12px; }
.spec-val { font-size: 14px; font-weight: 600; color: #111827; }
.spec-key { font-size: 10px; color: #9ca3af; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.4px; }

.score-row { display: flex; justify-content: space-between; font-size: 11px; color: #9ca3af; margin-top: 12px; margin-bottom: 4px; }
.score-track { background: #f0f1f6; border-radius: 4px; height: 4px; overflow: hidden; }
.score-fill  { height: 100%; border-radius: 4px; background: #4a55c8; }

.explanation-box {
    margin-top: 12px; padding: 10px 14px;
    background: #f5f6ff; border: 1px solid #e0e3f8;
    border-radius: 8px; font-size: 13px; color: #4b5563; line-height: 1.6;
}
.explanation-box strong { color: #3b45b0; }

/* Why ranked box */
.why-box {
    margin-top: 8px; padding: 8px 12px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 8px; font-size: 12px; color: #166534; line-height: 1.5;
}

.info-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 8px; padding: 12px 16px;
    font-size: 13px; color: #92400e; margin-bottom: 14px;
}

.empty-state { text-align: center; padding: 72px 32px; }
.empty-icon  { font-size: 40px; margin-bottom: 12px; }
.empty-title { font-size: 17px; font-weight: 600; color: #9ca3af; }
.empty-sub   { font-size: 13px; margin-top: 6px; color: #c4c9d4; }

.example-chip {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 9px 12px;
    font-size: 12px; color: #6b7280; line-height: 1.5; margin-bottom: 8px;
}

/* ── Model Diagnostics: dark grey text ── */
[data-testid="stExpander"] p,
[data-testid="stExpander"] span:not(.stExpanderToggleIcon):not([data-testid]),
[data-testid="stExpander"] div:not([data-testid]),
[data-testid="stExpander"] td,
[data-testid="stExpander"] th,
[data-testid="stExpander"] strong,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
[data-testid="stExpander"] [data-testid="stTable"] td,
[data-testid="stExpander"] [data-testid="stTable"] th {
    color: #374151 !important;
}
[data-testid="stExpander"] .section-label {
    color: #6b7280 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
LUXURY_BRANDS = ["BMW", "Audi", "Mercedes-Benz", "Land Rover", "Volvo", "Lexus", "Jaguar"]
FUEL_TYPES    = ["petrol", "diesel", "electric", "cng", "hybrid", "petrol hybrid"]
BODY_TYPES    = ["sedan", "hatchback", "suv", "mpv", "coupe suv", "pickup", "crossover"]
KNOWN_BRANDS  = [
    "maruti", "maruti suzuki", "suzuki", "hyundai", "tata", "mahindra", "kia",
    "toyota", "honda", "renault", "volkswagen", "vw", "skoda", "nissan",
    "ford", "jeep", "mg", "citroen", "byd", "bmw", "audi", "mercedes",
    "mercedes-benz", "land rover", "volvo", "lexus", "jaguar", "porsche",
    "datsun", "isuzu",
]
# Map query brand aliases → exact brand column values
BRAND_MAP = {
    "maruti": "Maruti Suzuki", "maruti suzuki": "Maruti Suzuki", "suzuki": "Maruti Suzuki",
    "vw": "Volkswagen", "volkswagen": "Volkswagen",
    "mercedes": "Mercedes-Benz", "mercedes-benz": "Mercedes-Benz",
    "land rover": "Land Rover",
}

# Keywords that strongly signal each intent — used for rule-based intent detection
# ─────────────────────────────────────────────
#  INTENT META  — icon + pill CSS class per intent
# ─────────────────────────────────────────────
INTENT_META = {
    "budget":      ("💰", "budget-pill"),
    "mileage":     ("⛽", "mileage-pill"),
    "performance": ("🏎️",  "performance-pill"),
    "safety":      ("🛡️",  "safety-pill"),
    "family":      ("👨\u200d👩\u200d👧", "family-pill"),
    "fuel":        ("🔋", "fuel-pill"),
    "comfort":     ("🛋️",  "comfort-pill"),
    "off_road":    ("🏔️",  "off-road-pill"),
    "automatic":   ("🕹️",  "automatic-pill"),
    "first_car":   ("🎓", "first-car-pill"),
}

INTENT_KEYWORDS = {
    "fuel":        ["petrol", "diesel", "electric", "cng", "hybrid", "ev", "fuel"],
    "mileage":     ["mileage", "kmpl", "fuel economy", "efficiency", "economical", "fuel efficient"],
    "performance": ["bhp", "power", "torque", "acceleration", "fast", "sporty", "performance", "speed"],
    "safety":      ["safety", "safe", "airbags", "ncap", "crash", "protection"],
    "budget":      ["cheap", "affordable", "budget", "price", "under", "lakhs", "lakh", "cost", "inexpensive"],
    "family":      ["family", "spacious", "kids", "children", "seating", "7 seat", "6 seat", "mpv", "suv"],
    "comfort":     ["comfortable", "comfort", "smooth ride", "plush", "suspension", "quiet", "nvh",
                    "sunroof", "leather", "legroom", "headroom", "ventilated", "long drive", "highway"],
    "off_road":    ["off road", "4x4", "4wd", "awd", "ground clearance", "terrain", "rough road",
                    "hill", "ladakh", "kachchi", "rugged", "mud", "pahad", "clearance"],
    "automatic":   ["automatic", "amt", "cvt", "dct", "no clutch", "without clutch", "auto",
                    "self shift", "clutchless", "bina clutch", "gear free"],
    "first_car":   ["first car", "first time", "beginner", "new driver", "pehli car", "starter car",
                    "easy to drive", "new buyer", "just got licence", "learn driving"],
}

# Scoring weights per intent — controls how much each factor contributes
INTENT_WEIGHTS = {
    "budget":      {"price": 0.70, "mileage": 0.20, "power": 0.05, "ncap": 0.05},
    "mileage":     {"price": 0.25, "mileage": 0.65, "power": 0.05, "ncap": 0.05},
    "performance": {"price": 0.20, "mileage": 0.05, "power": 0.70, "ncap": 0.05},
    "safety":      {"price": 0.20, "mileage": 0.10, "power": 0.05, "ncap": 0.65},
    "family":      {"price": 0.30, "mileage": 0.35, "power": 0.10, "ncap": 0.25},
    "fuel":        {"price": 0.35, "mileage": 0.50, "power": 0.10, "ncap": 0.05},
    # New intents
    "comfort":     {"price": 0.30, "mileage": 0.20, "power": 0.10, "ncap": 0.40},
    "off_road":    {"price": 0.25, "mileage": 0.10, "power": 0.55, "ncap": 0.10},
    "automatic":   {"price": 0.45, "mileage": 0.30, "power": 0.15, "ncap": 0.10},
    "first_car":   {"price": 0.55, "mileage": 0.25, "power": 0.05, "ncap": 0.15},
}


# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    import os as _os
    _base = _os.path.dirname(_os.path.abspath(__file__))
    cars = pd.read_csv(_os.path.join(_base, "indian_cars_synthetic_dataset (1).csv"))
    cars.columns = cars.columns.str.strip().str.lower()
    intent_df = pd.read_csv(_os.path.join(_base, "intent_data.csv"))
    intent_df["query"]  = intent_df["query"].str.lower()
    intent_df["intent"] = intent_df["intent"].str.lower()
    return cars, intent_df

def anchor_query(query, intent):
    if intent in INTENT_KEYWORDS:
        return f"{query} {random.choice(INTENT_KEYWORDS[intent])}"
    return query

# ─────────────────────────────────────────────
#  HINGLISH NORMALISATION
# ─────────────────────────────────────────────
HINGLISH_MAP = {
    "gadi": "car", "gaadi": "car", "gaddi": "car", "kaar": "car",
    "biwi": "family", "patni": "family", "bacche": "family",
    "bachche": "family", "parivar": "family", "ghar": "family",
    "sasta": "affordable", "sastaa": "affordable",
    "bijli": "electric", "tez": "fast",
    "suraksha": "safety", "average": "mileage",
    "kitna": "price", "kitne": "price",
    "zyada": "more", "kam": "less",
}

def normalise_hinglish(text: str) -> str:
    t = text.lower()
    for hi, en in HINGLISH_MAP.items():
        t = re.sub(r'\b' + re.escape(hi) + r'\b', en, t)
    return t


@st.cache_resource
def train_model():
    """
    Returns vectorizer, classifier, and cv_results dict containing:
      - fold_accs  : accuracy per fold
      - cv_mean    : mean CV accuracy
      - cv_std     : std across folds
      - cm         : confusion matrix (numpy array)
      - labels     : intent class labels
      - report     : per-class precision/recall/f1
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics         import (accuracy_score, confusion_matrix,
                                         classification_report)
    from sklearn.pipeline        import Pipeline

    _, intent_df = load_data()
    intent_df["query"] = intent_df.apply(
        lambda r: anchor_query(r["query"], r["intent"]), axis=1
    )
    intent_df["query"] = intent_df["query"].apply(normalise_hinglish)

    X = intent_df["query"].values
    y = intent_df["intent"].values

    # ── Upgraded model: word n-grams + char n-grams + LinearSVC ──
    feat = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), analyzer="word",    sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True)),
    ])
    pipe = Pipeline([
        ("feat", feat),
        ("norm", Normalizer()),
        ("clf",  CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=3000, class_weight="balanced"), cv=3
        )),
    ])

    # 5-fold stratified CV
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        fold_accs.append(accuracy_score(y[test_idx], preds))

    cv_mean = float(np.mean(fold_accs))
    cv_std  = float(np.std(fold_accs))

    # Cross-val predictions for confusion matrix (no data leakage)
    cv_preds = cross_val_predict(pipe, X, y, cv=skf)
    labels   = sorted(set(y))
    cm       = confusion_matrix(y, cv_preds, labels=labels)
    report   = classification_report(y, cv_preds, labels=labels, output_dict=True)

    # Final fit on ALL data for production inference
    pipe.fit(X, y)

    cv_results = {
        "fold_accs": fold_accs,
        "cv_mean":   cv_mean,
        "cv_std":    cv_std,
        "cm":        cm,
        "labels":    labels,
        "report":    report,
    }

    return pipe, pipe, cv_results

cars, intent_df = load_data()
vectorizer, clf_model, cv_results = train_model()


# ─────────────────────────────────────────────
#  MULTI-INTENT DETECTION
# ─────────────────────────────────────────────
def detect_intents(text: str, threshold: float = 0.10) -> dict:
    q = normalise_hinglish(text)

    probs = vectorizer.predict_proba([q])[0]
    ml_scores = {cls: float(p) for cls, p in zip(vectorizer.classes_, probs)}

    kw_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        kw_scores[intent] = hits / len(keywords)

    all_intents = set(ml_scores) | set(kw_scores)
    for i in all_intents:
        ml_scores.setdefault(i, 0.0)
        kw_scores.setdefault(i, 0.0)

    blended = {}
    for i in all_intents:
        blended[i] = 0.60 * ml_scores[i] + 0.40 * kw_scores[i]

    active = {i: w for i, w in blended.items() if w >= threshold}
    if not active:
        top = max(blended, key=blended.get)
        active = {top: 1.0}

    total = sum(active.values())
    return {i: round(w / total, 4) for i, w in active.items()}


def primary_intent(intent_weights: dict) -> str:
    return max(intent_weights, key=intent_weights.get)


# ─────────────────────────────────────────────
#  MULTI-INTENT SCORING
# ─────────────────────────────────────────────
def compute_blended_weights(intent_weights: dict) -> dict:
    blended = {"price": 0.0, "mileage": 0.0, "power": 0.0, "ncap": 0.0}
    default = {"price": 0.40, "mileage": 0.35, "power": 0.15, "ncap": 0.10}

    for intent, share in intent_weights.items():
        table = INTENT_WEIGHTS.get(intent, default)
        for factor, w in table.items():
            blended[factor] += share * w

    total = sum(blended.values())
    return {k: v / total for k, v in blended.items()}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def extract_fuel_type(query: str):
    q = query.lower()
    for f in FUEL_TYPES:
        if f in q:
            return f
    return None

def extract_body_type(query: str):
    q = query.lower()
    for b in sorted(BODY_TYPES, key=len, reverse=True):
        if b in q:
            return b
    return None

def extract_brand(query: str):
    import re as _re
    q = query.lower()
    for alias in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if " " in alias:
            if alias in q:
                return BRAND_MAP.get(alias, alias.title())
        else:
            if _re.search(r'\b' + _re.escape(alias) + r'\b', q):
                return BRAND_MAP.get(alias, alias.title())
    return None

def normalize(col: pd.Series) -> pd.Series:
    return (col - col.min()) / (col.max() - col.min() + 1e-6)


# ─────────────────────────────────────────────
#  BUDGET EXTRACTION FROM QUERY TEXT
# ─────────────────────────────────────────────
def extract_budget_from_query(text: str):
    import re as _re
    t     = text.lower().strip()
    LAKH  = 100_000
    CRORE = 10_000_000

    def _val(m, src, headroom=1.0):
        num = float(m.group(1))
        seg = src[max(0, m.start()-2):m.end()]
        unit = CRORE if any(x in seg for x in ["cr","crore"]) else LAKH
        return num * unit * headroom

    U = r"(?:lakh|lakhs|lac|lacs|l(?![a-z])|crore|crores|cr(?![a-z]))"
    N = r"[₹rs.\-]?\s*(\d+(?:\.\d+)?)\s*"

    patterns = [
        (rf"(?:under|within|below|upto|up to|less than|less|max|maximum|atmost|at most)\s*{N}{U}",
         lambda m, t: _val(m, t)),
        (rf"budget\s+(?:of\s+)?{N}{U}",
         lambda m, t: _val(m, t)),
        (rf"{N}{U}\s*(?:budget|range|mein|me(?!\w)|tak|se kam)",
         lambda m, t: _val(m, t)),
        (rf"(?:around|approximately|approx|roughly|about)\s*{N}{U}",
         lambda m, t: _val(m, t, headroom=1.10)),
        (rf"(?:^|\s){N}(?:lakh|lakhs|lac|lacs)(?:\s|$)",
         lambda m, t: _val(m, t)),
    ]

    for pattern, handler in patterns:
        m = _re.search(pattern, t)
        if m:
            try:
                val = handler(m, t)
                if 100_000 <= val <= 50_000_000:
                    return float(val)
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────
#  BUDGET ESTIMATION
# ─────────────────────────────────────────────
def resolve_budget(df: pd.DataFrame, intent_weights: dict, explicit_budget) -> float | None:
    prices = pd.to_numeric(df["price_inr"], errors="coerce").dropna()
    if prices.empty:
        return None

    if explicit_budget and explicit_budget > 0:
        return float(explicit_budget)

    has_budget = "budget" in intent_weights and intent_weights["budget"] >= 0.12
    has_family = "family" in intent_weights and intent_weights["family"] >= 0.12

    if has_budget and has_family:
        return float(prices.quantile(0.60))
    elif has_budget:
        return float(prices.quantile(0.35))
    return None


# ─────────────────────────────────────────────
#  FILTERING
# ─────────────────────────────────────────────
def rule_based_filter(df: pd.DataFrame, intent_weights: dict,
                      budget=None, fuel=None, body_type=None, brand=None) -> pd.DataFrame:
    filtered = df.copy()
    intents  = set(intent_weights.keys())

    filtered["price_inr"] = pd.to_numeric(filtered["price_inr"], errors="coerce")

    effective_budget = resolve_budget(df, intent_weights, budget)

    if effective_budget is not None:
        filtered = filtered[filtered["price_inr"] <= effective_budget]

    if "family" in intents and "seating" in filtered.columns:
        seating  = pd.to_numeric(filtered["seating"], errors="coerce")
        filtered = filtered[seating >= 7]

    if "safety" in intents and "global_ncap" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(filtered["global_ncap"], errors="coerce").notna()
        ]

    if "comfort" in intents and "transmission" in filtered.columns:
        auto = filtered[filtered["transmission"].str.lower() == "automatic"]
        if len(auto) >= 3:
            filtered = auto

    if "off_road" in intents:
        off = filtered[
            (pd.to_numeric(filtered["power_hp"], errors="coerce") >= 100) &
            (filtered["body_type"].str.lower().isin(["suv", "pickup", "crossover"]))
        ]
        if len(off) >= 3:
            filtered = off

    if "automatic" in intents and "transmission" in filtered.columns:
        auto = filtered[filtered["transmission"].str.lower() == "automatic"]
        if len(auto) >= 3:
            filtered = auto

    if "first_car" in intents:
        fc = filtered[
            filtered["body_type"].str.lower().isin(["hatchback", "sedan", "suv", "crossover"])
        ]
        if len(fc) >= 3:
            filtered = fc

    if fuel is not None and "fuel_type" in filtered.columns:
        filtered = filtered[filtered["fuel_type"].str.lower().str.contains(fuel, na=False)]

    if body_type is not None and "body_type" in filtered.columns:
        bt = filtered[filtered["body_type"].str.lower() == body_type.lower()]
        if len(bt) >= 1:
            filtered = bt

    if brand is not None and "brand" in filtered.columns:
        br = filtered[filtered["brand"].str.lower() == brand.lower()]
        if len(br) >= 1:
            filtered = br

    return filtered


# ─────────────────────────────────────────────
#  RANKING
# ─────────────────────────────────────────────
def rank_cars(df: pd.DataFrame, factor_weights: dict) -> pd.DataFrame:
    ranked = df.copy()
    for col in ["price_inr", "mileage_kmpl", "power_hp"]:
        ranked = ranked[pd.to_numeric(ranked[col], errors="coerce").notna()].copy()

    ranked["price_n"]   = 1 - normalize(ranked["price_inr"].astype(float))
    ranked["mileage_n"] = normalize(ranked["mileage_kmpl"].astype(float))
    ranked["power_n"]   = normalize(ranked["power_hp"].astype(float))
    if "global_ncap" in ranked.columns:
        ranked["ncap_n"] = normalize(
            pd.to_numeric(ranked["global_ncap"], errors="coerce").fillna(0)
        )
    else:
        ranked["ncap_n"] = 0.0

    ranked["score"] = (
        factor_weights["price"]   * ranked["price_n"]   +
        factor_weights["mileage"] * ranked["mileage_n"] +
        factor_weights["power"]   * ranked["power_n"]   +
        factor_weights["ncap"]    * ranked["ncap_n"]
    )
    return ranked.sort_values("score", ascending=False)


# ─────────────────────────────────────────────
#  EXPLAINABILITY
# ─────────────────────────────────────────────

def render_ncap_stars(rating) -> str:
    try:
        stars = int(float(rating))
    except (ValueError, TypeError):
        return "<span style='color:#94a3b8;font-size:11px;'>Not Rated</span>"
    filled = "★" * stars
    empty  = "☆" * (5 - stars)
    color  = "#f59e0b" if stars >= 4 else "#fb923c" if stars >= 2 else "#ef4444"
    return f"<span style='color:{color};font-size:14px;'>{filled}{empty}</span>"


def explain_ranking(row, intent_weights: dict) -> str:
    parts = []
    intents = sorted(intent_weights.items(), key=lambda x: -x[1])

    for intent, share in intents:
        if share < 0.15:
            continue
        if intent == "budget":
            parts.append(f"priced at ₹{int(row['price_inr']):,} (budget-friendly)")
        elif intent == "mileage":
            parts.append(f"{row['mileage_kmpl']} kmpl (high mileage)")
        elif intent == "performance":
            parts.append(f"{row['power_hp']} HP (strong performance)")
        elif intent == "safety":
            ncap = row.get("global_ncap", None)
            if pd.notna(ncap):
                parts.append(f"{int(float(ncap))}/5 NCAP safety rating")
        elif intent == "family":
            seating = row.get("seating", None)
            if pd.notna(seating):
                parts.append(f"{int(seating)}-seater (family-friendly)")
        elif intent == "fuel":
            parts.append(f"{str(row.get('fuel_type','—')).title()} fuel type")
        elif intent == "comfort":
            trans = row.get("transmission", "")
            ncap  = row.get("global_ncap", None)
            comfort_bits = []
            if str(trans).lower() == "automatic":
                comfort_bits.append("Automatic")
            if pd.notna(ncap):
                try:
                    comfort_bits.append(f"{int(float(ncap))}/5 safety")
                except: pass
            if comfort_bits:
                parts.append(" · ".join(comfort_bits) + " (comfort-oriented)")
        elif intent == "off_road":
            parts.append(f"{row['power_hp']} HP · {row.get('body_type','SUV')} (off-road capable)")
        elif intent == "automatic":
            trans = row.get("transmission", "")
            if str(trans).lower() == "automatic":
                parts.append("Automatic transmission (no clutch)")
        elif intent == "first_car":
            parts.append(f"₹{int(row['price_inr']):,} · {row['mileage_kmpl']} kmpl (great first car)")

    return " · ".join(parts) if parts else f"₹{int(row['price_inr']):,}, {row['mileage_kmpl']} kmpl"



# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI-Powered · Multi-Intent NLP · Real-time</div>
  <h1>Find Your Perfect <span>Indian Car</span></h1>
  <p class="hero-sub">
    Describe what you need in plain language — our hybrid NLP engine detects
    <em>multiple intents simultaneously</em>, filters the catalogue, and ranks
    the best match using blended scoring.
  </p>
  <div class="hero-stats">
    <div class="stat-item"><span class="stat-num">10</span><span class="stat-label">Intent Classes</span></div>
    <div class="stat-item"><span class="stat-num">Multi</span><span class="stat-label">Intent Support</span></div>
    <div class="stat-item"><span class="stat-num">SVC</span><span class="stat-label">char+word Model</span></div>
    <div class="stat-item"><span class="stat-num">100%</span><span class="stat-label">Explainable</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1.05, 2.3], gap="small")

with col_left:
    st.markdown('<div class="section-label">Query Input</div>', unsafe_allow_html=True)

    query  = st.text_input(
        "Describe your requirement",
        placeholder="e.g. safe affordable petrol family car",
        key="query_input",
    )

    # Auto-extract budget from query text
    _auto_budget = extract_budget_from_query(query) if query else None
    _auto_detected = _auto_budget is not None

    budget = st.number_input(
        "Budget (₹)" + (" 🟢 auto-detected" if _auto_detected else ""),
        min_value=0,
        step=50_000,
        format="%d",
        value=int(_auto_budget) if _auto_detected else 0,
        help="Auto-filled from your query. You can override this manually.",
    )
    run = st.button("Analyse & Recommend")

    # Show auto-detection notice inline under input
    if _auto_detected and budget == int(_auto_budget):
        st.markdown(
            f'''<div style="background:#dcfce7;border-left:3px solid #16a34a;padding:6px 10px;
            border-radius:4px;font-size:12px;color:#166534;margin-top:-8px;">
            💡 Budget <strong>₹{int(_auto_budget):,}</strong> extracted from your query
            </div>''',
            unsafe_allow_html=True
        )

    st.markdown("""
    <div style="margin-top:20px;">
      <div class="section-label">Example Queries (Multi-Intent)</div>
      <div class="example-chip">💰 "Safe affordable petrol family car under 12 lakhs"</div>
      <div class="example-chip">🔋 "High mileage CNG car under 8 lakhs"</div>
      <div class="example-chip">🏎️ "Powerful turbo SUV with sport mode"</div>
      <div class="example-chip">🛡️ "5 star NCAP safest car for family"</div>
      <div class="example-chip">🛋️ "Comfortable highway car with sunroof"</div>
      <div class="example-chip">🏔️ "4x4 off road SUV for Ladakh trip"</div>
      <div class="example-chip">🕹️ "Automatic car for city traffic under 10 lakhs"</div>
      <div class="example-chip">🎓 "Pehli car chahiye easy to drive under 8L"</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Diagnostics expander ──
    with st.expander("🔬 Model Diagnostics", expanded=False):

        # CV accuracy per fold
        st.markdown(
            '<div class="section-label" style="margin-top:0;">5-Fold Cross-Validation</div>',
            unsafe_allow_html=True,
        )
        fold_data = {
            "Fold": [f"Fold {i+1}" for i in range(len(cv_results["fold_accs"]))],
            "Accuracy": [f"{a:.1%}" for a in cv_results["fold_accs"]],
        }
        st.table(pd.DataFrame(fold_data).set_index("Fold"))
        st.markdown(
            f'<p style="color:#374151;font-size:14px;font-weight:600;margin:4px 0;">'
            f'Mean accuracy: {cv_results["cv_mean"]:.1%} ± {cv_results["cv_std"]:.1%}</p>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Confusion matrix as styled dataframe
        st.markdown(
            '<div class="section-label">Confusion Matrix (CV predictions)</div>',
            unsafe_allow_html=True,
        )
        labels = cv_results["labels"]
        cm_df  = pd.DataFrame(
            cv_results["cm"],
            index=[f"True: {l}" for l in labels],
            columns=[f"Pred: {l}" for l in labels],
        )
        st.dataframe(
            cm_df.style.background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )

        st.divider()

        # Per-class precision / recall / f1
        st.markdown(
            '<div class="section-label">Per-Class Metrics</div>',
            unsafe_allow_html=True,
        )
        report = cv_results["report"]
        rows   = []
        for lbl in labels:
            m = report.get(lbl, {})
            rows.append({
                "Intent":    lbl.title(),
                "Precision": f'{m.get("precision", 0):.2f}',
                "Recall":    f'{m.get("recall", 0):.2f}',
                "F1":        f'{m.get("f1-score", 0):.2f}',
                "Support":   int(m.get("support", 0)),
            })
        st.table(pd.DataFrame(rows).set_index("Intent"))

with col_right:

    if not run or not query.strip():
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🚗</div>
          <div class="empty-title">Ready to recommend</div>
          <div class="empty-sub">Enter a query on the left and click Analyse & Recommend</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Hinglish detection ──
        normalised_query = normalise_hinglish(query)
        hinglish_hit     = normalised_query != query.lower()

        # ── Multi-intent detection ──
        intent_weights  = detect_intents(query)
        factor_weights  = compute_blended_weights(intent_weights)
        fuel            = extract_fuel_type(query)
        body_type       = extract_body_type(query)
        brand           = extract_brand(query)
        top_intent      = primary_intent(intent_weights)
        is_multi        = len(intent_weights) > 1
        top_conf        = intent_weights[top_intent]

        # ── Hinglish notice ──
        if hinglish_hit:
            st.markdown(
                '<div class="info-box" style="background:#f0fdf4;border-color:#bbf7d0;color:#166534;">' +
                '🌐 <strong>Hinglish detected</strong> — query normalised to English before classification. ' +
                f'Interpreted as: <em>"{normalised_query}"</em></div>',
                unsafe_allow_html=True,
            )

        # ── Analysis section ──
        st.markdown('<div class="section-label">Intent Analysis</div>', unsafe_allow_html=True)

        # Show all detected intent pills
        pills_html = ""
        for intent, weight in sorted(intent_weights.items(), key=lambda x: -x[1]):
            icon, css = INTENT_META.get(intent, ("🎯", "intent-pill"))
            label     = intent.replace("_", " ").title()
            pills_html += (
                f'<span class="intent-pill {css}">' +
                f'{icon} {label} ({int(weight*100)}%)</span>'
            )
        if fuel:
            icon, css  = INTENT_META.get("fuel", ("🔋", "fuel-pill"))
            pills_html += f'<span class="intent-pill {css}">{icon} {fuel.title()}</span>'
        if body_type:
            pills_html += f'<span class="intent-pill" style="background:#f0f9ff;border-color:#bae6fd;color:#0369a1;">🚘 {body_type.title()}</span>'
        if brand:
            pills_html += f'<span class="intent-pill" style="background:#faf5ff;border-color:#e9d5ff;color:#6b21a8;">🏷️ {brand}</span>'
        st.markdown(pills_html, unsafe_allow_html=True)

        # Multi-intent notice
        if is_multi:
            intent_list = ", ".join(
                f"{i.title()} ({int(w*100)}%)"
                for i, w in sorted(intent_weights.items(), key=lambda x: -x[1])
            )
            st.markdown(
                f'<div class="multi-badge">⚡ <strong>Multi-intent detected</strong> — '
                f'scoring blended across: {intent_list}</div>',
                unsafe_allow_html=True,
            )

        # Intent weight breakdown bars
        INTENT_BAR_COLORS = {
            "budget": "#eab308", "mileage": "#22c55e", "performance": "#ef4444",
            "safety": "#3b82f6", "family": "#a855f7", "fuel": "#10b981",
            "comfort": "#f97316", "off_road": "#16a34a",
            "automatic": "#0ea5e9", "first_car": "#d946ef",
        }
        breakdown_html = '<div class="intent-breakdown">'
        for intent, weight in sorted(intent_weights.items(), key=lambda x: -x[1]):
            pct       = int(weight * 100)
            bar_color = INTENT_BAR_COLORS.get(intent, "#4a55c8")
            icon_str, _ = INTENT_META.get(intent, ("", ""))
            label_str = intent.replace("_", " ").title()
            breakdown_html += (
                '<div class="intent-row">'
                f'<span class="intent-name">{icon_str} {label_str}</span>'
                '<div class="intent-track">'
                f'<div class="intent-fill" style="width:{pct}%;background:{bar_color};"></div>'
                '</div>'
                f'<span class="intent-pct">{pct}%</span>'
                '</div>'
            )
        breakdown_html += '</div>'
        st.markdown(breakdown_html, unsafe_allow_html=True)

        # Low confidence fallback (based on top intent share)
        is_high = top_conf >= 0.50
        if not is_high:
            st.markdown("""
            <div class="info-box">
              ⚠️ Low confidence — try rephrasing your query with more specific keywords.
            </div>
            """, unsafe_allow_html=True)

        # ── Filter & rank ──
        filtered = rule_based_filter(
            cars, intent_weights,
            budget,
            fuel,
            body_type,
            brand,
        )

        # Show effective budget if auto-estimated from intent
        eff_budget = resolve_budget(cars, intent_weights, budget)
        if eff_budget and (not budget or budget == 0):
            st.markdown(
                f'<div style="font-size:12px;color:#6b7280;margin-bottom:12px;">'
                f'💡 No budget entered — auto-estimated cap of '
                f'<strong>₹{int(eff_budget):,}</strong> applied based on "affordable" intent.</div>',
                unsafe_allow_html=True,
            )
        ranked = rank_cars(filtered, factor_weights)
        top    = ranked.head(5).copy()

        st.markdown(
            '<div class="results-header">'
            '<div class="results-title">Top Recommendations</div>'
            f'<div class="results-count">{len(top)} of {len(filtered)} matches</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if top.empty:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🔍</div>
              <div class="empty-title">No matches found</div>
              <div class="empty-sub">Try increasing your budget or broadening your query</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            max_score = top["score"].max()

            for rank_num, (_, row) in enumerate(top.iterrows(), 1):
                score_pct         = int((row["score"] / max_score) * 100) if max_score > 0 else 0
                price_fmt         = "₹{:,}".format(int(row["price_inr"]))
                fuel_type_display = str(row.get("fuel_type", "—")).title()
                brand             = str(row["brand"])
                model             = str(row["model"])
                mileage           = str(row["mileage_kmpl"])
                power             = str(row["power_hp"])
                why               = explain_ranking(row, intent_weights)

                seating_val  = row.get("seating", None)
                seating_box  = ""
                if pd.notna(seating_val):
                    try:
                        s = int(float(seating_val))
                        seating_box = (
                            '<div class="spec-box">'
                            f'<div class="spec-val">{s} 💺</div>'
                            '<div class="spec-key">Seating</div>'
                            '</div>'
                        )
                    except Exception:
                        pass

                grid_cols = "repeat(3,1fr)"

                # NCAP box
                ncap_raw   = row.get("global_ncap", None)
                ncap_valid = (ncap_raw is not None) and pd.notna(ncap_raw)
                if ncap_valid:
                    stars    = render_ncap_stars(ncap_raw)
                    ncap_box = (
                        '<div class="spec-box">'
                        '<div class="spec-val">' + stars + '</div>'
                        '<div class="spec-key">NCAP Safety</div>'
                        '</div>'
                    )
                else:
                    ncap_box  = ""

                card_html = (
                    '<div class="car-card">'
                    '<div class="card-rank">#' + str(rank_num) + '</div>'

                    '<div style="display:flex;justify-content:space-between;'
                    'align-items:flex-start;margin-bottom:14px;">'
                    '<div>'
                    '<div class="car-brand">' + brand + '</div>'
                    '<div class="car-name">'  + model + '</div>'
                    '</div>'
                    '<div>'
                    '<div class="price-label">Price</div>'
                    '<div class="price-tag">' + price_fmt + '</div>'
                    '</div>'
                    '</div>'

                    '<div style="display:grid;grid-template-columns:' + grid_cols + ';gap:10px;margin-bottom:14px;">'
                    '<div class="spec-box"><div class="spec-val">' + mileage + '</div><div class="spec-key">Mileage (kmpl)</div></div>'
                    '<div class="spec-box"><div class="spec-val">' + power   + '</div><div class="spec-key">Power (HP)</div></div>'
                    '<div class="spec-box"><div class="spec-val">' + fuel_type_display + '</div><div class="spec-key">Fuel Type</div></div>'
                    + seating_box
                    + ncap_box
                    + '</div>'

                    '<div class="score-row"><span>Blended Match Score</span>'
                    '<span style="color:#4a55c8;font-weight:600;">' + str(score_pct) + '%</span></div>'
                    '<div class="score-track"><div class="score-fill" style="width:' + str(score_pct) + '%;"></div></div>'

                    '<div class="explanation-box">'
                    '<strong>' + brand + ' ' + model + '</strong> matches your requirements — ' + why + '.'
                    '</div>'
                    '</div>'
                )

                st.markdown(card_html, unsafe_allow_html=True)