import streamlit as st
import torch
import numpy as np
import re
import os
from pathlib import Path
from gensim.models import Word2Vec
from transformers import pipeline
import pandas as pd
from collections import Counter

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="VoxPop | Crisis Intelligence",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Minimal UI styling (does not change model logic)
# =============================================================================
st.markdown(
    """
    <style>
      /* Layout */
      .block-container { max-width: 1180px; padding-top: 2.0rem; padding-bottom: 3rem; }
      h1, h2, h3 { letter-spacing: -0.02em; }

      /* Tabs */
      button[role="tab"] { padding: 0.55rem 0.95rem; border-radius: 999px; }

      /* Inputs */
      .stTextArea textarea, .stTextInput input { border-radius: 12px !important; }

      /* Buttons */
      .stButton > button {
        border-radius: 12px;
        padding: 0.55rem 0.95rem;
        border: 1px solid rgba(128,128,128,0.35);
      }

      /* Cards */
      .vp-card {
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.28);
        background: rgba(128,128,128,0.07);
      }
      .vp-muted { opacity: 0.72; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='font-size:2.2rem; font-weight:700;'>VoxPop Crisis Intelligence</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='vp-muted' style='margin-top:0.2rem; margin-bottom:1.6rem;'>"
    "Analyze text sentiment, detect crisis signals, summarize high-volume complaints, and extract named entities."
    "</div>",
    unsafe_allow_html=True
)

# =============================================================================
# Path helpers (FIX: robust model paths)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

def _is_git_lfs_pointer(p: Path) -> bool:
    """Detects if a file is a Git LFS pointer text file instead of the real binary."""
    try:
        if not p.exists() or p.is_dir():
            return False
        with open(p, "rb") as f:
            head = f.read(200)
        return b"git-lfs.github.com/spec/v1" in head
    except Exception:
        return False

def _find_model_file(filename: str) -> Path | None:
    """
    Search common locations for model files:
    - repo root (your GitHub screenshot)
    - models/
    - jupyter notes/ (your old local path)
    - env override (MODEL_DIR)
    """
    candidates = []

    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / filename)

    candidates += [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "jupyter notes" / filename,
        Path.cwd() / filename,
        Path.cwd() / "models" / filename,
        Path.cwd() / "jupyter notes" / filename,
    ]

    for p in candidates:
        if p.exists():
            return p

    return None

# =============================================================================
# Model Components
# =============================================================================

class BetterBiLSTM2(torch.nn.Module):
    """BiLSTM model for anger/sentiment scoring."""
    def __init__(self, embedding_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_cat = self.dropout(hidden_cat)
        logits = self.fc(hidden_cat).squeeze(-1)
        return logits


@st.cache_resource
def load_models():
    """Load BiLSTM2 model, Word2Vec, BART summarizer, and BERT NER."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"Runtime device: {device}")
    st.caption(f"App dir: {BASE_DIR}")
    st.caption(f"Working dir: {Path.cwd()}")

    # -------------------------
    # Locate model files
    # -------------------------
    w2v_path = _find_model_file("word2vec.model")
    bilstm_path = _find_model_file("bilstm_anger_model.pt")

    # -------------------------
    # Load Word2Vec
    # -------------------------
    w2v_model = None
    try:
        if w2v_path is None:
            raise FileNotFoundError("word2vec.model not found in repo root / models/ / jupyter notes/")

        if _is_git_lfs_pointer(w2v_path):
            raise RuntimeError(
                f"{w2v_path} looks like a Git LFS pointer (not the real binary). "
                "If deploying on Streamlit Cloud, you must ensure the real LFS file is available "
                "(e.g., use GitHub Releases or download at startup)."
            )

        w2v_model = Word2Vec.load(str(w2v_path))
        st.success(f"Word2Vec model loaded: {w2v_path}")
    except Exception as e:
        st.error(f"Word2Vec load error: {e}")
        w2v_model = None

    # -------------------------
    # Load BiLSTM2
    # -------------------------
    model_bilstm2 = None
    try:
        if bilstm_path is None:
            raise FileNotFoundError("bilstm_anger_model.pt not found in repo root / models/ / jupyter notes/")

        if _is_git_lfs_pointer(bilstm_path):
            raise RuntimeError(
                f"{bilstm_path} looks like a Git LFS pointer (not the real binary). "
                "If deploying on Streamlit Cloud, you must ensure the real LFS file is available."
            )

        model_bilstm2 = BetterBiLSTM2(
            embedding_dim=100,
            hidden_dim=256,
            output_dim=1,
            num_layers=2,
            dropout=0.3
        ).to(device)

        obj = torch.load(str(bilstm_path), map_location=device)

        # supports either state_dict OR full model save
        if isinstance(obj, dict):
            model_bilstm2.load_state_dict(obj)
        else:
            model_bilstm2 = obj.to(device)

        model_bilstm2.eval()
        st.success(f"BiLSTM2 model loaded: {bilstm_path}")
    except Exception as e:
        st.error(f"BiLSTM2 load error: {e}")
        model_bilstm2 = None

    # -------------------------
    # Load BART summarizer (FIX: use text2text-generation instead of summarization)
    # -------------------------
    summarizer = None
    try:
        summarizer = pipeline(
            "text2text-generation",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        st.success("BART summarizer loaded (text2text-generation)")
    except Exception as e:
        st.error(f"BART summarizer error: {e}")
        summarizer = None

    # -------------------------
    # Load BERT NER (CPU to avoid CUDA crash)
    # -------------------------
    ner_model = None
    try:
        ner_model = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1
        )
        st.success("BERT NER model loaded")
    except Exception as e:
        st.error(f"NER model error: {e}")
        ner_model = None

    return model_bilstm2, w2v_model, summarizer, ner_model, device


# =============================================================================
# Text Processing Functions
# =============================================================================

def basic_tokenize(text: str):
    """Basic tokenization matching training preprocessing."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def text_to_w2v_array(text, w2v, max_len, embedding_dim):
    """Convert text to Word2Vec array."""
    tokens = basic_tokenize(text)
    kv = w2v.wv if hasattr(w2v, "wv") else w2v

    vectors = []
    for tok in tokens[:max_len]:
        if tok in kv:
            vectors.append(kv[tok])
        else:
            vectors.append(np.zeros(embedding_dim, dtype=np.float32))

    while len(vectors) < max_len:
        vectors.append(np.zeros(embedding_dim, dtype=np.float32))

    return np.stack(vectors).astype(np.float32)


@torch.no_grad()
def predict_anger_score(text, model, w2v, max_len, embedding_dim, device):
    """Predict anger/sentiment score for a single text."""
    model.eval()
    arr = text_to_w2v_array(text, w2v, max_len, embedding_dim)
    x = torch.tensor(arr).unsqueeze(0).to(device)
    logits = model(x).view(-1)
    score = torch.sigmoid(logits).item()
    return score


def classify_sentiment(score: float) -> str:
    """Classify sentiment based on score."""
    if score < 0.10:
        return " Very Negative"
    elif score < 0.30:
        return " Negative"
    elif score < 0.55:
        return " Neutral"
    elif score < 0.85:
        return " Positive"
    else:
        return " Very Positive"


def chunk_by_words(text, max_words=350):
    """Split text into chunks by word count."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


@st.cache_data(show_spinner=False)
def generate_crisis_report(texts, _summarizer):
    """
    Generate a 3-sentence crisis report from up to 1,000 reviews.

    Returns:
        (crisis_report: str, sample_len: int, level1_chunks: int, level2_chunks: int)
    """
    if _summarizer is None:
        return ("Summarizer is not available.", 0, 0, 0)

    if not texts:
        return ("No reviews to analyze.", 0, 0, 0)

    clean_texts = [str(t).strip().replace("\n", " ") for t in texts if str(t).strip()]
    if not clean_texts:
        return ("No valid text data to summarize.", 0, 0, 0)

    sample_len = len(clean_texts)
    big_doc = "\n".join([f"- {t}" for t in clean_texts])

    tokenizer = _summarizer.tokenizer
    max_model_tokens = getattr(_summarizer.model.config, "max_position_embeddings", 1024)
    chunk_tokens = min(900, max_model_tokens - 100)

    def tokenize_to_ids(text):
        encoded = tokenizer(text, return_tensors="pt", truncation=False)
        return encoded["input_ids"][0]

    def ids_to_text(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    def split_ids(ids, size):
        return [ids[i:i + size] for i in range(0, len(ids), size)]

    def summarize_text(text, max_len=250, min_len=150):
        out = _summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        if not out:
            return ""
        # FIX: summarization pipeline returns summary_text; text2text-generation returns generated_text
        d = out[0]
        return d.get("summary_text") or d.get("generated_text") or ""

    # Level 1
    ids = tokenize_to_ids(big_doc)
    level1 = split_ids(ids, chunk_tokens)
    level1_chunks = len(level1)

    chunk_summaries = []
    for ch_ids in level1:
        ch_text = ids_to_text(ch_ids)
        if ch_text.strip():
            chunk_summaries.append(summarize_text(ch_text, max_len=120, min_len=40))

    if not chunk_summaries:
        return ("Crisis report generation failed. Please try with fewer / cleaner reviews.", sample_len, level1_chunks, 0)

    merged = " ".join(chunk_summaries)

    # Level 2
    ids2 = tokenize_to_ids(merged)
    level2 = split_ids(ids2, chunk_tokens)
    level2_chunks = len(level2)

    mid_parts = []
    for ch_ids in level2:
        ch_text = ids_to_text(ch_ids)
        if ch_text.strip():
            mid_parts.append(summarize_text(ch_text, max_len=110, min_len=35))

    if not mid_parts:
        return ("Crisis report generation failed at merge stage. Try fewer reviews.", sample_len, level1_chunks, level2_chunks)

    merged2 = " ".join(mid_parts)

    # Final (3-sentence-ish)
    crisis_report = summarize_text(merged2, max_len=90, min_len=40)
    return (crisis_report, sample_len, level1_chunks, level2_chunks)


# =============================================================================
# NER Helpers (fixes broken wordpieces like "Mu", "Ch", etc.)
# =============================================================================

STOP_ENTITY_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "to", "of", "in", "on", "for", "with", "without", "from",
    "this", "that", "these", "those", "it", "its", "i", "we", "you", "he", "she", "they", "my", "your", "our", "their",
    "also", "see", "here", "there", "was", "were", "is", "are", "be", "been", "as", "at", "by"
}

def _merge_wordpieces(word: str) -> str:
    parts = str(word).strip().split()
    if not parts:
        return ""
    out = ""
    for p in parts:
        if p.startswith("##"):
            out += p[2:]
        else:
            out += (" " if out else "") + p
    return re.sub(r"\s+", " ", out).strip()

def _expand_to_word(text: str, start: int, end: int) -> str:
    if start is None or end is None:
        return ""
    n = len(text)
    start = max(0, min(start, n))
    end = max(0, min(end, n))

    l = start
    while l > 0 and (text[l - 1].isalnum() or text[l - 1] in {"_", "-"}):
        l -= 1

    r = end
    while r < n and (text[r].isalnum() or text[r] in {"_", "-"}):
        r += 1

    return text[l:r]

def _clean_entity(s: str) -> str:
    s = str(s).replace("\n", " ")
    s = re.sub(r"^[#@]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" ,.;:!?()[]{}\"'`")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_top_entities(reviews, ner_model, top_k=10):
    if ner_model is None or not reviews:
        return []

    all_entities = []
    for review in reviews:
        text_in = str(review)
        if not text_in.strip():
            continue

        try:
            ents = ner_model(text_in)
        except Exception:
            continue

        spans = []
        for ent in ents:
            group = ent.get("entity_group") or ent.get("entity")
            if group not in {"ORG", "PER", "LOC", "MISC"}:
                continue

            start = ent.get("start")
            end = ent.get("end")

            if isinstance(start, int) and isinstance(end, int):
                raw = _expand_to_word(text_in, start, end)
            else:
                raw = ent.get("word", "")

            raw = _merge_wordpieces(raw)
            raw = _clean_entity(raw)

            if raw:
                spans.append({"start": start if isinstance(start, int) else None,
                              "end": end if isinstance(end, int) else None,
                              "group": group,
                              "text": raw})

        if not spans:
            continue

        if all(s["start"] is not None and s["end"] is not None for s in spans):
            spans.sort(key=lambda x: x["start"])
            merged = []
            for s in spans:
                if not merged:
                    merged.append(s)
                    continue
                prev = merged[-1]
                gap = s["start"] - prev["end"]
                delim = text_in[prev["end"]:s["start"]] if gap >= 0 else ""
                if (s["group"] == prev["group"] and 0 <= gap <= 2 and re.fullmatch(r"[\s\-\/&]+", delim or " ")):
                    prev["end"] = s["end"]
                    prev["text"] = _clean_entity(prev["text"] + (delim or " ") + s["text"])
                else:
                    merged.append(s)
            candidates = [m["text"] for m in merged]
        else:
            candidates = [s["text"] for s in spans]

        for cand in candidates:
            c = _clean_entity(cand)
            if not c:
                continue
            cl = c.lower()
            if cl in STOP_ENTITY_WORDS:
                continue
            if len(c) < 4 and not c.isupper():
                continue
            if not re.search(r"[A-Za-z]", c):
                continue
            all_entities.append(c)

    if not all_entities:
        return []
    return Counter(all_entities).most_common(top_k)


# =============================================================================
# Brand Assistant Helpers
# =============================================================================
CATEGORY_KEYWORDS = {
    "battery / charging": ["battery", "drain", "drains", "charging", "charge", "charged", "case", "hours", "power"],
    "disconnect / bluetooth": ["disconnect", "disconnected", "drops", "dropping", "connection", "bluetooth", "pair", "pairing"],
    "app / firmware": ["app", "crash", "crashes", "update", "firmware", "login", "log in", "logs", "settings", "re-pair", "repair"],
    "mic / call quality": ["mic", "microphone", "muffled", "audio", "voice", "call", "calls", "zoom", "meet"],
    "noise cancellation / static": ["anc", "noise cancellation", "noise-cancellation", "hiss", "static", "transparency"],
    "support / refund": ["support", "customer support", "service", "refund", "replacement", "warranty", "return", "policy"],
    "delivery / shipping": ["delivery", "shipping", "late", "delayed", "tracking", "courier"],
    "comfort / fit": ["uncomfortable", "pain", "painful", "fit", "ear tips", "eartips"],
    "controls": ["touch", "controls", "pause", "skips", "skip", "sensitive"],
}

def _clean_line(line: str) -> str:
    line = re.sub(r"\[\d{2}:\d{2}\]\s*", "", str(line)).strip()
    line = re.sub(r"^\s*[-•]\s*", "", line).strip()
    return re.sub(r"\s+", " ", line).strip()

def _split_reviews_into_lines(reviews):
    lines = []
    for r in reviews:
        txt = str(r).strip()
        if not txt:
            continue
        if "\n" in txt:
            for ln in txt.split("\n"):
                cl = _clean_line(ln)
                if len(cl) >= 10:
                    lines.append(cl)
        else:
            parts = re.split(r"(?<=[.!?])\s+", txt)
            for p in parts:
                cl = _clean_line(p)
                if len(cl) >= 10:
                    lines.append(cl)
    return lines

def _best_category(text: str):
    t = text.lower()
    best = None
    best_score = 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score = score
            best = cat
    return best, best_score

def top_complaint_lines(reviews, top_k=3):
    lines = _split_reviews_into_lines(reviews)
    if not lines:
        return []

    bucket = {}
    counts = Counter()
    for line in lines:
        cat, score = _best_category(line)
        if score <= 0:
            cat = "other"
        counts[cat] += 1
        bucket.setdefault(cat, []).append(line)

    ordered = [c for c, _ in counts.most_common() if c != "other"] + (["other"] if "other" in counts else [])
    chosen = []
    for cat in ordered:
        if len(chosen) >= top_k:
            break
        norm_counter = Counter(re.sub(r"[^a-z0-9\s]", " ", ln.lower()).strip() for ln in bucket[cat])
        top_norm, _ = norm_counter.most_common(1)[0]
        rep = None
        for ln in bucket[cat]:
            if re.sub(r"[^a-z0-9\s]", " ", ln.lower()).strip() == top_norm:
                rep = ln
                break
        chosen.append(rep or bucket[cat][0])

    return chosen[:top_k]

def filter_complaints_by_question(reviews, question: str, max_items=8):
    q = (question or "").lower()
    lines = _split_reviews_into_lines(reviews)
    if not lines:
        return []

    focus_cats = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            focus_cats.append(cat)

    if not focus_cats:
        q_words = [w for w in re.sub(r"[^a-z\s]", " ", q).split() if len(w) >= 4]
        if not q_words:
            return []
        hits = []
        for ln in lines:
            l = ln.lower()
            if any(w in l for w in q_words):
                hits.append(ln)
        return hits[:max_items]

    hits = []
    for ln in lines:
        cat, score = _best_category(ln)
        if cat in focus_cats and score > 0:
            hits.append(ln)
    return hits[:max_items]


# =============================================================================
# Main App
# =============================================================================
def main():
    with st.spinner("Loading AI models..."):
        model_bilstm2, w2v_model, summarizer, ner_model, device = load_models()

    # Only these two are mandatory for Tab 1 (sentiment)
    if model_bilstm2 is None or w2v_model is None:
        st.error("Failed to load required models for sentiment. Ensure word2vec.model and bilstm_anger_model.pt are present (and not Git LFS pointers).")
        return

    # Sidebar configuration
    st.sidebar.header("Configuration")
    max_len = 50
    embedding_dim = 100

    st.sidebar.caption("Models")
    st.sidebar.markdown(
        """
        - Sentiment scorer: **BiLSTM2 (bilstm_2)**
        - Embeddings: **Word2Vec**
        - Summarizer: **facebook/bart-large-cnn**
        - NER: **dslim/bert-base-NER**
        """
    )

    tab1, tab2, tab3 = st.tabs(["Sentiment analysis", "Crisis report", "About"])

    # ===== TAB 1 =====
    with tab1:
        st.header("Text Sentiment analysis")
        st.markdown("Enter a text to get its score and sentiment classification.")

        review_text = st.text_area(
            "Review text",
            placeholder="E.g., 'This product is terrible, I am very upset...'",
            height=120
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Analyze", use_container_width=True):
                if review_text.strip():
                    with st.spinner("Analyzing..."):
                        score = predict_anger_score(
                            review_text,
                            model_bilstm2,
                            w2v_model,
                            max_len,
                            embedding_dim,
                            device
                        )

                    st.markdown("---")
                    col_score, col_class = st.columns(2)
                    with col_score:
                        st.metric("Score", f"{score:.3f}", delta=f"{score*100:.1f}%")
                    with col_class:
                        st.metric("Sentiment", classify_sentiment(score))

                    st.markdown("### Score Breakdown")
                    st.progress(score)

                    st.markdown("### Interpretation")
                    if score < 0.10:
                        st.error("Highly critical — extreme dissatisfaction; immediate action recommended.")
                    elif score < 0.25:
                        st.warning("Negative — indicates customer dissatisfaction.")
                    elif score < 0.55:
                        st.info("Neutral — sentiment is mixed or unclear.")
                    elif score < 0.75:
                        st.success("Positive — indicates customer satisfaction.")
                    else:
                        st.success("Highly positive — strong satisfaction.")
                else:
                    st.warning("Please enter a review text.")

        with col2:
            st.markdown("### Tips")
            st.markdown(
                """
                - **Be specific**: Include details about the issue  
                - **Express emotion**: Use emotional language  
                - **Be constructive**: Explain what went wrong  
                """
            )

    # ===== TAB 2 =====
    with tab2:
        st.header("Crisis report generator")
        st.markdown(
            """
            Upload or paste up to 1,000 negative reviews to generate an AI-powered **3-sentence Crisis Report**
            that identifies key issues and recommendations.
            """
        )

        if "vp_reviews" not in st.session_state:
            st.session_state.vp_reviews = []
        if "vp_crisis" not in st.session_state:
            st.session_state.vp_crisis = None
        if "vp_ner_top" not in st.session_state:
            st.session_state.vp_ner_top = None
        if "vp_brand_answer" not in st.session_state:
            st.session_state.vp_brand_answer = None

        input_method = st.radio(
            "How would you like to input reviews?",
            ["Paste text", "Upload CSV"],
            key="vp_input_method"
        )

        reviews = []

        if input_method == "Paste text":
            st.markdown("Enter reviews (one per line):")
            text_input = st.text_area(
                "Reviews:",
                placeholder="Review 1\nReview 2\nReview 3\n...",
                height=200,
                label_visibility="collapsed",
                key="vp_paste_text"
            )
            if text_input.strip():
                reviews = [r.strip() for r in text_input.split("\n") if r.strip()]
        else:
            uploaded_file = st.file_uploader("Upload CSV with reviews", type=["csv"], key="vp_csv")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    columns = df.columns.tolist()
                    selected_col = st.selectbox("Select text column:", columns, key="vp_csv_col")
                    reviews = df[selected_col].dropna().astype(str).tolist()
                    st.success(f"Loaded {len(reviews)} reviews")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        if reviews:
            st.info(f"Total reviews loaded: {len(reviews)}")
            if len(reviews) > 1000:
                reviews = reviews[:1000]
                st.warning("Limited to the first 1,000 reviews for analysis")

        if st.button("Generate crisis report", use_container_width=True, key="vp_generate"):
            if not reviews:
                st.error("Please provide at least one review.")
            elif summarizer is None:
                st.error("BART summarizer not available. Check transformers install / internet access / model download.")
            else:
                st.session_state.vp_reviews = reviews

                with st.spinner(f"Generating crisis report from {len(reviews)} reviews..."):
                    crisis_report, sample_len, level1_chunks, level2_chunks = generate_crisis_report(reviews, summarizer)

                st.session_state.vp_crisis = {
                    "text": crisis_report,
                    "sample_len": sample_len,
                    "level1_chunks": level1_chunks,
                    "level2_chunks": level2_chunks
                }

                with st.spinner("Extracting entities (NER)..."):
                    st.session_state.vp_ner_top = extract_top_entities(reviews, ner_model, top_k=10)

                st.session_state.vp_brand_answer = None

        if st.session_state.vp_crisis is not None:
            r = st.session_state.vp_crisis

            st.markdown("---")
            st.subheader("Crisis report")
            st.caption(
                f"Sample size used: {r['sample_len']} reviews · "
                f"Level-1 chunks: {r['level1_chunks']} · Level-2 chunks: {r['level2_chunks']}"
            )
            st.markdown(f"<div class='vp-card'>{r['text']}</div>", unsafe_allow_html=True)

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Sample Size Used", r["sample_len"])
            with col_b:
                st.metric("Report Length", f"{len(r['text'].split())} words")
            with col_c:
                st.metric("Severity", "High")

            st.markdown("---")
            st.subheader("Entity intelligence")
            top_entities = st.session_state.vp_ner_top or []
            if top_entities:
                st.markdown("#### Top mentioned entities")
                for entity, count in top_entities:
                    st.write(f"• **{entity}** — {count} mentions")
            else:
                st.info("No significant named entities detected in the reviews.")

        st.markdown("---")
        st.subheader("Brand assistant")

        user_question = st.text_input("Ask a question about the reviews:", key="vp_brand_q")

        if st.button("Ask Brand Assistant", key="vp_ask_brand"):
            question = (user_question or "").strip().lower()
            reviews_for_chat = st.session_state.vp_reviews if st.session_state.vp_reviews else reviews

            if not reviews_for_chat:
                st.warning("Please paste/upload reviews (and optionally click **Generate Crisis Report**).")
            else:
                if ("top" in question and "complaint" in question) or ("complaints" in question) or ("complaint" in question):
                    top_lines = top_complaint_lines(reviews_for_chat, top_k=3)
                    st.session_state.vp_brand_answer = ("top_complaints", top_lines) if top_lines else ("info", "No clear complaint lines found.")

                elif any(k in question for k in ["battery", "charge", "charging", "disconnect", "bluetooth", "app", "crash", "firmware", "mic", "microphone", "support", "refund", "delivery", "shipping", "anc", "noise", "hiss", "static", "fit", "uncomfortable", "touch", "controls"]):
                    hits = filter_complaints_by_question(reviews_for_chat, question, max_items=8)
                    st.session_state.vp_brand_answer = ("complaint_hits", hits) if hits else ("info", "No matching complaint lines found for that topic.")

                elif ("company" in question) or ("brand" in question):
                    if ner_model is None:
                        st.session_state.vp_brand_answer = ("warn", "NER model not available.")
                    else:
                        orgs = []
                        for review in reviews_for_chat:
                            try:
                                for ent in ner_model(review):
                                    if (ent.get("entity_group") or ent.get("entity")) == "ORG":
                                        orgs.append(_merge_wordpieces(ent.get("word", "")).strip())
                            except Exception:
                                continue
                        orgs = [o for o in orgs if o]
                        st.session_state.vp_brand_answer = ("company", Counter(orgs).most_common(1)[0]) if orgs else ("info", "No companies detected.")

                elif ("person" in question) or ("ceo" in question):
                    if ner_model is None:
                        st.session_state.vp_brand_answer = ("warn", "NER model not available.")
                    else:
                        persons = []
                        for review in reviews_for_chat:
                            try:
                                for ent in ner_model(review):
                                    if (ent.get("entity_group") or ent.get("entity")) == "PER":
                                        persons.append(_merge_wordpieces(ent.get("word", "")).strip())
                            except Exception:
                                continue
                        persons = [p for p in persons if p]
                        st.session_state.vp_brand_answer = ("person", Counter(persons).most_common(1)[0]) if persons else ("info", "No persons detected.")

                elif ("severity" in question) or ("overall" in question) or ("crisis" in question):
                    st.session_state.vp_brand_answer = ("crisis", st.session_state.vp_crisis["text"]) if st.session_state.vp_crisis else ("info", "Generate the crisis report first.")

                else:
                    st.session_state.vp_brand_answer = ("help",
                        "Try: 'top 3 complaints', or ask about battery/app/support/delivery/disconnect/ANC, "
                        "or 'most mentioned company', 'most mentioned person', 'severity'."
                    )

        if st.session_state.vp_brand_answer is not None:
            kind, payload = st.session_state.vp_brand_answer

            st.markdown("### Assistant Response")
            if kind == "top_complaints":
                st.markdown("#### Top 3 complaints")
                if payload:
                    for i, line in enumerate(payload, 1):
                        st.write(f"{i}. {line}")
                else:
                    st.info("No clear complaint lines found.")
            elif kind == "complaint_hits":
                st.markdown("#### Matching complaints")
                for i, line in enumerate(payload, 1):
                    st.write(f"{i}. {line}")
            elif kind == "company":
                org, count = payload
                st.success(f"Most mentioned company: **{org}** ({count} mentions)")
            elif kind == "person":
                person, count = payload
                st.success(f"Most mentioned person: **{person}** ({count} mentions)")
            elif kind == "crisis":
                st.success(payload)
            elif kind == "warn":
                st.warning(payload)
            elif kind == "help":
                st.info(payload)
            else:
                st.info(payload)

    # ===== TAB 3 =====
    with tab3:
        st.header("About")
        st.markdown(
            """
            This app is part of **VoxPop: AI-Driven Global Brand Sentiment & Crisis Intelligence**.
            It keeps the original modeling pipeline intact and focuses on presenting the results in a clean UI.
            """
        )

        st.markdown("### Models used")
        st.markdown(
            """
            - **BiLSTM2 (bilstm_2)**: produces an anger/sentiment score in the range **0–1** using **Word2Vec** embeddings.
            - **facebook/bart-large-cnn**: generates a concise **3-sentence crisis report** from up to **1,000** reviews.
            - **dslim/bert-base-NER**: extracts named entities (PER/ORG/LOC/MISC) mentioned in the reviews.
            """
        )

        st.markdown("### Evaluation (BiLSTM2)")
        report_df = pd.DataFrame(
            {
                "precision": [0.8266, 0.8344, 0.8305, 0.8305],
                "recall": [0.8351, 0.8259, 0.8305, 0.8305],
                "f1-score": [0.8308, 0.8302, 0.8305, 0.8305],
                "support": [159494, 160506, 320000, 320000],
            },
            index=["Not Angry (0)", "Angry (1)", "macro avg", "weighted avg"],
        )
        st.dataframe(report_df, use_container_width=True)
        st.markdown("**Accuracy:** 0.8305")

        st.markdown("### Confusion matrix (threshold = 0.5)")
        cm_df = pd.DataFrame(
            [[133194, 26300], [27942, 132564]],
            index=["True: Not Angry", "True: Angry"],
            columns=["Pred: Not Angry", "Pred: Angry"],
        )
        st.table(cm_df)

        st.markdown("### What you can do")
        st.markdown(
            """
            - **Review analysis**: score a single review and interpret the result.
            - **Crisis report**: summarize many reviews into a short crisis narrative.
            - **Entity intelligence**: see which people, organizations, or locations are mentioned.
            - **Brand assistant**: ask for top complaints or filter complaints by topic.
            """
        )

if __name__ == "__main__":
    main()
