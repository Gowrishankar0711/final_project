import streamlit as st
import torch
import numpy as np
import re
from gensim.models import Word2Vec
from transformers import pipeline
import pandas as pd
from collections import Counter

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="VoxPop | Crisis Intelligence",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Minimal UI styling (does not change model logic)
# ============================================================================
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
st.markdown("<div class='vp-muted' style='margin-top:0.2rem; margin-bottom:1.6rem;'>Analyze text sentiment, detect crisis signals, summarize high-volume complaints, and extract named entities.</div>", unsafe_allow_html=True)

# ============================================================================
# Model Components
# ============================================================================

@st.cache_resource
def load_models():
    """Load BiLSTM2 model, Word2Vec, BART summarizer, and BERT NER."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"Runtime device: {device}")

    # Load Word2Vec
    try:
        w2v_model = Word2Vec.load("jupyter notes/word2vec.model")
        st.success("Word2Vec model loaded")
    except Exception as e:
        st.error(f"Word2Vec not found: {e}")
        w2v_model = None

    # Load BiLSTM2
    try:
        model_bilstm2 = BetterBiLSTM2(
            embedding_dim=100,
            hidden_dim=256,
            output_dim=1,
            num_layers=2,
            dropout=0.3
        ).to(device)
        model_bilstm2.load_state_dict(torch.load("jupyter notes/bilstm_anger_model.pt", map_location=device))
        model_bilstm2.eval()
        st.success("BiLSTM2 model loaded")
    except Exception as e:
        st.error(f"BiLSTM2 model not found: {e}")
        model_bilstm2 = None

    # Load BART summarizer
    try:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        st.success("BART summarizer loaded")
    except Exception as e:
        st.error(f"BART summarizer error: {e}")
        summarizer = None

    # Load BERT NER (CPU to avoid CUDA crash)
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


# ============================================================================
# Text Processing Functions
# ============================================================================

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
        chunks.append(" ".join(words[i:i+max_words]))
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

    # Clean texts
    clean_texts = [str(t).strip().replace("\n", " ") for t in texts if str(t).strip()]
    if not clean_texts:
        return ("No valid text data to summarize.", 0, 0, 0)

    sample_len = len(clean_texts)

    # Build one document (bullet list helps structure)
    big_doc = "\n".join([f"- {t}" for t in clean_texts])

    tokenizer = _summarizer.tokenizer
    max_model_tokens = getattr(_summarizer.model.config, "max_position_embeddings", 1024)
    # Keep a margin under the model limit.
    chunk_tokens = min(900, max_model_tokens - 100)

    def tokenize_to_ids(text):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=False
        )
        return encoded["input_ids"][0]

    def ids_to_text(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    def split_ids(ids, size):
        return [ids[i:i+size] for i in range(0, len(ids), size)]

    def summarize_text(text, max_len=250, min_len=150):
        out = _summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        return out[0]["summary_text"] if out else ""

    # -------------------------
    # Level 1: summarize token-safe chunks of the full document
    # -------------------------
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

    # -------------------------
    # Level 2: summarize merged summaries again in token-safe chunks
    # -------------------------
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

    # -------------------------
    # Final: produce a concise 3-sentence report
    # -------------------------
    crisis_report = summarize_text(merged2, max_len=90, min_len=40)

    return (crisis_report, sample_len, level1_chunks, level2_chunks)


# ============================================================================
# NER Helpers (fixes broken wordpieces like "Mu", "Ch", etc.)
# ============================================================================

# Common non-entity words that sometimes get picked up as fragments by CoNLL NER models
STOP_ENTITY_WORDS = {
    "a","an","the","and","or","but","if","so","to","of","in","on","for","with","without","from",
    "this","that","these","those","it","its","i","we","you","he","she","they","my","your","our","their",
    "also","see","here","there","was","were","is","are","be","been","as","at","by"
}

def _merge_wordpieces(word: str) -> str:
    """Merge '##' wordpieces if they appear in a single string."""
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
    """
    Expand a (start, end) span to full word boundaries in the original text.
    This fixes cases where the model tags only a prefix like 'Mu' in 'Mumbai'.
    """
    if start is None or end is None:
        return ""
    n = len(text)
    start = max(0, min(start, n))
    end = max(0, min(end, n))

    # expand left
    l = start
    while l > 0 and (text[l-1].isalnum() or text[l-1] in {"_", "-"}):
        l -= 1

    # expand right
    r = end
    while r < n and (text[r].isalnum() or text[r] in {"_", "-"}):
        r += 1

    return text[l:r]

def _clean_entity(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ")
    s = re.sub(r"^[#@]+", "", s)  # remove leading hashtags/handles
    s = re.sub(r"\s+", " ", s).strip()
    # trim surrounding punctuation
    s = s.strip(" ,.;:!?()[]{}\"'`")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_top_entities(reviews, ner_model, top_k=10):
    """Run NER on reviews and return top entities (PER/ORG/LOC/MISC) with robust post-processing."""
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
            group = ent.get("entity_group") or ent.get("entity")  # compatibility across transformers versions
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
                spans.append({
                    "start": start if isinstance(start, int) else None,
                    "end": end if isinstance(end, int) else None,
                    "group": group,
                    "text": raw
                })

        if not spans:
            continue

        # If we have character positions, merge adjacent entities of same type (e.g., 'San' + 'Francisco')
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

                if (
                    s["group"] == prev["group"]
                    and 0 <= gap <= 2
                    and re.fullmatch(r"[\s\-\/&]+", delim or " ")
                ):
                    prev["end"] = s["end"]
                    prev["text"] = _clean_entity(prev["text"] + (delim or " ") + s["text"])
                else:
                    merged.append(s)

            candidates = [m["text"] for m in merged]
        else:
            candidates = [s["text"] for s in spans]

        # Filter junk/fragments
        for cand in candidates:
            c = _clean_entity(cand)
            if not c:
                continue
            cl = c.lower()

            # remove common stopwords
            if cl in STOP_ENTITY_WORDS:
                continue

            # remove very short fragments unless all caps (e.g., USA, UK)
            if len(c) < 4 and not c.isupper():
                continue

            # remove fragments that are purely digits/punct
            if not re.search(r"[A-Za-z]", c):
                continue

            all_entities.append(c)

    if not all_entities:
        return []

    return Counter(all_entities).most_common(top_k)

# ============================================================================
# Brand Assistant Helpers (returns complaint lines, not word counts)
# ============================================================================

CATEGORY_KEYWORDS = {
    "battery / charging": [
        "battery", "drain", "drains", "charging", "charge", "charged", "case", "hours", "power"
    ],
    "disconnect / bluetooth": [
        "disconnect", "disconnected", "drops", "dropping", "connection", "bluetooth", "pair", "pairing"
    ],
    "app / firmware": [
        "app", "crash", "crashes", "update", "firmware", "login", "log in", "logs", "settings", "re-pair", "repair"
    ],
    "mic / call quality": [
        "mic", "microphone", "muffled", "audio", "voice", "call", "calls", "zoom", "meet"
    ],
    "noise cancellation / static": [
        "anc", "noise cancellation", "noise-cancellation", "hiss", "static", "transparency"
    ],
    "support / refund": [
        "support", "customer support", "service", "refund", "replacement", "warranty", "return", "policy"
    ],
    "delivery / shipping": [
        "delivery", "shipping", "late", "delayed", "tracking", "courier"
    ],
    "comfort / fit": [
        "uncomfortable", "pain", "painful", "fit", "ear tips", "eartips"
    ],
    "controls": [
        "touch", "controls", "pause", "skips", "skip", "sensitive"
    ],
}


def _clean_line(line: str) -> str:
    # remove timestamps like [07:10]
    line = re.sub(r"\[\d{2}:\d{2}\]\s*", "", str(line)).strip()
    # remove bullet markers
    line = re.sub(r"^\s*[-•]\s*", "", line).strip()
    return re.sub(r"\s+", " ", line).strip()


def _split_reviews_into_lines(reviews):
    """
    If user pastes multiple complaints in one big paragraph, this tries to break it into lines/sentences.
    We still keep it simple to avoid heavy dependencies.
    """
    lines = []
    for r in reviews:
        txt = str(r).strip()
        if not txt:
            continue
        # if it already looks line-like, keep
        if "\n" in txt:
            for ln in txt.split("\n"):
                cl = _clean_line(ln)
                if len(cl) >= 10:
                    lines.append(cl)
        else:
            # split sentences
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
        score = 0
        for kw in kws:
            if kw in t:
                score += 1
        if score > best_score:
            best_score = score
            best = cat
    return best, best_score


def top_complaint_lines(reviews, top_k=3):
    """
    Returns top_k complaint lines (one representative per top category).
    Output is ONLY complaint text lines.
    """
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

    # take top categories excluding "other" if possible
    ordered = [c for c, _ in counts.most_common() if c != "other"] + (["other"] if "other" in counts else [])
    chosen = []
    for cat in ordered:
        if len(chosen) >= top_k:
            break
        # representative = most frequent normalized line in that category (dedupe)
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
    """
    If user asks for a specific area (battery/app/support...), return matching complaint lines.
    """
    q = (question or "").lower()
    lines = _split_reviews_into_lines(reviews)
    if not lines:
        return []

    # infer focus categories from keywords in the question
    focus_cats = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            focus_cats.append(cat)

    # if no category match, fallback: keyword search in lines
    if not focus_cats:
        # take meaningful words from question
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


# ============================================================================
# Main App
# ============================================================================

def main():
    # Load models
    with st.spinner("Loading AI models..."):
        model_bilstm2, w2v_model, summarizer, ner_model, device = load_models()

    if model_bilstm2 is None or w2v_model is None:
        st.error("Failed to load required models. Please ensure model files are present.")
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


    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Sentiment analysis", "Crisis report", "About"])

    # ===== TAB 1: Individual Review Analysis =====
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

                    # Display results
                    st.markdown("---")
                    col_score, col_class = st.columns(2)

                    with col_score:
                        st.metric("Score", f"{score:.3f}", delta=f"{score*100:.1f}%")

                    with col_class:
                        st.metric("Sentiment", classify_sentiment(score))

                    # Score visualization
                    st.markdown("### Score Breakdown")
                    progress_val = score
                    st.progress(progress_val)

                    # Interpretation
                    st.markdown("### Interpretation")
                    if score <  0.10:
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
            st.markdown("""
            - **Be specific**: Include details about the issue
            - **Express emotion**: Use emotional language
            - **Be constructive**: Explain what went wrong

            """)

    # ===== TAB 2: Crisis Report Generator =====
    with tab2:
        st.header("Crisis report generator")
        st.markdown("""
        Upload or paste up to 1,000 negative reviews to generate an AI-powered **3-sentence Crisis Report**
        that identifies key issues and recommendations.
        """)

        # -----------------------------
        # Session state (keeps results after reruns)
        # -----------------------------
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

        # Limit to 1000 reviews
        if reviews:
            st.info(f"Total reviews loaded: {len(reviews)}")
            if len(reviews) > 1000:
                reviews = reviews[:1000]
                st.warning("Limited to the first 1,000 reviews for analysis")

        # -----------------------------
        # Generate Crisis Report (stores outputs so they don't disappear)
        # -----------------------------
        if st.button("Generate crisis report", use_container_width=True, key="vp_generate"):
            if not reviews:
                st.error("Please provide at least one review.")
            elif summarizer is None:
                st.error("BART summarizer not available. Cannot generate report.")
            else:
                st.session_state.vp_reviews = reviews  # cache for later questions

                with st.spinner(f"Generating crisis report from {len(reviews)} reviews..."):
                    crisis_report, sample_len, level1_chunks, level2_chunks = generate_crisis_report(reviews, summarizer)

                st.session_state.vp_crisis = {
                    "text": crisis_report,
                    "sample_len": sample_len,
                    "level1_chunks": level1_chunks,
                    "level2_chunks": level2_chunks
                }

                # Precompute NER top entities on same reviews (cleaned & merged)
                with st.spinner("Extracting entities (NER)..."):
                    st.session_state.vp_ner_top = extract_top_entities(reviews, ner_model, top_k=10)

                # Reset last assistant answer after regeneration
                st.session_state.vp_brand_answer = None

        # -----------------------------
        # Show Crisis Report (from session_state)
        # -----------------------------
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

            # -----------------------------
            # NER Insights (from session_state)
            # -----------------------------
            st.markdown("---")
            st.subheader("Entity intelligence")

            top_entities = st.session_state.vp_ner_top or []
            if top_entities:
                st.markdown("#### Top mentioned entities")
                for entity, count in top_entities:
                    st.write(f"• **{entity}** — {count} mentions")
            else:
                st.info("No significant named entities detected in the reviews.")

        # -----------------------------
        # Brand Assistant (works even after reruns)
        # -----------------------------
        st.markdown("---")
        st.subheader("Brand assistant")

        user_question = st.text_input("Ask a question about the reviews:", key="vp_brand_q")

        if st.button("Ask Brand Assistant", key="vp_ask_brand"):
            question = (user_question or "").strip().lower()
            reviews_for_chat = st.session_state.vp_reviews if st.session_state.vp_reviews else reviews

            if not reviews_for_chat:
                st.warning("Please paste/upload reviews (and optionally click **Generate Crisis Report**).")
            else:
                # 1) TOP COMPLAINTS (returns ONLY complaint lines, not word counts)
                if ("top" in question and "complaint" in question) or ("top" in question and "issue" in question) or ("what" in question and "complaint" in question) or ("complaint" in question) or ("complaints" in question):
                    top_lines = top_complaint_lines(reviews_for_chat, top_k=3)
                    if top_lines:
                        st.session_state.vp_brand_answer = ("top_complaints", top_lines)
                    else:
                        st.session_state.vp_brand_answer = ("info", "No clear complaint lines found.")

                # 2) Filtered complaints (battery/app/support/etc.)
                elif any(k in question for k in ["battery", "charge", "charging", "disconnect", "bluetooth", "app", "crash", "firmware", "mic", "microphone", "support", "refund", "delivery", "shipping", "anc", "noise", "hiss", "static", "fit", "uncomfortable", "touch", "controls"]):
                    hits = filter_complaints_by_question(reviews_for_chat, question, max_items=8)
                    if hits:
                        st.session_state.vp_brand_answer = ("complaint_hits", hits)
                    else:
                        st.session_state.vp_brand_answer = ("info", "No matching complaint lines found for that topic.")

                # 3) MOST MENTIONED COMPANY/BRAND
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
                        if orgs:
                            top_org = Counter(orgs).most_common(1)[0]
                            st.session_state.vp_brand_answer = ("company", top_org)
                        else:
                            st.session_state.vp_brand_answer = ("info", "No companies detected.")

                # 4) MOST MENTIONED PERSON/CEO
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
                        if persons:
                            top_person = Counter(persons).most_common(1)[0]
                            st.session_state.vp_brand_answer = ("person", top_person)
                        else:
                            st.session_state.vp_brand_answer = ("info", "No persons detected.")

                # 5) OVERALL SEVERITY / CRISIS
                elif ("severity" in question) or ("overall" in question) or ("crisis" in question):
                    if st.session_state.vp_crisis is not None:
                        st.session_state.vp_brand_answer = ("crisis", st.session_state.vp_crisis["text"])
                    else:
                        st.session_state.vp_brand_answer = ("info", "Generate the crisis report first.")

                # 6) Unknown question
                else:
                    st.session_state.vp_brand_answer = ("help",
                        "Try: 'top 3 complaints', or ask about battery/app/support/delivery/disconnect/ANC, or 'most mentioned company', 'most mentioned person', 'severity'."
                    )

        # Render the assistant's last answer (persists on reruns)
        if st.session_state.vp_brand_answer is not None:
            kind, payload = st.session_state.vp_brand_answer

            st.markdown("### Assistant Response")
            if kind == "top_complaints":
                st.markdown("#### Top 3 complaints")
                for i, line in enumerate(payload, 1):
                    st.write(f"{i}. {line}")
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


    # ===== TAB 3: About =====
    with tab3:
        st.header("About")
        st.markdown(
            """
            This app is part of **VoxPop: AI‑Driven Global Brand Sentiment & Crisis Intelligence**.
            It keeps the original modeling pipeline intact and focuses on presenting the results in a clean UI.
            """
        )

        st.markdown("### Models used")
        st.markdown(
            """
            - **BiLSTM2 (bilstm_2)**: produces an anger/sentiment score in the range **0–1** using **Word2Vec** embeddings.
            - **facebook/bart-large-cnn**: generates a concise **3‑sentence crisis report** from up to **1,000** reviews.
            - **dslim/bert-base-NER**: extracts named entities (PER/ORG/LOC/MISC) mentioned in the reviews.
            """
        )

        st.markdown("### Evaluation (BiLSTM2)")
        # Values are taken from the user's saved evaluation screenshot.
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
