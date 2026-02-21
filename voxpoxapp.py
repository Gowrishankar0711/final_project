import streamlit as st
import torch
import numpy as np
import re
from gensim.models import Word2Vec
from transformers import pipeline
import pandas as pd

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="VoxPop Crisis Detective",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 VoxPop Crisis Detective")
st.markdown("""
Analyze sentiment from customer reviews and detect crisis patterns using AI.
""")

# ============================================================================
# Model Components
# ============================================================================

@st.cache_resource
def load_models():
    """Load BiLSTM2 model, Word2Vec, and BART summarizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"💻 Using device: {device}")
    
    # Load Word2Vec
    try:
        w2v_model = Word2Vec.load("jupyter notes/word2vec.model")
        st.success("✓ Word2Vec model loaded")
    except Exception as e:
        st.error(f"⚠ Word2Vec not found: {e}")
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
        st.success("✓ BiLSTM2 model loaded")
    except Exception as e:
        st.error(f"⚠ BiLSTM2 model not found: {e}")
        model_bilstm2 = None
    
    # Load BART summarizer
    try:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        st.success("✓ BART summarizer loaded")
    except Exception as e:
        st.error(f"⚠ BART summarizer error: {e}")
        summarizer = None
    
    return model_bilstm2, w2v_model, summarizer, device


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


@st.cache_data
def generate_crisis_report(texts, summarizer):
    """Generate a 3-sentence crisis report from negative reviews."""
    if not texts or len(texts) == 0:
        return "No negative reviews to analyze."
    
    # Clean texts
    clean_texts = [str(t).strip().replace("\n", " ") for t in texts if str(t).strip()]
    
    if not clean_texts:
        return "No valid text data to summarize."
    
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    
    # Step 1: Combine texts
    status_placeholder.text("📝 Step 1/3: Preparing text data...")
    big_doc = "\n".join([f"- {t}" for t in clean_texts])
    progress_bar.progress(25)
    
    # Step 2: First-level summarization
    status_placeholder.text("📝 Step 2/3: First-level summarization...")
    chunks = chunk_by_words(big_doc, max_words=350)
    chunk_summaries = []
    
    for i, ch in enumerate(chunks):
        try:
            out = summarizer(
                ch,
                max_length=120,
                min_length=40,
                do_sample=False
            )[0]["summary_text"]
            chunk_summaries.append(out)
        except Exception as e:
            st.warning(f"Chunk {i+1} summarization failed: {str(e)[:100]}")
    
    progress_bar.progress(60)
    
    # Step 3: Hierarchical summarization
    status_placeholder.text("📝 Step 3/3: Generating crisis report...")
    merged = " ".join(chunk_summaries)
    merged_chunks = chunk_by_words(merged, max_words=300)
    
    mid_parts = []
    for ch in merged_chunks:
        try:
            out = summarizer(
                ch,
                max_length=120,
                min_length=40,
                do_sample=False
            )[0]["summary_text"]
            mid_parts.append(out)
        except Exception as e:
            st.warning(f"Merge summarization failed: {str(e)[:100]}")
    
    progress_bar.progress(85)
    
    # Step 4: Final crisis report
    merged2 = " ".join(mid_parts)
    
    try:
        crisis_report = summarizer(
            merged2,
            max_length=120,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        crisis_report = "Crisis report generation failed. Please try with fewer reviews."
    
    progress_bar.progress(100)
    status_placeholder.empty()
    
    return crisis_report


# ============================================================================
# Main App
# ============================================================================

def main():
    # Load models
    with st.spinner("Loading AI models..."):
        model_bilstm2, w2v_model, summarizer, device = load_models()
    
    if model_bilstm2 is None or w2v_model is None:
        st.error("❌ Failed to load required models. Please ensure model files are present.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_len = 50
    embedding_dim = 100
    
    # Main tabs
    tab1, tab2 = st.tabs(["🎯 Individual Review Analysis", "🚨 Crisis Report Generator"])
    
    # ===== TAB 1: Individual Review Analysis =====
    with tab1:
        st.header("🎯 Analyze Individual Review")
        st.markdown("Enter a customer review to get its sentiment score and crisis classification.")
        
        review_text = st.text_area(
            "Enter review text:",
            placeholder="E.g., 'This product is terrible, I am very upset...'",
            height=120
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Analyze Review", use_container_width=True):
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
                        st.metric("Sentiment Score", f"{score:.3f}", delta=f"{score*100:.1f}%")
                    
                    with col_class:
                        st.metric("Classification", classify_sentiment(score))
                    
                    # Score visualization
                    st.markdown("### Score Breakdown")
                    progress_val = score
                    st.progress(progress_val)
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    if score <  0.10:
                        st.error("⛔ **Highly Critical** - This review expresses extreme dissatisfaction. Immediate action needed.")
                    elif score < 0.25:
                        st.warning("⚠️ **Negative** - This review indicates customer dissatisfaction.")
                    elif score < 0.55:
                        st.info("ℹ️ **Neutral** - This review is balanced or unclear in sentiment.")
                    elif score < 0.75:
                        st.success("✓ **Positive** - This review shows customer satisfaction.")
                    else:
                        st.success("✓✓ **Highly Positive** - This review expresses strong satisfaction.")
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
        st.header("🚨 Crisis Report Generator")
        st.markdown("""
        Upload or paste up to 1,000 negative reviews to generate an AI-powered **3-sentence Crisis Report**
        that identifies key issues and recommendations.
        """)
        
        input_method = st.radio("How would you like to input reviews?", 
                               ["📝 Paste Text", "📤 Upload CSV"])
        
        reviews = []
        
        if input_method == "📝 Paste Text":
            st.markdown("Enter reviews (one per line):")
            text_input = st.text_area(
                "Reviews:",
                placeholder="Review 1\nReview 2\nReview 3\n...",
                height=200,
                label_visibility="collapsed"
            )
            if text_input.strip():
                reviews = [r.strip() for r in text_input.split("\n") if r.strip()]
        
        else:  # CSV upload
            uploaded_file = st.file_uploader("Upload CSV with reviews", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Let user select column
                    columns = df.columns.tolist()
                    selected_col = st.selectbox("Select text column:", columns)
                    
                    reviews = df[selected_col].dropna().astype(str).tolist()
                    st.success(f"Loaded {len(reviews)} reviews")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
        
        # Limit to 1000 reviews
        if reviews:
            st.info(f"📊 Total reviews loaded: {len(reviews)}")
            if len(reviews) > 1000:
                reviews = reviews[:1000]
                st.warning("⚠️ Limited to first 1,000 reviews for analysis")
        
        # Generate report button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Generate Crisis Report", use_container_width=True):
                if not reviews:
                    st.error("Please provide at least one review.")
                elif summarizer is None:
                    st.error("BART summarizer not available. Cannot generate report.")
                else:
                    with st.spinner(f"Generating crisis report from {len(reviews)} reviews..."):
                        crisis_report = generate_crisis_report(reviews, summarizer)
                    
                    st.markdown("---")
                    st.markdown("### 🚨 CRISIS REPORT")
                    st.markdown(f"""
                    <div style="background-color: #ffe6e6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff0000;">
                    <p style="font-size: 16px; line-height: 1.6;">
                    {crisis_report}
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Summary stats
                    st.markdown("---")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Reviews Analyzed", len(reviews))
                    with col_b:
                        st.metric("Report Length", f"{len(crisis_report.split())} words")
                    with col_c:
                        st.metric("Severity", "High ⛔")
        
        with col2:
            st.markdown("### Report Details")
            st.markdown("""
            The crisis report includes:
            - **Sentence 1**: Overall sentiment severity
            - **Sentence 2**: Key recurring themes
            - **Sentence 3**: Recommended actions
            
            Use this to:
            - Identify urgent issues
            - Prioritize fixes
            - Communicate with teams
            - Track customer sentiment trends
            """)
        
        # Sample data option
        if not reviews:
            with st.expander("📋 Try with sample data"):
                sample_reviews = [
                    "The product quality is awful, it broke after 2 days.",
                    "Terrible customer service, no one responded to my complaint.",
                    "Shipping took forever and the package arrived damaged.",
                    "The price is way too high for such poor quality.",
                    "Completely disappointed with my purchase, waste of money.",
                ] * 20  # Repeat to have 100 reviews
                
                if st.button("Load Sample Reviews"):
                    st.success(f"Loaded {len(sample_reviews)} sample reviews")
                    st.session_state.sample_reviews = sample_reviews


if __name__ == "__main__":
    main()
