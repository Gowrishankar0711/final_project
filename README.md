# VoxPop: AI-Driven Global Brand Sentiment & Crisis Intelligence

## Project Overview

This project implements a comprehensive sentiment analysis and crisis intelligence pipeline for monitoring social media feedback (tweets/reviews). It combines multiple NLP and deep learning techniques to detect sentiment, identify crisis signals, generate executive summaries, extract named entities, and perform statistical analysis on sentiment trends and a streamlit application is also built.

## Key Features

### 1. **Data Loading & Preprocessing**
   - Downloads Sentiment140 dataset from Hugging Face
   - Cleans raw social media text using regex patterns
   - Removes URLs, mentions, hashtags, and special characters
   - Handles case normalization and extra whitespace

### 2. **Feature Engineering**
   - **Tokenization**: Uses NLTK for word tokenization
   - **Word Embeddings**: Trains Word2Vec model using Gensim
   - **TF-IDF Vectorization**: Extracts text features for clustering and classification

### 3. **Clustering Analysis**
   - **K-Means Clustering** with TF-IDF features
   - **Elbow Method**: Finds optimal number of clusters
   - **Evaluation Metrics**:
     - Silhouette Score (measures cluster cohesion)
     - Calinski-Harabasz Score (ratio-based evaluation)
   - **Visualization**: PCA 2D projection of clusters

### 4. **Sentiment Classification**
   - **Logistic Regression** for binary sentiment classification
   - **Metrics Computed**:
     - Accuracy, Precision, Recall, F1-Score
     - Confusion Matrix
     - ROC-AUC Score
   - Feature importance analysis of top positive/negative words

### 5. **Constructiveness Detection**
   - Identifies noisy vs. constructive feedback
   - Uses heuristic-based labeling (meaningful word count)
   - Trains separate Logistic Regression model
   - Balanced class weights for imbalanced data

### 6. **Deep Learning - BiLSTM Model**
   - **Architecture**:
     - Bidirectional LSTM (2 layers, 256 hidden units)
     - Dropout (0.3) for regularization
     - Fully connected layers with ReLU activation
     - Sigmoid output for binary classification
   - **Training**:
     - BCEWithLogitsLoss as loss function
     - Adam optimizer with learning rate scheduling
     - Gradient clipping to prevent explosions
   - **Anger Scoring**: Generates continuous anger scores [0,1]
   - **Performance**: Classification report and confusion matrix

### 7. **Text Summarization (BART)**
   - Extracts negative tweets/reviews
   - Multi-level chunking for long documents
   - Uses Facebook's BART model for abstractive summarization
   - Generates executive **3-sentence crisis reports**
   - Hierarchical summarization with re-summarization

### 8. **Named Entity Recognition (NER)**
   - Uses BERT-base-NER model (dslim/bert-base-NER)
   - Extracts entity types:
     - **ORG**: Organizations/Competitors
     - **PER**: People/CEO names
     - **MISC**: Other important entities
   - Groups and ranks entities by frequency



## Data Pipeline Flow

```
Raw Tweets/Reviews (CSV)
    ↓
Text Cleaning & Preprocessing
    ↓
Tokenization & Word2Vec Embeddings
    ↓
├─→ Clustering (K-Means + TF-IDF)
├─→ Logistic Regression (Sentiment)
├─→ Constructiveness Detection
├─→ BiLSTM (Anger Scoring)
├─→ BART (Crisis Summarization)
├─→ NER (Entity Extraction)
└─→ Statistical Tests
```

## Technical Stack

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
  - TfidfVectorizer, KMeans, LogisticRegression
  - Classification metrics, confusion matrix

### NLP & Deep Learning
- **nltk**: Natural Language Toolkit (tokenization)
- **gensim**: Word2Vec model training
- **torch / pytorch**: PyTorch for BiLSTM implementation
- **transformers**: Hugging Face models (BART, BERT-NER)

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Enhanced visualizations (confusion matrix heatmap)

### Statistical Analysis
- **scipy**: Chi-squared test, statistical computations

## Input Data Format

### CSV Requirements
- **Required column**: `text` (raw social media text/reviews)
- **Optional columns**: 
  - `timestamp` or `date` (for temporal analysis)
  - `sentiment` (pre-labeled: 0=negative, 4=positive for Sentiment140)
  - `source` (tweet, review, etc.)

### Sample Data Structure
```
| text                              | date                | sentiment |
|-----------------------------------|--------------------|-----------|
| "Love this product! #amazing"     | 2026-02-22 10:15  | 4 (positive) |
| "Battery dies instantly. Useless" | 2026-02-22 12:05  | 0 (negative) |
```

## Model Architecture Details

### BiLSTM Model
```python
- Input: [Batch, Max_len=50, Embedding_dim=100]
- LSTM Layer: bidirectional=True, num_layers=2, hidden_dim=256
- Output Layer: [Batch, 1] (logit for binary classification)
- Loss: BCEWithLogitsLoss
```

### Hyperparameters
- **Max Sequence Length**: 50 tokens
- **Embedding Dimension**: 100
- **LSTM Hidden Dimension**: 256
- **Batch Size**: 64
- **Epochs**: 8 (with learning rate scheduling)
- **Learning Rate**: 0.001 (Step decay)
- **Dropout**: 0.3

## Output and Results

### Generated Outputs
1. **Cluster Assignments**: Sentiment cluster labels (0, 1, 2, etc.)
2. **Anger Scores**: Continuous [0,1] scores for each text
3. **Crisis Labels**: Categorical (Negative/Neutral/Positive)
4. **Crisis Report**: 3-sentence executive summary
5. **Entity Lists**: Top extracted organizations, people, products
6. **Statistical Tests**: Chi-squared p-value and Cramer's V

### Evaluation Metrics
- BiLSTM Test Accuracy: ~80-85%
- Logistic Regression Accuracy: ~75-80%
- Silhouette Score (Clustering): ~0.3-0.5 range

## Usage Instructions

### 1. Setup Environment
```bash
conda create -n voxpop python=3.10 -y
conda activate voxpop
pip install -r requirements.txt
```

### 2. Run Notebook
```bash
jupyter notebook voxpop_jupyternotebook.ipynb
```

### 3. Key Sections to Execute
- **Cell 1-3**: Download Sentiment140 data
- **Cell 4-15**: Data loading and exploration
- **Cell 16-50**: Text preprocessing and clustering
- **Cell 51-100**: Sentiment classification
- **Cell 101-130**: BiLSTM model training
- **Cell 131-150**: Text summarization and NER
- **Cell 151-164**: Statistical analysis

## Model Training Notes

### BiLSTM Training
- Trains on ~1,280,000 samples (80% of Sentiment140)
- GPU acceleration recommended (CUDA)
- CPU fallback available (slower but functional)
- Gradient clipping applied to stabilize training

### BART Summarization
- Uses `facebook/bart-large-cnn` model
- Generates 3-sentence reports from 1000 negative samples
- Hierarchical chunking for long texts
- Safe token limits: 900 tokens per chunk

## Troubleshooting

### Common Issues

1. **CUDA/GPU Memory Issues**
   - Solution: Set `device = torch.device("cpu")` or reduce batch size

2. **BART Summarization Too Slow**
   - Solution: Use `facebook/distilbart-cnn-12-6` (smaller model)
   - Reduce sample size from 1000 to 300-500

3. **NER Missing Entities**
   - Solution: Increase sample size or use ensemble approach
   - Consider regex-based post-processing for domain-specific entities

4. **Out of Memory during Training**
   - Solution: Reduce `MAX_LEN` from 50 to 30
   - Reduce batch size from 64 to 32

## Advanced Configurations

### Word2Vec Parameters
```python
vector_size=100      # Embedding dimension
window=5            # Context window
min_count=5         # Min frequency threshold
workers=4           # Parallel processing
```

### K-Means Parameters
```python
n_clusters=3        # Number of clusters (tuned via elbow method)
random_state=42     # Reproducibility
n_init=10           # Number of initializations
max_iter=300        # Maximum iterations
```

### BiLSTM Hyperparameters
```python
hidden_dim=256      # LSTM hidden state dimension
num_layers=2        # Number of stacked LSTM layers
dropout=0.3         # Dropout rate
lr=0.001           # Learning rate
step_size=3         # Learning rate schedule (step size)
gamma=0.5          # Learning rate decay factor
```

## Performance Metrics

### Classification Results
- **Accuracy**: 80-85%
- **Precision (Positive)**: ~82%
- **Recall (Positive)**: ~78%
- **F1-Score**: ~80%
- **ROC-AUC**: ~0.85

### Clustering Evaluation
- **Silhouette Score**: 0.35-0.45
- **Calinski-Harabasz Index**: 400-600
- **Within-cluster sum of squares (Elbow point)**: k=3

## Statistical Significance
- **Chi-Squared Test**: p-value < 0.05 (sentiment distribution varies over time)
- **Cramer's V**: 0.1-0.3 (weak to moderate effect size)

## File Structure
```
voxpop_jupyternotebook.ipynb    # Main notebook
requirements.txt                 # Python dependencies
bilstm_anger_model.pt           # Saved BiLSTM weights
word2vec.model                  # Saved Word2Vec model
```

# streamlit
A Streamlit app that detects sentiment and potential crisis signals, generates an executive-ready **3-sentence Crisis Report**, extracts key **named entities** (brands/competitors, products, people/CEO names), and provides a lightweight **Brand Assistant** interface for common questions such as *"What are the top 3 complaints ?"*.

## Key Features

- **Sentiment & risk scoring (BiLSTM baseline)**
  - Uses your trained BiLSTM sentiment/anger model to score incoming text and flag crisis intensity.
- **Crisis Report (BART summarization)**
  - Condenses up to **1,000 negative items** into a **3-sentence executive summary**.
  - Robust chunking + re-summarization for long inputs.
- **Entity Intelligence (BERT NER)**
  - Extracts entities using Hugging Face model `dslim/bert-base-NER`.
  - Groups entities into practical buckets: **Organizations/Competitors**, **Products**, **People/CEO names**.
- **Brand Assistant (Q&A UX)**
  - For the question **“top 3 complaints ”**, returns **complaints only** (no extra summary).
  - Handles other questions with a short, grounded answer using the same filtered data.
- **Streamlit UI**
  - Multi-tab flow (Sentiment + Crisis Report + NER + Assistant).

