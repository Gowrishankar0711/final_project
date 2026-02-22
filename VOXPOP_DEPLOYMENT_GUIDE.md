# Streamlit Deployment Guide: VoxPop (AI-Driven Global Brand Sentiment & Crisis Intelligence)

This guide is adapted from your existing Streamlit deployment template and customized for the **VoxPop** project (BiLSTM sentiment/anger model + Crisis Report summarization + NER + Brand Assistant). fileciteturn1file0L1-L25

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Prepare Your Project](#prepare-your-project)
3. [GitHub Setup](#github-setup)
4. [Streamlit Community Cloud Deployment](#streamlit-community-cloud-deployment)
5. [Configuration & Secrets Management](#configuration--secrets-management)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance & Updates](#maintenance--updates)
9. [Alternative Deployment Options](#alternative-deployment-options)

---

## Prerequisites

Before deploying, ensure you have:

- ✅ A **GitHub account**
- ✅ **Git** installed locally
- ✅ A **Streamlit Community Cloud** account (sign in using GitHub)
- ✅ Project files organized locally
- ✅ Your trained VoxPop artifacts saved in the repo (or hosted externally if large):
  - BiLSTM model: `bilstm_2` (example file names: `bilstm_2.pt` or `bilstm_2.pth`)
  - Word2Vec model (if used): `w2v.model` (or equivalent)
  - Any metadata/config used at inference time (e.g., `max_len.json`, `label_map.json`)
- ✅ A working `requirements.txt`

> Tip: Streamlit Community Cloud runs on Linux. File names and paths become **case-sensitive** after deployment.

---

## Prepare Your Project

### 1. Update `requirements.txt`

Your VoxPop app typically needs:

```txt
streamlit>=1.30.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.7.0

# NLP / preprocessing
nltk>=3.8.0
regex>=2023.0.0

# If using Word2Vec in the app
gensim>=4.3.0

# PyTorch BiLSTM inference
torch>=2.1.0

# Task 4: Transformers (Crisis Report + NER)
transformers>=4.40.0
tokenizers>=0.15.0
sentencepiece>=0.1.99

# Optional (only if you use it in your code)
accelerate>=0.26.0
```

**Generate/update automatically:**
```bash
pip freeze > requirements.txt
```

> If deployment fails due to heavy dependencies, prefer **CPU-only** runtime and keep requirements minimal.

---

### 2. Organize Project Structure

A clean VoxPop structure (recommended):

```
voxpop/
├── app.py                         # Main Streamlit app (entry point)
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
│   ├── confusion_matrix.png        # For About tab (optional)
│   └── classification_report.txt   # For About tab (optional)
├── models/
│   ├── bilstm_2.pt                 # Your BiLSTM model weights (example name)
│   └── w2v.model                   # Word2Vec model (if used)
├── data/                           # OPTIONAL: sample data only (keep small)
│   └── sample_reviews.csv
└── .streamlit/
    └── config.toml
```

**Important**
- Avoid committing massive training datasets (e.g., full Sentiment140). Keep only a small `sample_reviews.csv` for demos.
- If you use local file paths anywhere in the code, change to **relative paths** before deployment. fileciteturn1file3L99-L110

---

### 3. Create a `.gitignore`

Use this baseline (add/remove as needed):

```plaintext
# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual env
env/
venv/
.venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Large/derived data (keep repo light)
data/*.csv
data/*.parquet

# Model backups / temp
*.bak
*.tmp

# Jupyter
.ipynb_checkpoints/
```

---

### 4. Add Streamlit config: `.streamlit/config.toml`

A minimal, professional theme:

```toml
[theme]
primaryColor = "#2E6BE6"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F6F8FB"
textColor = "#0F172A"
font = "sans serif"

[client]
showErrorDetails = false
toolbarMode = "minimal"

[server]
maxUploadSize = 200
headless = true
runOnSave = true
```

(Your template uses the same idea; this is just VoxPop styling.) fileciteturn1file0L101-L124

---

### 5. Test locally before deployment

From your project folder:

```bash
streamlit run app.py
```

Confirm:
- ✅ Tab 1: Sentiment / anger prediction works with sample text/CSV
- ✅ Tab 2: Crisis Report generates output (BART)
- ✅ NER extracts entities (dslim/bert-base-NER)
- ✅ Brand Assistant answers queries (e.g., “top 3 complaints”)
- ✅ About tab loads model details (bilstm_2 + report + confusion matrix)

---

## GitHub Setup

### 1. Initialize Git repo

```bash
cd path/to/voxpop
git init
git add .
git commit -m "Initial commit: VoxPop Streamlit app"
```

### 2. Create GitHub repository

On GitHub:
- Repository name: `voxpop`
- Description: `AI-Driven Global Brand Sentiment & Crisis Intelligence (BiLSTM + Crisis Report + NER + Brand Assistant)`
- Public recommended for easy Streamlit deployment

### 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/voxpop.git
git branch -M main
git push -u origin main
```

(Your existing guide uses the same flow.) fileciteturn1file3L17-L34

---

## Streamlit Community Cloud Deployment

### 1. Sign in

1. Go to Streamlit Cloud (share site)
2. Click **Sign in with GitHub**
3. Authorize access to your repos

### 2. Deploy

1. Click **New app**
2. Choose:
   - **Repository**: `YOUR_USERNAME/voxpop`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click **Deploy**

Streamlit will:
- clone your repo
- install dependencies from `requirements.txt`
- run `streamlit run app.py`

(The steps match your template.) fileciteturn1file3L38-L72

---

## Configuration & Secrets Management

### 1. Optional: Hugging Face token (recommended)

If your app downloads models from Hugging Face at runtime, an **HF token** helps avoid rate limits.

In Streamlit Cloud:
- App → **Settings** → **Secrets**
- Add:

```toml
HF_TOKEN = "hf_your_token_here"
```

In code, read it like:

```python
import os
hf_token = os.getenv("HF_TOKEN")
```

> If you don’t set a token, the app can still work, but may be slower or hit request limits during rebuilds.

---

### 2. Verify file paths (critical)

Avoid absolute paths. Use relative paths from the repo root:

```python
# ❌ Don't do this (will break on Streamlit Cloud)
# model_path = "C:/Users/Shankar/Desktop/voxpop/models/bilstm_2.pt"

# ✅ Do this
model_path = "models/bilstm_2.pt"
```

Same rule applies to images for About tab (`assets/confusion_matrix.png`) and report files.

---

### 3. Handle large model files

If your BiLSTM / Word2Vec files are large (>100MB), you have two common options:

**Option A: Git LFS** (recommended when allowed)
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.model"
git add .gitattributes
git commit -m "Track large model files with Git LFS"
git push
```

**Option B: Host externally and download at startup**
- Put models on Google Drive (direct link), S3, Hugging Face Hub (private repo), etc.
- Download only if the file doesn’t exist, and cache it:

```python
import os
import urllib.request
import streamlit as st

@st.cache_resource
def ensure_model(local_path: str, url: str):
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path
```

(This mirrors the approach in your template.) fileciteturn1file4L63-L95

---

## Post-Deployment Verification

After your app is live:

1. Open the public URL (example: `https://voxpop.streamlit.app`)
2. Test:
   - **Sentiment/anger prediction** with a short sample input
   - **Bulk CSV** (if you have it)  
     - Ensure your CSV column name matches what your code expects (often `text`)
   - **Crisis Report**
     - Confirm it outputs the expected length (e.g., ~100 words if you changed it)
   - **NER**
     - Confirm it lists entities clearly
   - **Brand Assistant**
     - Confirm “top 3 complaints” returns complaint snippets (not summaries)
   - **About**
     - Confirm: model shown as `bilstm_2` + classification report + confusion matrix

(The checklist is the same idea as the template.) fileciteturn1file2L39-L48

---

## Troubleshooting

### Issue: `ModuleNotFoundError` (e.g., `gensim`, `torch`, `transformers`)
**Fix**
1. Add the missing package to `requirements.txt`
2. Commit & push
3. Streamlit will redeploy automatically

Example:
```bash
echo "gensim>=4.3.0" >> requirements.txt
git add requirements.txt
git commit -m "Add gensim dependency"
git push
```

---

### Issue: `FileNotFoundError: models/bilstm_2.pt not found`
**Fix**
- Ensure the file is committed (and name matches exactly; Linux is case-sensitive)
- If using Git LFS, confirm LFS files are actually pushed
- Or switch to external download at startup

(Comparable to your template’s model-not-found section.) fileciteturn1file2L81-L92

---

### Issue: Transformers model download is slow / fails
**Fix**
- Add `HF_TOKEN` in Streamlit Secrets
- Cache pipelines with `@st.cache_resource`
- Avoid reloading models inside button-click blocks (load once at startup)

---

### Issue: App reruns too often / feels slow
**Fix**
- Cache heavy objects:
  - `@st.cache_resource` for model/pipeline loading
  - `@st.cache_data` for loading datasets or computed tables

---

### Issue: Out of memory
**Fix**
- Reduce batch sizes (e.g., summarize fewer than 1,000 texts on free tier)
- Keep only required columns in memory
- Avoid storing entire raw dataset in session state
- Consider paid tier / Docker deployment if your workloads are heavy

---

## Maintenance & Updates

### Update the app

```bash
# edit code
streamlit run app.py   # test locally

git add .
git commit -m "Update: improve VoxPop UI / fixes"
git push origin main
```

Streamlit Cloud redeploys automatically after the push. fileciteturn1file2L120-L131

### Update models

1. Replace your model file(s) under `models/`
2. Commit & push
3. Redeploy happens automatically

> If you store models externally, update the hosted file and keep the URL stable.

---

## Alternative Deployment Options

If you need more control (RAM/CPU), deploy using Docker.

### Dockerfile (basic)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

(Your template includes similar alternatives.) fileciteturn1file1L32-L43

---

## Quick Reference

### Git commands
```bash
git status
git add .
git commit -m "message"
git push origin main
git pull origin main
```

### Streamlit commands
```bash
streamlit run app.py
streamlit cache clear
```

---

## Summary

To deploy VoxPop on Streamlit Community Cloud:

1. ✅ Prepare repo (`app.py`, `requirements.txt`, model artifacts)
2. ✅ Push to GitHub
3. ✅ Create a new Streamlit app pointing to your repo
4. ✅ Add optional `HF_TOKEN` secret (recommended)
5. ✅ Verify all tabs (Sentiment, Crisis Report, NER, Brand Assistant, About)
6. ✅ Maintain by pushing updates to GitHub

Your VoxPop app will be live with a shareable URL once deployment succeeds.
