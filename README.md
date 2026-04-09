## 📌 Project Overview

This project builds an **intelligent loan approval system** that:

- ✅ **Predicts** whether a loan should be approved or rejected using Machine Learning
- 📚 **Retrieves** relevant financial guidelines using a RAG (Retrieval-Augmented Generation) pipeline
- 💬 **Explains** the decision in plain English using an LLM (Llama3 via Groq)
- 🎨 **Displays** everything in a clean, interactive **Streamlit web app**

> Built as a college project demonstrating the integration of classical ML with modern Generative AI techniques.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                         │
│         (Input Form → Results → AI Explanation)        │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌─────────────────┐      ┌─────────────────────┐
│   ML PIPELINE   │      │    RAG PIPELINE      │
│                 │      │                      │
│ • Random Forest │      │ • Knowledge Base     │
│ • 200 Trees     │      │ • MiniLM Embeddings  │
│ • 15 Features   │      │ • FAISS Vector DB    │
│                 │      │ • Top-4 Retrieval    │
│ → APPROVED /    │      │ → Relevant Rules     │
│   REJECTED      │      │   & Guidelines       │
└────────┬────────┘      └──────────┬───────────┘
         │                          │
         └────────────┬─────────────┘
                      ▼
          ┌───────────────────────┐
          │    LLM INTEGRATION    │
          │                       │
          │  Groq API (Llama3)    │
          │  Prompt Engineering   │
          │  Anti-Hallucination   │
          │                       │
          │ → Clear Explanation   │
          │ → Risk Analysis       │
          │ → Recommendations     │
          └───────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Prediction** | Random Forest classifier with 89%+ accuracy |
| 📊 **Risk Analysis** | Visual risk factor breakdown (Credit, DTI, Income, Employment, Collateral) |
| 🔍 **RAG Retrieval** | FAISS-powered semantic search over 8 financial guideline documents |
| 💬 **AI Explanation** | Llama3 generates grounded, hallucination-free explanations |
| 📈 **Evaluation Plots** | Confusion matrix, ROC curve, feature importance charts |
| 🎨 **Streamlit UI** | Clean, interactive web interface with real-time metrics |

---

## 📁 Project Structure

```
loan_approval_rag/
│
├── 📁 data/
│   └── generate_dataset.py      ← Generates 2,000 synthetic loan records
│
├── 📁 ml/
│   ├── preprocess.py            ← Cleaning, imputation, feature engineering, scaling
│   └── train_model.py           ← Train Random Forest + evaluation metrics + plots
│
├── 📁 rag/
│   ├── knowledge_base.py        ← 8 expert financial rule documents
│   ├── build_vectorstore.py     ← Chunking + MiniLM embeddings + FAISS index
│   └── retriever.py             ← Semantic search over knowledge base
│
├── 📁 llm/
│   └── explainer.py             ← Prompt engineering + Groq/OpenAI API calls
│
├── 📁 app/
│   └── streamlit_app.py         ← Full Streamlit web UI
│
├── 📁 models/                   ← Auto-created after training
│   ├── loan_model.pkl           ← Saved Random Forest model
│   ├── scaler.pkl               ← StandardScaler
│   ├── faiss_index.idx          ← FAISS vector store
│   └── chunks.pkl               ← Text chunks for retrieval
│
├── requirements.txt             ← All dependencies
└── README.md                    ← You are here
```

---

## 🛠️ Tech Stack

### Machine Learning
| Library | Purpose |
|---------|---------|
| `scikit-learn` | Random Forest, preprocessing, evaluation metrics |
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical computations, array operations |
| `matplotlib` + `seaborn` | Confusion matrix, ROC curve, feature importance plots |
| `joblib` | Save/load trained models |

### RAG Pipeline
| Library | Purpose |
|---------|---------|
| `sentence-transformers` | `all-MiniLM-L6-v2` — converts text to 384-dim embeddings |
| `faiss-cpu` | Facebook AI Similarity Search — fast vector retrieval |

### LLM Integration
| Library | Purpose |
|---------|---------|
| `groq` | Free, fast LLM API (Llama3-8B) |
| `openai` | Alternative — GPT-4o-mini |

### Web UI
| Library | Purpose |
|---------|---------|
| `streamlit` | Interactive web application framework |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- A free Groq API key → [console.groq.com](https://console.groq.com)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/loan-approval-rag.git
cd loan-approval-rag
```

### 2. Create Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Dataset
```bash
python data/generate_dataset.py
```
> Creates `data/loan_data.csv` with 2,000 synthetic loan records

### 5. Train ML Model
```bash
python ml/train_model.py
```
> Trains Random Forest, prints metrics, saves model + evaluation plots

### 6. Build RAG Vector Store
```bash
python rag/build_vectorstore.py
```
> Downloads MiniLM model (~80MB first run), builds FAISS index

### 7. Set API Key
```bash
# Windows
set GROQ_API_KEY=your_groq_api_key_here

# Mac/Linux
export GROQ_API_KEY=your_groq_api_key_here
```

### 8. Launch the App 🎉
```bash
streamlit run app/streamlit_app.py
```
Opens at → **http://localhost:8501**

---

## 🧪 Example Test Cases

### ✅ Approved Applicant
| Field | Value |
|-------|-------|
| Annual Income | $85,000 |
| Credit Score | 750 |
| Loan Amount | $150,000 |
| Employment | Employed |
| Existing Debt | $10,000 |
| Collateral | Yes |
| DTI Ratio | ~12% |

### ❌ Rejected Applicant
| Field | Value |
|-------|-------|
| Annual Income | $25,000 |
| Credit Score | 480 |
| Loan Amount | $400,000 |
| Employment | Unemployed |
| Existing Debt | $18,000 |
| Collateral | No |
| DTI Ratio | ~72% |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~89% |
| Precision | ~88% |
| Recall | ~87% |
| F1-Score | ~88% |
| AUC-ROC | ~94% |

---

## 🧠 How RAG Works in This Project

```
1. 8 financial documents written (credit score rules, DTI limits, etc.)
        ↓
2. Documents split into chunks (400 chars, 80 overlap)
        ↓
3. Each chunk converted to 384-dim vector (MiniLM embedding)
        ↓
4. All vectors stored in FAISS index
        ↓
5. At query time → user profile converted to vector
        ↓
6. FAISS finds 4 most similar chunks (cosine similarity)
        ↓
7. Retrieved chunks passed to LLM as context
        ↓
8. LLM generates explanation using ONLY provided context
```

---

## 💡 Key Design Decisions

**Why Random Forest?**
Handles mixed data types, provides feature importance, robust to outliers, and outperforms Logistic Regression on tabular financial data.

**Why FAISS over ChromaDB?**
Lightweight, no server needed, extremely fast exact search for small-to-medium knowledge bases.

**Why Groq + Llama3?**
Free API, very fast inference, and Llama3-8B is more than capable for structured explanation tasks.

**Anti-Hallucination Strategy:**
LLM is strictly instructed to use only provided applicant data and retrieved context. Temperature set to 0.3 for factual responses. No external knowledge allowed.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Amisha Singh
**
- GitHub: [Amisha](https://github.com/Amishasingh2005)
- LinkedIn: [Amisha_Linkedin](linkedin.com/in/amisha-singh-0680a7312)

---

<p align="center">
  ⭐ Star this repo if you found it helpful!
</p>
