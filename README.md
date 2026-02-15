
# Arabic RAG Evaluation with RAGAS (ArabicaQA) - LLaMA-3 vs Mistral

This repository provides the implementation and evaluation pipeline for the paper:

**A Comparative Study of LLM-Based Retrieval-Augmented Generation for Arabic Question Answering**

It benchmarks a fixed Retrieval-Augmented Generation (RAG) setup on **ArabicaQA**, comparing two instruction-tuned LLMs under identical retrieval conditions:

- **LLaMA-3 (8B)** (via Ollama)
- **Mistral-7B-Instruct** (via Ollama)

Evaluation includes classic QA metrics, retrieval ranking metrics, and **RAGAS** grounding-aware metrics.

---

## Project Structure
```
├── ArabicaQA_RAG_Eval.ipynb
├── requirements.txt
├── LICENSE
├── DATASET_LICENSE
├── README.md
│
└── arabicaqa_rag_results
    ├── dataset
    ├── predictions
    └── ragas_full
        ├── .ipynb_checkpoints
        ├── figures
        ├── figures_extra
        └── figures_full

```
### Directory Description

- `dataset/` — full ArabicaQA data structure (MRC, OpenQA) and processed evaluation subsets (e.g., 1000-example balanced sample), including intermediate analysis files.
- `predictions/` — model outputs and retrieved contexts  
- `ragas_full/` — full RAGAS evaluation outputs and generated figures  

## Repository Contents

- **`ArabicaQA_RAG_Eval.ipynb`**  
  End-to-end notebook: preprocessing => retrieval indexing => generation => evaluation (EM/overlap/retrieval/RAGAS) => results export.

---

## Dataset

This work uses the ArabicaQA dataset:
https://huggingface.co/datasets/abdoelsayed/ArabicaQA

## System Overview
```
Question
   |
   v
Retriever (MiniLM + Chroma)
   |
   V
Top-k Context
   |
   V
LLM (LLaMA-3 / Mistral via Ollama)
   |
   V
Answer
   |
   V
Task / Retrieval / RAGAS Evaluation

```

## Method Summary

### RAG pipeline (fixed across models)

1. Load ArabicaQA contexts and questions  
2. Chunk contexts into overlapping passages  
3. Embed chunks with a multilingual sentence embedding model (paraphrase-multilingual-MiniLM-L12-v2)
4. Store embeddings in a vector DB (Chroma)  
5. Retrieve top-*k* passages per query  
6. Generate answers using an LLM (Ollama backend)  
7. Evaluate using:
   - Task metrics (EM, EM25, no-answer accuracy)
   - Lexical overlap metrics (Token-F1, ROUGE-L, BLEU-short)
   - Retrieval metrics (MRR@5, nDCG@5)
   - RAGAS metrics (faithfulness, answer relevancy, context precision/recall, etc.)

---

## Experimental Configuration

| Component | Value |
|------------|--------|
| Chunk size | 500 |
| Chunk overlap | 100 |
| Embedding model | paraphrase-multilingual-MiniLM-L12-v2 |
| Retriever top-k | 5 |
| Temperature | 0.0 |
| Evaluation subset size | 1000 |
| Answerable / Unanswerable | 500 / 500 |
| Random seed | 42 |

## Installation

### 1) Create environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
pip install -U pip
````

### 2) Install dependencies

To install dependencies use `requirements.txt`, run:

```bash
pip install -r requirements.txt
```
---

## Ollama Setup (LLM Inference)

Install Ollama from its official site, then pull the models used:

```bash
ollama pull llama3:8b
ollama pull mistral:7b-instruct
```

You can verify:

```bash
ollama list
```

---

## How to Run

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

- `ArabicaQA_RAG_Eval.ipynb`

Then run cells in order.

### Expected outputs

Depending on your notebook settings, you will typically produce:

- Model predictions (answers)
- Retrieved contexts per query
- Metric tables (per subset / overall)
- RAGAS score reports (JSON/CSV)
- Summary plots (optional)

---

## Evaluation Metrics

### Task-level

- **EM** (Exact Match after normalization)
- **EM25**: Exact match computed after truncating model outputs to 25 characters post-normalization.

- **No-answer accuracy** (for unanswerable subset)

### Overlap / short-answer metrics

- **Token-F1**
- **ROUGE-L**
- **BLEU-short**

### Retrieval ranking

- **MRR@5**
- **nDCG@5**

### RAGAS (grounding-aware)

Commonly used:

- **Faithfulness**
- **Answer Relevancy**
- **Context Precision**
- **Context Recall**
- **Answer Similarity**
- **Answer Correctness**

---

## Reproducibility Notes

This repo aims to make comparisons fair by:

- Keeping the retriever fixed across models
- Using consistent chunking / top-k retrieval settings
- Using deterministic generation settings when possible (e.g., temperature=0)
- Saving retrieved contexts for consistent scoring

To reproduce exactly, ensure:

- Same dataset version / split seed
- Same chunking parameters
- Same embedding model
- Same top-k retrieval value
- Same decoding configuration

---

## Results

The paper reports model trade-offs such as:

- stronger lexical overlap / answer matching vs.
- stronger grounding (faithfulness) and abstention behavior

See the paper for full tables, subset analysis (answerable vs unanswerable), and RAGAS interpretation.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{almalki_arabicrag_2026,
  title={A Comparative Study of LLM-Based Retrieval-Augmented Generation for Arabic Question Answering},
  author={Almalki, Zohoor and Alshehri, Shahad and Alrehaili, Shatha and Althagafi, Amjad and Mars, Mourad},
  year={2026}
}
```

---
## Known Limitations

- Fixed dense retriever (no hybrid or reranking)
- Deterministic decoding only (temperature=0)
- Evaluation limited to ArabicaQA
- No human evaluation

## Compute Infrastructure

### Local Development Environment

- OS: Windows 11 Home Single Language (Build 26200)
- System Model: MSI GF65 Thin 10SDR
- CPU: Intel Core i7 (~2.6 GHz)
- RAM: 16 GB
- Python: 3.10.19
- Ollama: 0.16.1

This environment was used for RAG pipeline development, retrieval experiments, and model inference via Ollama.

---

### Cloud Execution (RAGAS Evaluation)

RAGAS evaluation was executed on **Lambda Cloud** using:

- GPU: NVIDIA A100
- Instance type: Single-GPU configuration
- OS: Ubuntu (Lambda default image)
- Session management: tmux
- Total runtime: ~15h 59m
  - ~14–15h RAGAS evaluation
  - ~1–2h setup and execution management

Long-running jobs were managed via terminal session multiplexing (tmux) to ensure uninterrupted execution in the event of SSH or network disconnections.

During the final execution phase, the internet connection was interrupted, which prevented output from streaming to the interactive notebook interface. However, because the job was running inside a tmux session on the Lambda server, execution completed successfully and all outputs were generated and saved.

## Acknowledgments

This research was supported by **Umm Al-Qura University** (Grant Number: **26UQU4350491GSSR03**).

---

## License

### Code
The source code in this repository is licensed under the terms of the included LICENSE file.

### Dataset
This work uses the ArabicaQA dataset:
The original dataset license is provided in DATASET_LICENSE.
