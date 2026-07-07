# RAG Project Scope and Objectives

## 1. Project Title
**RAG-Powered Document QA System**

## 2. Objectives
- Build a full-stack Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents.
- Integrate OCR/text extraction, chunking, embeddings, FAISS retrieval, and a local LLM endpoint.
- Deliver a production-ready API and web interface with CI/CD, containerization, and monitoring hooks.

## 3. Scope
### In Scope
- Preprocess and clean raw PDF files.
- Extract and chunk text for retrieval.
- Build vector embeddings with SentenceTransformers + FAISS.
- Integrate local model serving (LM Studio or GGUF-based endpoint).
- Implement RAG orchestration with LangChain-compatible components.
- Expose functionality through FastAPI/Streamlit frontend.
- Provide CI/CD with GitHub Actions and Docker deployment.

### Out of Scope
- Model fine-tuning.
- Complex multilingual NLP pipelines.
- Distributed cloud orchestration (Kubernetes and similar).

## 4. Input Data
- Source: PDF manuals, guides, and specification sheets.
- Formats: digitally generated PDFs and scanned PDFs (OCR fallback required).

## 5. Outputs
- Web-based chatbot/API interface.
- Structured logs of retrieval and Q&A interactions.
- Reproducible FAISS vector index and metadata artifacts.

## 6. Architecture Overview
```text
[PDF]
  -> [Text Extraction + OCR Fallback]
  -> [Cleaning + Chunking]
  -> [Embeddings (MiniLM)]
  -> [FAISS Vector Store]
  -> [Retriever]
  -> [Prompt Assembly + Context Injection]
  -> [Local LLM Endpoint]
  -> [Answer + Citations]
```

## 7. Timeline Summary
| Phase | Deadline |
| --- | --- |
| Project setup | Day 1 |
| Data extraction and cleaning | Day 3 |
| Chunking and embeddings | Day 5 |
| RAG integration | Day 6 |
| UI development | Day 8 |
| Evaluation and feedback loop | Day 9 |
| CI/CD and deployment | Day 12 |

## 8. Responsibilities
| Role | Owner |
| --- | --- |
| Project Lead | Hamza Touirs |
| Developer | Hamza Touirs |
| QA and evaluation | Hamza Touirs |
| Deployment and DevOps | Hamza Touirs |

## 9. Success Criteria
- Achieve at least 90% answer correctness on an evaluation question set.
- Maintain clean code with tests, lint checks, and version control hygiene.
- Provide reproducible deployment with Docker (and optional Hugging Face Space).
