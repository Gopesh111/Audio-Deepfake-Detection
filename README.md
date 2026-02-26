# Automotive NER & QA Pipeline 🚗💨
### *Fine-tuned BERT Transformers for Domain-Specific NLP & MLOps Lifecycle*

**Automotive NER & QA Pipeline** is an end-to-end machine learning system designed to extract insights from unstructured automotive technical data. By fine-tuning **BERT transformer models**, the system performs high-accuracy **Named Entity Recognition (NER)** to identify car models and components, alongside a **Question Answering (QA)** engine for technical troubleshooting. The project demonstrates a complete **MLOps lifecycle**, from token-level noise reduction to scalable deployment on **GCP Cloud Run**.

---

## 🚀 Key Engineering Highlights

* **Fine-tuned Transformers:** Specialized **BERT** models trained using **Hugging Face** and **PyTorch** for domain-specific entity extraction (e.g., CAR_MODEL, PART_NUMBER, SYMPTOM).
* **Token-Level Noise Reduction:** Custom preprocessing pipeline designed to handle "noisy" unstructured data from technical logs, standardized technical shorthand, and hex artifacts.
* **Scalable REST APIs:** High-performance inference endpoints built with **FastAPI**, featuring model pre-loading to mitigate "cold start" latency.
* **Complete MLOps Lifecycle:** Fully containerized via **Docker** and deployed using **GCP Cloud Run** for automated scaling and resource management.

---

## 🛠️ Technical Stack

* **ML Frameworks:** PyTorch, Hugging Face Transformers
* **Language:** Python 3.9+
* **Backend:** FastAPI (REST API)
* **Cloud & DevOps:** Google Cloud Platform (GCP), Cloud Run, Docker
* **Data Processing:** Pandas, NumPy, Regex

---

## 📂 Project Structure

    ├── api/            # Scalable REST API layer (FastAPI)
    ├── core/           # The ML Engine: BERT Models & Noise Reduction
    ├── cloud_run/      # GCP Deployment scripts & IaC (service.yaml)
    ├── data/           # Automotive-specific data loaders
    ├── tests/          # Validation for inference & API scalability
    ├── Dockerfile      # Optimized containerization for Cloud Run
    └── requirements.txt # Pinned ML & API dependencies

---



## ⚙️ Installation & Setup

1. **Clone the repository:**
    ```bash
    git clone git@github.com:yourusername/Automotive-NER-QA.git
    cd Automotive-NER-QA
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Deploy to GCP Cloud Run:**
    ```bash
    chmod +x cloud_run/deploy.sh
    ./cloud_run/deploy.sh
    ```

---

## 🧪 Model Validation

This project includes a suite of tests to ensure model consistency and API reliability under load.

**Run the validation suite:**
```bash
pytest -v