# Audio Deepfake Detection (Multimodal ML) 🎙️🚫
### *Synthetic Speech Detection utilizing IndicWav2Vec and Speaker-Disjoint Evaluation*

**Audio Deepfake Detection** is a multimodal machine learning project focused on identifying synthetic audio outputs across diverse linguistic profiles. By leveraging **Huggingface’s IndicWav2Vec** and a custom **PyTorch** classifier, the system achieves robust detection performance on a large-scale dataset of over **78,000 segments**. A key contribution of this research is the implementation of a **strict speaker-disjoint test set** to ensure the model generalizes across unseen identities rather than memorizing speaker-specific artifacts.

---

## 🚀 Key Research Highlights

* **Wav2Vec Feature Extraction:** Utilizes **IndicWav2Vec** to extract high-dimensional audio embeddings (768-D), capturing nuanced acoustic features essential for detecting synthetic speech.
* **Large-Scale Data Diversity:** Handled and evaluated a massive dataset consisting of **78,453 audio segments** from **21,140 unique speakers**.
* **Mitigated Data Leakage:** Implemented a **strict speaker-disjoint evaluation strategy**, ensuring no overlap between speakers in the training and testing sets to validate true generalization.
* **Proven Metrics:** Achieved an **AUC of 0.8293**, an **F1-score of 0.7221**, and a total accuracy of **74.74%** on a speaker-disjoint test set.

---

## 📊 Performance Summary

| Metric | Result |
| :--- | :--- |
| **AUC** | 0.8293 |
| **F1-Score** | 0.7221 |
| **Accuracy** | 74.74% |
| **Unique Speakers** | 21,140 |

---

## 🛠️ Technical Stack

* **ML Frameworks:** PyTorch, Hugging Face Transformers
* **Audio Processing:** Librosa, SoundFile
* **Feature Extraction:** IndicWav2Vec
* **Environment:** Docker, Jupyter Notebooks

---

## 📂 Project Structure

    ├── core/           # ML Engine: Wav2Vec Extractor & MLP Classifier
    ├── data/           # Data Loaders & Speaker Distribution Analysis
    ├── evaluation/     # Speaker-Disjoint Splitting & Performance Metrics
    ├── notebooks/      # Research EDA & Training Experiments
    ├── Dockerfile      # Reproducible Research Environment
    └── requirements.txt # Pinned Multimodal dependencies

---



## ⚙️ Setup & Evaluation

1. **Clone the repository:**
    ```bash
    git clone git@github.com:yourusername/Audio-Deepfake-Detection.git
    cd Audio-Deepfake-Detection
    ```

2. **Environment via Docker:**
    ```bash
    docker build -t deepfake-detection .
    docker run --gpus all deepfake-detection
    ```

3. **Run Metrics Validation:**
    ```bash
    python -m evaluation.metrics
    ```

---

## 🛡️ Research Integrity
To maintain the highest standards of research integrity, this project rejects the traditional segment-based split. By splitting the dataset by **Speaker ID**, we ensure the model learns the intrinsic difference between real and synthetic vocal textures rather than simply recognizing the speaker's unique voice signature.

---
**Author:** Gopesh Pandey | B.Tech Computer Science (AI & ML)
