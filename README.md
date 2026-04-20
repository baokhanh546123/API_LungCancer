# Pulmonary Carcinoma Detection Framework

**A Web-Based Diagnostic Interface Integrating Deep Learning and Clinician-Centric Design**

---

## Abstract

This repository houses a **web-based diagnostic support system** designed to facilitate the early detection of pulmonary anomalies, specifically lung cancer, via chest X-ray imagery. The architecture integrates a high-performance **FastAPI** backend with a lightweight, interactive frontend. By utilizing **GSAP** (GreenSock Animation Platform), the system ensures a fluid Human-Computer Interaction (HCI) experience, while the core inference engine leverages **Deep Learning** methodologies (TensorFlow/PyTorch) to analyze radiological data.

> **Note:** The original documentation is available in Vietnamese [here](README_vi.md).

---

## 1. System Capabilities & Architecture

The application relies on a robust, multi-tier architecture designed for scalability and research deployment.

### Core Features
* **Radiological Image Processing:** Automated ingestion and preprocessing of raw X-ray input data for diagnostic inference.
* **High-Fidelity Interface:** An interactive UI utilizing **GSAP** for sophisticated motion design, enhancing user engagement and visual feedback.
* **Asynchronous Backend:** Powered by **FastAPI**, ensuring low-latency responses and high concurrency for server-side operations.
* **Deep Learning Integration:** Deploys pre-trained convolutional neural networks (CNNs) via **TensorFlow** and **PyTorch** for binary or multi-class classification tasks.

### Database
The model development and validation processes utilize the following open-source radiological dataset:
* **Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/quynhlecl/lung-cancer-x-ray)

### Clinical Validation & Regulatory Compliance (FDA/CE Standards)
To align with international medical device standards (such as FDA 21 CFR or CE marking for SaMD), the framework incorporates the following diagnostic metrics:

* **Binary Decision:** A definitive categorical classification based on model inference.
* **Decision Score:** The raw numerical output representing the model's internal evaluation.
* **Approved Threshold:** A calibrated cut-off point optimized for clinical sensitivity and specificity.
* **Risk Probabilities ($P_{risk}$):** Quantified statistical likelihood of the presence of pathology.
* **Prediction Confidence:** A metric indicating the certainty level of the algorithm’s conclusion.
* **Regulatory Disclaimer:** This system is classified as **AI-Support Only**; it is intended to assist clinicians and should not be used as a standalone diagnostic tool without professional medical oversight.

### Technology Stack
| Component | Technology | Description |
| :--- | :--- | :--- |
| **Runtime & Logic** | **Python 3.9+** | Core backend environment. |
| **API Framework** | **FastAPI** | High-performance web framework. |
| **Server** | **Uvicorn** | ASGI server implementation. |
| **Machine Learning** | **TensorFlow / PyTorch** | Deep Learning libraries for inference. |
| **Frontend** | **HTML5 / CSS3 / JS** | Structural and stylistic markup. |
| **Animation** | **GSAP** | Advanced UI transition library. |

---

## 2. Prerequisites

Before deploying the application, ensure the host environment meets the following specifications:

1.  **Python Runtime:** Version **3.9** or higher.
    * *Verify (Linux/macOS):* `python3 --version`
    * *Verify (Windows):* `python --version`
2.  **Version Control:** Git (Recommended for repository management).

---

## 3. Installation Protocol

### Step 1: Repository Acquisition

Clone the source code to your local machine using Git or download the archive directly.

```bash
git clone https://github.com/baokhanh546123/API_LungCancer
```

### Step 2: Environment Configuration
It is strictly recommended to isolate dependency management by establishing a virtual environment **(venv)**.

**Windows**
```bash
cd API_LungCancer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/MacOS**
```bash
cd API_LungCancer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Execution Protocol
Initialize the server in the mode appropriate for your use case.

### Development Mode (Hot Reloading)
Recommended for debugging and active development cycles. The server will auto-restart upon code changes.
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode
Standard execution entry point for stable deployment.

**Windows**
```bash
python main.py
```

**Linux/MacOS**
```bash
python3 main.py
```

**Access Point**: Once initialized, the web interface is accessible via the local loopback address:

    http://127.0.0.1:8000/

## 5. Contribution & Feedback
This project is open for academic and technical contributions. We welcome feedback regarding diagnostic accuracy, UI responsiveness, or architectural improvements.
- Issue Tracking: Please submit pull requests or open issues via the [GitHub Repository](https://github.com/baokhanh546123/API_LungCancer).
- Direct Contact: Send feedback, bug reports, or collaboration proposals via email [here](mailto:tranphucdangkhanh1011dl@gmail.com).

Your contributions are a vital component in the refinement of this diagnostic tool.

## 6. Infomation Author

**Author**: Mrs.Binh Phuong Vo , Nhi Ngoc Thi Nguyen , Khanh Dang Phuc Tran.

**Affiliation**: Dalat University 

**Date of Release**: December 27, 2025

## 7. Disclaimer & License
This software is developed exclusively for **educational and research purposes**.
1. **Attribution**: Any form of reproduction, modification, or redistribution of the source code, in whole or in part, requires explicit permission from the author and proper citation of the source.
2. **Non-Commercial**: Use of this project for commercial or for-profit activities is strictly prohibited without prior written consent.

