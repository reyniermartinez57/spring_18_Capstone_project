# Heart Disease Risk Classification & Conversational Interface

## Project Description
* This application is an end-to-end intelligent prediction layer that evaluates patient cardiovascular data to classify heart disease risks.
* It is built as a tool for clinicians, healthcare workers, or researchers looking to bridge complex numerical diagnostic tables with natural language summaries.
* Traditional tabular ML models require strict, clean arrays of numerical data, which are hard to input manually under tight time constraints. This application allows users to submit conversational patient notes in plain English, automatically maps them to appropriate feature arrays, computes a classification, and returns structural safety warnings.

---

## Setup Instructions

### 1. Installation
Clone the repository and install the requirements file containing pinned dependencies:
```bash
git clone [https://github.com/reyniermartinez57/spring_18_Capstone_project.git](https://github.com/reyniermartinez57/spring_18_Capstone_project.git)
cd spring_18_Capstone_project
pip install -r requirements.txt

2. Get the Data
The dataset used contains balanced patient observations for heart risk classification. Raw data files and local model binaries are systematically managed and excluded from Git using .gitignore to prevent repository bloat.
Usage Instructions
Start the Application Server: Run the FastAPI application using Uvicorn on your local host:

Bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
Interact with the Endpoint: Open a separate terminal shell and test the active pipeline using standard curl commands:

Valid Profile Run:

Bash
curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" -H "Content-Type: application/json" -d "{\"query\": \"Patient is a 65-year-old male with a resting blood pressure of 145 and cholesterol of 210. Heart rate reached 130.\"}"
Out-of-Scope Safe Intersection:

Bash
curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" -H "Content-Type: application/json" -d "{\"query\": \"Can you give me a recipe for chocolate chip cookies?\"}"
Architecture Overview
The system utilizes a two-layer hybrid pipeline:

Predictive Layer: A scikit-learn Random Forest classifier trained on a 28-feature input space, tracked and verified using an MLflow tracking directory.

Interface Layer: A FastAPI service acting as the conversational middleware. It utilizes rule-based validation routines to dynamically parse medical metrics from string queries, route the clean values to the underlying model binary, and inject the classification directly into localized structured JSON wrappers alongside rigorous medical disclaimers.

Results Summary
Best Selected Model: Random Forest Classifier (Loaded dynamically from the local MLflow metadata run records).

Metrics: Evaluated across multiple runs, demonstrating robust accuracy, high classification metrics, and balanced F1 scores on test splits.

Key Findings: Feature distributions show that advanced age combined with lower maximum heart rates heavily shift classifications toward a present (1) risk category, whereas normal metrics preserve an absent (0) prediction score.

Reflection
What I Learned: I discovered how to bridge unstructured natural language strings directly into fixed-shape tabular machine learning features, keeping inputs predictable without risking pipeline crashes.

What Was Challenging: Resolving local Windows container network limitations and external routing address blocks during deployment phases proved difficult. This required implementing resilient local fallback routing exceptions to safeguard high application availability.

What I Would Improve: With more time, I would replace the rule-based parsing mechanism with a lightweight, locally embedded tokenizer module to eliminate reliance on external cloud-hosted endpoints entirely.