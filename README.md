# Heart Disease Classification & Natural Language Interface

## Project Description
This application is an end-to-end intelligent system designed to help assess cardiovascular health risks. It combines a trained scikit-learn Random Forest classifier with an advanced natural language interface built on FastAPI. Users can interact with the system using plain English health descriptions. The application dynamically parses clinical indicators, runs inference through the machine learning model, and returns a structured risk prediction along with clear explanations and necessary medical caveats.

## Architecture Overview
The system relies on two tightly integrated layers:
1. **Predictive Engine:** A 28-feature classification model tracked and managed using MLflow artifacts.
2. **Interface Layer:** A FastAPI conversational endpoint (`/chat`) that evaluates inputs, leverages keywords for safe local edge-case filtering, extracts features dynamically, and formats structured JSON responses.

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/reyniermartinez57/spring_18_Capstone_project.git](https://github.com/reyniermartinez57/spring_18_Capstone_project.git)
cd spring_18_Capstone_project