# Fairfluence  
*Influence-Based Data Quality and Fairness Analysis for Tabular ML Datasets*

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit App](https://img.shields.io/badge/Run%20via-Streamlit-blueviolet)


Fairfluence is a modular Python library for **data profiling**, **outlier detection** and **fairness analysis** in tabular machine learning datasets. It combines classical statistical techniques with **influence functions** to help identify data points that degrade model performance or introduce bias — with an interactive UI and exportable reports.

---

## 📦 Features

- **Dataset ingestion** from Kaggle, OpenML, and Hugging Face
- **Automated data profiling**: missing values, duplicates, class balance
- Dual-pipeline outlier detection: **statistical (Mahalanobis)** vs. **influence-based**
- **Fairness analysis** using:
  - Traditional metrics (DPD, EOD, PPV)
  - Influence-based **pattern analysis**
- Generates visual reports summarizing **influence**, **data quality**, **fairness** and **model performance**
- Interactive UI that lets you input a **dataset URL (Kaggle, OpenML, or Hugging Face)**, select a model, run the analysis and download the full report
---


## ▶️ How It Works

Fairfluence computes **influence scores** using convex models like logistic regression or SVM. These scores indicate how much each training point affects model predictions.

You can then:
- Flag and remove high-influence outliers
- Compare statistical vs. influence-based removal effects
- Discover fairness issues in small but high-impact subgroups
- Generate a PDF report and visual summaries

---

## 🗂️ Project Structure

```
fairfluence/
├── data/                           # Downloaded datasets and local CSVs
├── src/
│   ├── ingestion/                  # Dataset loaders (OpenML, Kaggle, HF)
│   │   ├── ingestorFactory.py      # Automatically recognises source of data, loads it and returns raw data
│   │   ├── loader.py               # Initializes ingestorFactory
│   │   ├── test_ingestion.py       # Jupyternotebook for testing ingestion functionalities
│   ├── preprocessing/             
│   │   ├── preprocessing.py        # complete preprocessing logic for data quality and fairness analysis
│   │   ├── preprocessing_dict.py   # Dictionary with all search words for identifying sensitive columns
│   │   ├── preprocessing_test.py   # Jupyternotebook for testing preprocessing functionalities
│   ├── profiling/                  # Statistics
│   │   ├── stats.py                # Summary stats, distributions
│   ├── model/
│   │   ├── registry.py             # Model registry or config-driven loader
│   │   ├── train.py                # Model training
│   ├── influence/                  # Influence score computation
│   │   ├── base.py                 #
│   │   ├── logistic_influence.py   #
│   ├── quality/
│   │   ├── no_influence.py         # Outlier calculation without influence
│   │   ├── with_influence.py       # Outlier estimation with influence
│   │   ├── clean.py                # Outlier removal and summary for data quality
│   │   └── compare.py              # Outlier analysis for data quality
│   ├── fairness/
│   │   ├── no_influence.py         # Fairness metrics (no influence)
│   │   ├── with_influence.py       # Fairness + influence debugging
│   ├── utils/                      # Shared helper functions (output-related helpers, logging, configuration)
├── outputs/                        # Generated reports, scores, visualizations
├── tests/                          # Unit tests for individual modules
├── app.py                          # Streamlit UI entry point
├── main.py                         # CLI pipeline runner
├── requirements.txt                # Project dependencies
├── Makefile                        # Build and setup commands
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aimanalhazmi/fairfluence.git
cd fairfluence
```

### 2. Set up the environment using `make`
```bash
make
```

This will:
- Create a virtual environment in `.venv/`
- Install all required packages from `requirements.txt`

### 3. Activate the environment
```bash
source .venv/bin/activate
```

### 4. Register Jupyter kernel (optional)
```bash
make jupyter-kernel
```

You can now select **fairfluence** as a kernel in Jupyter Notebook/Lab.

### 5. Use Kaggle API (optional)
To be able to use datasets from Kaggle, and as thus the API from Kaggle, you need an API token. Follow these steps:
1. **Create or log in to your Kaggle account**  
   Visit [kaggle.com](https://www.kaggle.com) and sign up or log in.

2. **Generate an API token**  
   - Click on your profile picture (top right) and select **“My Account”**.  
   - Scroll down to the **“API”** section and click **“Create New API Token”**.  
   - A file named `kaggle.json` will be downloaded to your computer.
  
3. **Create the `.kaggle` folder and add the token**
Move the downloaded kaggle.json file to the .kaggle folder inside your home directory:
   - **Windows**: C:\\Users\\<YourUserName>\\.kaggle\\kaggle.json
   - **Linux/macOS**: ~/.kaggle/kaggle.json

If the .kaggle folder doesn't exist, create it manually.
On Linux/macOS, run this to set proper permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Tipp
- If you need a dataset for a quick start we can recommend: https://huggingface.co/datasets/scikit-learn/adult-census-income with the target column "income"

## ⚙️ Makefile Commands

| Command             | Description                                                |
|---------------------|------------------------------------------------------------|
| `make install`      | Set up virtual environment and install dependencies        |
| `make activate`     | Print the command to activate the environment              |
| `make jupyter-kernel` | Register Jupyter kernel as `fairfluence`          |
| `make remove-kernel`  | Unregister existing kernel (if needed)                  |
| `make clean`        | Delete the virtual environment folder                      |


---

## How to Run

Fairfluence can be used in three ways:

### Streamlit App (Optional GUI)
```bash
streamlit run app.py
```
This launches a web interface where you can:

- Input a dataset URL (Kaggle, OpenML, Hugging Face)
- Choose model and target column 
- Visualize profiling and analysis results interactively 
- Download a detailed PDF report

---
### Manual Mode (Interactive CLI)
```bash
python main.py
```
You’ll be prompted to:

- Enter a dataset URL (Kaggle, OpenML, or Hugging Face)
- Choose a target column (or auto-select one)
- Select a model (e.g., Logistic Regression, Support Vector Machine)

The system will:

- Run data profiling, outlier detection, and fairness analysis
- Generate a visual PDF report at:  
  `outputs/final_report.pdf`

---

### Auto Mode (Batch via JSON Config)

#### 1. Create a config file (e.g., `datasets.json`)

```json
{
  "https://huggingface.co/datasets/scikit-learn/adult-census-income": {
     "platform": "huggingface.co",
      "target_column": "income",
      "no_dataset": 1,
      "model_type": "Logistic Regression",
      "timeout": 3600
  },
  "https://www.kaggle.com/datasets/...": {
    "target_column": "target",
    "model_type": "Support Vector Machine"
  }
}
```

#### 2. Run Fairfluence in auto mode

```bash
python main.py --mode auto --datasets datasets.json
```

This will:
- Load all datasets from the JSON
- Run both pipelines (quality and fairness)
- Generate individual reports in `outputs/` for each dataset


---

## Evaluation
We ran Fairfluence’s full pipeline on 100 tabular datasets.
### Data‑Quality Results
- **Statistical trimming** lead to highest F₁ value in **42.4%** of datasets.  
- **Influence‑based trimming** lead to highest F₁ value in **19.2%** of datasets.  
- In **38.4%** of cases both methods performed equally.
- The difference in the resulting F1 value lies on average within **±1.6%** of the baseline.  
- Overlap between the two outlier sets was only **0.61%**, confirming they capture different phenomena.

### Fairness Results
- **Classical parity metrics** (DPD, EOD, PPV) flagged **75%** of sensitive attributes as unfair.  
- **Influence‑driven pattern mining** found **52%** of top‑*k* subgroups to be unfair—even outside classical sensitive columns.  
- Only **15%** of those high‑impact patterns involved known sensitive attributes.  
- Cohen’s _d_ agreed with the influence labels **52%** of the time (just **22%** flagged by Cohen’s _d_), highlighting influence mining’s greater sensitivity to small‑slice biases.

## Discussion & Future Work

Influence functions add a complementary, model‑aware lens that uncovers harmful points and subtle bias patterns classical methods miss. The main challenge remains choosing robust thresholds across diverse datasets. Future work should focus on:
1. Faster influence‐score approximations  
2. Data‐adaptive threshold selection  
3. Expanded visualization and reporting features  

---

## 👥 Contributors

- **Aiman Al-Hazmi** 
- **Letian Wang** 
- **Luciano Duarte**  
- **Nicolas Korjahn**

---
