# Fairfluence

A Python library for **profiling**, **influence-based quality assessment**, **AutoML performance evaluation**, and **fairness analysis** of tabular datasets from machine learning repositories like **OpenML**, **Kaggle**, and **Hugging Face**.

**Fairfluence** goes beyond standard profiling by training a model and using influence functions to identify which data points most affect the model’s predictions. This allows for precise fairness debugging and targeted quality analysis.

---

## 📦 Features

- Dataset ingestion from OpenML, Kaggle, Hugging Face
- Automatic profiling: missing values, outliers, imbalance, redundant features
- Model training and **influence score computation** per data point
- **Data quality checks** via PyOD, focused on high-influence records
- **Fairness analysis** using Fairlearn and sensitive attributes
- AutoML evaluation for performance analysis
- Visual report generation summarizing influence, quality, fairness, and performance
- Supports inputting a dataset URL, choosing a model, viewing analysis and downloading the report through an interactive UI

---

## 🔧 Project Structure

```
fairfluence/
├── data/                           # Downloaded datasets and local CSVs
├── notebooks/                      # Exploratory notebooks for analysis & prototyping
├── src/
│   ├── ingestion/                  # Dataset loaders (OpenML, Kaggle, HF)
│   │   ├── ingestorFactory.py      # 
│   │   ├── laoder.py               #
│   ├── preprocessing/             
│   │   ├── base.py                 # Shared preprocessing logic (e.g. encoders, scalers)
│   │   ├── quality.py              # Quality-specific preprocessing
│   │   └── fairness.py             # Fairness-specific preprocessing
│   ├── analysis/                   # Comparison, statistics & visualization
│   │   ├── stats.py                # Summary stats, distributions, correlations
│   │   ├── compare.py              # Before vs after comparisons
│   │   ├── visual.py               # Matplotlib/seaborn/plotly plots
│   ├── model/
│   │   ├── builder.py              # ModelBuilder class to manage model types
│   │   ├── registry.py             # Model registry or config-driven loader
│   │   ├── train.py                # Model training
│   ├── influence/                  # Influence score computation
│   │   ├── compute.py              #
│   ├── quality/
│   │   ├── no_influence.py         # Quality checks without influence
│   │   ├── with_influence.py       # Quality analysis with influence
│   │   └── clean.py                # Cleaning logic for data quality
│   ├── fairness/
│   │   ├── no_influence.py         # Fairness metrics (no influence)
│   │   ├── with_influence.py       # Fairness + influence debugging
│   │   └── clean.py                # Fairness-based filtering/repair
│   ├── utils/                      # Shared helper functions (logging, configuration)
├── outputs/                        # Generated reports, scores, visualizations
├── tests/                          # Unit tests for individual modules
├── app.py                          # Streamlit UI entry point
├── main.py                         # End-to-end CLI script to run the full pipeline
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
   Go to [kaggle.com](https://www.kaggle.com) and sign up or log in.

2. **Generate an API token**  
   - Click on your profile picture (top right) and select **“My Account”**.  
   - Scroll down to the **“API”** section and click **“Create New API Token”**.  
   - A file named `kaggle.json` will be downloaded to your computer.
  
3. **Create the `.kaggle` folder**  
   Open File Explorer and navigate to your user’s home directory, for example:  C:\Users<YourUserName>\
   If it doesn’t already exist, create a hidden folder called `.kaggle`: C:\Users\<YourUserName>\.kaggle
   Copy the downloaded kaggle.json into the new folder.


## ⚙️ Makefile Commands

| Command             | Description                                                |
|---------------------|------------------------------------------------------------|
| `make install`      | Set up virtual environment and install dependencies        |
| `make activate`     | Print the command to activate the environment              |
| `make jupyter-kernel` | Register Jupyter kernel as `fairfluence`          |
| `make remove-kernel`  | Unregister existing kernel (if needed)                  |
| `make clean`        | Delete the virtual environment folder                      |


---

## 📈 Example Usage (after setup)

### Run Streamlit App
```bash
streamlit run app.py  
```
### Run CLI Pipeline
```bash
python main.py
```
---

## 👥 Contributors

- **Aiman Al-Hazmi** 
- **Letian Wang** 
- **Luciano Duarte**  
- **Nicolas Korjahn**

---
