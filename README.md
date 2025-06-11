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

---

## 🔧 Project Structure

```
fairfluence/
├── data/                   # Downloaded datasets and local CSVs
├── notebooks/              # Exploratory notebooks for analysis & prototyping
├── src/
│   ├── ingestion/          # Dataset loaders (OpenML, Kaggle, HF)
│   ├── model/              # Model training utilities (e.g., train_model.py)
│   ├── influence/          # Influence score computation
│   ├── quality/            # Data quality assessment logic
│   ├── fairness/           # Fairness analysis and bias detection
│   ├── automl/             # Integration with automated ML workflows
│   ├── reports/            # Visual and text-based reporting utilities
│   ├── utils/              # Shared helper functions (logging, configuration, report generation)
│   └── main.py             # End-to-end CLI script to run the full pipeline
├── outputs/                # Generated reports, scores, visualizations
├── tests/                  # Unit tests for individual modules
├── requirements.txt        # Project dependencies
├── Makefile                # Build and setup commands
└── README.md               # Project documentation
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
