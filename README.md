# ML Data Profiler

A Python library for **profiling**, **quality assessment**, **AutoML performance evaluation**, and **fairness analysis** of datasets from machine learning repositories like **OpenML**, **Kaggle**, and **Hugging Face**.

---

## 📦 Features

- Automatic data profiling: missing values, imbalance, outliers, constant/correlated columns
- AutoML support with performance metrics
- Fairness analysis using sensitive attributes
- Visual report generation for each dataset
- Works with dataset URLs from OpenML, Hugging Face, and Kaggle

---

## 🔧 Project Structure

```
ml_data_profiler/
├── ingestion/         # Dataset download and preprocessing
├── profiling/         # Data quality analysis (missing values, outliers, etc.)
├── automl/            # AutoML training & evaluation module
├── fairness/          # Fairness metrics and bias detection
├── reports/           # Report generation (Plots, summary)
├── utils/             # Shared helper functions
├── main.py            # Runs the full pipeline
├── requirements.txt   # Python dependencies
└── Makefile           # Build and setup commands
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aimanalhazmi/ml-data-profiler.git
cd ml-data-profiler
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

You can now select **ml-data-profiler** as a kernel in Jupyter Notebook/Lab.


## ⚙️ Makefile Commands

| Command             | Description                                                |
|---------------------|------------------------------------------------------------|
| `make install`      | Set up virtual environment and install dependencies        |
| `make activate`     | Print the command to activate the environment              |
| `make jupyter-kernel` | Register Jupyter kernel as `ml-data-profiler`          |
| `make remove-kernel`  | Unregister existing kernel (if needed)                  |
| `make clean`        | Delete the virtual environment folder                      |


---

## 📈 Example Usage (after setup)

```bash
python main.py
```

To get Influence(with Logistic Regression):
```
influencer = LogisticInfluence(mode, X_train, y_train)
influences = Influencer.get_influence(X_test[0], y_test[0])
```
---

## 👥 Contributors

- **Aiman Al-Hazmi** 
- **Letian Wang** 
- **3**  
- **4**

---
