# Heart Disease Prediction & Classification

A machine learning project to **predict and distinguish between different types of heart disease** using clinical patient data from the UCI Cleveland Heart Disease Dataset.

---

## 📁 Project Structure

```
mll/
├── data/                          # Dataset (auto-downloaded)
├── models/                        # Saved trained models (.pkl)
├── plots/                         # Generated visualizations
├── src/
│   ├── data_loader.py             # Load dataset from UCI
│   ├── preprocessing.py           # Clean, encode, and scale data
│   ├── train.py                   # Train 5 ML classifiers
│   ├── evaluate.py                # Metrics, confusion matrix, ROC
│   └── model_comparison.py       # Side-by-side model comparison
├── tests/
│   └── test_pipeline.py           # Automated unit tests
├── heart_disease_analysis.ipynb   # Full Jupyter notebook walkthrough
├── run.py                         # ▶ Main entry point (run everything)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**UCI Cleveland Heart Disease Dataset**
- **303 patients**, 14 clinical features (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- **Target classes:**
  - `0` → Healthy (no heart disease)
  - `1` → Type 1 Heart Disease (mild)
  - `2` → Type 2 Heart Disease
  - `3` → Type 3 Heart Disease
  - `4` → Type 4 Heart Disease (severe)

---

## 🤖 Models Trained

| Model                | Notes                                  |
|---------------------|----------------------------------------|
| Logistic Regression  | Multinomial, strong baseline           |
| Random Forest        | 200 trees, handles non-linearity well  |
| SVM                  | RBF kernel, good for small datasets    |
| K-Nearest Neighbors  | Distance-weighted                      |
| XGBoost              | Gradient boosting, often top performer |

---

## 🚀 Quick Start

### 1. Install dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the full pipeline
```powershell
python run.py
```
This will:
- Download the dataset automatically
- Train all 5 models
- Save models to `models/`
- Generate all plots to `plots/`
- Print classification reports

### 3. Open the Jupyter Notebook (detailed walkthrough)
```powershell
jupyter notebook heart_disease_analysis.ipynb
```

### 4. Run unit tests
```powershell
python -m pytest tests/test_pipeline.py -v
```

---

## 📈 Generated Plots

After running, check the `plots/` directory for:

| File | Description |
|------|-------------|
| `cm_*.png` | Confusion matrices per model |
| `roc_*.png` | Multi-class ROC curves per model |
| `comparison_accuracy.png` | Accuracy bar chart |
| `comparison_training_time.png` | Training time comparison |
| `comparison_accuracy_vs_f1.png` | Accuracy vs Macro-F1 |

---

## 🔑 Key Clinical Features

| Feature | Description |
|---------|-------------|
| `age` | Patient age (years) |
| `sex` | Sex (1=male, 0=female) |
| `cp` | Chest pain type (0=typical angina → 3=asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `thalach` | Maximum heart rate achieved |
| `oldpeak` | ST depression induced by exercise |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia (1=normal, 2=fixed defect, 3=reversible defect) |

---

## 🎓 Learning Objectives

- Multi-class classification on real medical data
- Handling missing values and feature scaling
- Training and comparing multiple classifiers
- Visualizing model performance with confusion matrices and ROC curves
- Saving/loading trained ML models with `joblib`
