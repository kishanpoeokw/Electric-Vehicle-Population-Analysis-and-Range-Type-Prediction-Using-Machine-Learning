# ⚡ Electric Vehicle Population Analysis & Range/Type Prediction

An end-to-end Data Science and Machine Learning project analyzing **300,142+ electric vehicle registration records** from the Washington State Department of Licensing (`elc.csv`) to uncover fleet adoption trends, predict certified electric driving range (Regression), and classify vehicle powertrain types between BEV and PHEV (Classification)[cite: 32, 40].

---

## 📊 Key Exploratory Data Analysis (EDA) Findings

* **Fleet Breakdown:** **80.58%** (241,840) are Battery Electric Vehicles (BEVs), while **19.42%** (58,302) are Plug-in Hybrid Electric Vehicles (PHEVs)[cite: 42].
* **Market Dominance:** **Tesla** leads all manufacturers with **122,291 vehicles** (40.74% of the dataset), followed by Chevrolet (20,218), Ford (15,997), and Nissan (15,994)[cite: 33].
* **Adoption Growth:** Registrations accelerated sharply after 2020, peaking in **Model Year 2023** (61,487 vehicles)[cite: 34].

---

## 🛠️ Project Workflow

1. **Data Cleaning & Preprocessing:**
   * Dropped non-predictive ID and coordinate columns (`VIN (1-10)`, `DOL Vehicle ID`, `Vehicle Location`, `2020 GEOID`, `Postal Code`)[cite: 32].
   * Imputed missing `Legislative District` values using the statistical mode and removed remaining null rows, retaining **300,092 clean records**[cite: 32, 43].
   * Applied One-Hot Encoding (`pd.get_dummies`) to `Make` alongside `Model Year` and `Legislative District` (54 total features)[cite: 32, 43].
2. **Regression (Predicting Certified Electric Range):**
   * Filtered records with verified positive range (`Electric Range > 0`, **103,014 records**)[cite: 32, 43].
   * Compared **Simple Linear Regression** (`Model Year`) against **Multiple Linear Regression** (`Model Year`, `Legislative District`, and one-hot encoded `Make`)[cite: 32].
3. **Classification (Predicting BEV vs. PHEV):**
   * Trained and evaluated **Decision Tree** (`max_depth=5`) and **Random Forest** (`n_estimators=50`, `max_depth=8`) classifiers across all **300,092 cleaned records** using an 80:20 train-test split (`random_state=42`)[cite: 32, 43].

---

## 🤖 Model Performance

### 1. Regression Models (Test Set: 20,603 records)
| Model | Features Used | MAE (miles) | MSE | RMSE (miles) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** | `Model Year`, `Legislative District`, `Make` (54 features)[cite: 32, 43] | **31.10**[cite: 43] | **2,175.26**[cite: 43] | **46.64**[cite: 43] | **0.7666**[cite: 43] |
| **Simple Linear Regression** | `Model Year` (1 feature)[cite: 32] | 79.55[cite: 43] | 8,551.00[cite: 43] | 92.47[cite: 43] | 0.0824[cite: 43] |

### 2. Classification Models (Test Set: 60,019 records)
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **88.60%**[cite: 45] | **0.90**[cite: 45] | **0.72**[cite: 45] | **0.77**[cite: 45] |
| **Decision Tree Classifier** | 86.28%[cite: 45] | 0.87[cite: 45] | 0.66[cite: 45] | 0.71[cite: 45] |

* **Top Predictive Features (Random Forest):** `Make_TESLA`, `Make_TOYOTA`, `Make_JEEP`, `Make_CHRYSLER`, and `Model Year`[cite: 39].

---

## 💻 Tech Stack & Libraries

* **Language:** Python 3
* **Data Manipulation:** `pandas`, `numpy`[cite: 32]
* **Visualization:** `matplotlib`, `seaborn`[cite: 32]
* **Machine Learning:** `scikit-learn`[cite: 32]

---

