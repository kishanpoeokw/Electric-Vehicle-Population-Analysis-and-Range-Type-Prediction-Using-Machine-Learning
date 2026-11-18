# EV Population Project (Washington State)
# predicting electric range and BEV/PHEV type

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


# ---------------- UNIT 1 ----------------
# Objective 1: load, inspect and EDA

df = pd.read_csv("elc.csv")   # csv should be in the same folder as this file
print(df.head())
print(df.shape)
df.info()
print(df.describe())
print(df['Electric Vehicle Type'].value_counts())

# bev vs phev
df['Electric Vehicle Type'].value_counts().plot(kind='bar')
plt.title("BEV vs PHEV")
plt.ylabel("Count")
plt.show()

# top 10 makes
df['Make'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 EV Makes")
plt.ylabel("Count")
plt.show()

# evs per model year
df['Model Year'].value_counts().sort_index().plot(kind='line')
plt.title("EVs per Model Year")
plt.xlabel("Model Year")
plt.ylabel("Count")
plt.show()

print(df.groupby('Electric Vehicle Type')['Electric Range'].mean())


# Objective 2: cleaning and encoding

print(df.isnull().sum())

# these columns are not needed
df = df.drop(['VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location', '2020 GEOID', 'Postal Code'], axis=1)

df['Legislative District'] = df['Legislative District'].fillna(df['Legislative District'].mode()[0])
df = df.dropna()
print(df.isnull().sum())
print(df.shape)

# one hot encoding for make
x = pd.get_dummies(df[['Model Year', 'Legislative District', 'Make']], columns=['Make'])
print(x.head())


# ---------------- UNIT 2 ----------------
# electric range is 0 for a lot of cars (not researched),
# so for regression only taking rows where range is known
reg_df = df[df['Electric Range'] > 0]
print(reg_df.shape)

# Objective 3: simple linear regression (Model Year -> Electric Range)

x1 = reg_df[['Model Year']]
y1 = reg_df['Electric Range']
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

slr = LinearRegression()
slr.fit(x1_train, y1_train)
y1_pred = slr.predict(x1_test)

print("Simple Linear Regression")
print("MAE :", mean_absolute_error(y1_test, y1_pred))
print("MSE :", mean_squared_error(y1_test, y1_pred))
print("RMSE:", np.sqrt(mean_squared_error(y1_test, y1_pred)))
print("R2  :", r2_score(y1_test, y1_pred))

# graph: data points with the regression line
plt.scatter(x1_test['Model Year'], y1_test, alpha=0.3)
plt.plot(x1_test['Model Year'], y1_pred, color='red')
plt.title("Simple Linear Regression")
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.show()


# Objective 4: multiple linear regression

x2 = x.loc[reg_df.index]   # same rows as reg_df
y2 = reg_df['Electric Range']
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=42)

mlr = LinearRegression()
mlr.fit(x2_train, y2_train)
y2_pred = mlr.predict(x2_test)

print("Multiple Linear Regression")
print("MAE :", mean_absolute_error(y2_test, y2_pred))
print("MSE :", mean_squared_error(y2_test, y2_pred))
print("RMSE:", np.sqrt(mean_squared_error(y2_test, y2_pred)))
print("R2  :", r2_score(y2_test, y2_pred))

# graph: actual vs predicted (red line = perfect prediction)
plt.scatter(y2_test, y2_pred, alpha=0.3)
plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], color='red')
plt.title("Multiple Linear Regression")
plt.xlabel("Actual Range")
plt.ylabel("Predicted Range")
plt.show()

# comparing simple vs multiple
r2_scores = [r2_score(y1_test, y1_pred), r2_score(y2_test, y2_pred)]
rmse_scores = [np.sqrt(mean_squared_error(y1_test, y1_pred)), np.sqrt(mean_squared_error(y2_test, y2_pred))]

plt.subplot(1, 2, 1)
plt.bar(["Simple", "Multiple"], r2_scores)
plt.title("R2 Score")
plt.subplot(1, 2, 2)
plt.bar(["Simple", "Multiple"], rmse_scores)
plt.title("RMSE")
plt.show()


# ---------------- UNIT 3 ----------------
# Objective 5: decision tree and random forest (BEV / PHEV)

y = df['Electric Vehicle Type']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# decision tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# random forest
rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# comparing the two models
models = ["Decision Tree", "Random Forest"]
preds = [y_pred_dt, y_pred_rf]
acc = [accuracy_score(y_test, p) for p in preds]
prec = [precision_score(y_test, p, average='macro') for p in preds]
rec = [recall_score(y_test, p, average='macro') for p in preds]
f1 = [f1_score(y_test, p, average='macro') for p in preds]

pos = np.arange(2)
plt.bar(pos - 0.3, acc, 0.2, label="Accuracy")
plt.bar(pos - 0.1, prec, 0.2, label="Precision")
plt.bar(pos + 0.1, rec, 0.2, label="Recall")
plt.bar(pos + 0.3, f1, 0.2, label="F1")
plt.xticks(pos, models)
plt.title("Decision Tree vs Random Forest")
plt.legend()
plt.show()

# top 10 important features in random forest
imp = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False).head(10)
imp.plot(kind='bar')
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.show()

# conclusion
# - multiple regression (R2 ~0.77) did way better than simple regression (R2 ~0.08)
# - random forest (~89%) slightly better than decision tree (~86%)
# - both models miss a lot of PHEVs because there are less of them in the data
