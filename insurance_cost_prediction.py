"""
Medical Insurance Cost Prediction  -  Data Science Mini Project
Dataset : Kaggle - Medical Cost Personal Dataset (insurance.csv)
Run     : python insurance_cost_prediction.py
Outputs : images/*.png, outputs/metrics.json, outputs/model_comparison.csv,
          models/best_model.joblib
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42
DATA = "data/insurance.csv"

# ------------------------------------------------------------------ 1. LOAD
df = pd.read_csv(DATA)
info = {
    "rows": int(df.shape[0]), "cols": int(df.shape[1]),
    "missing": int(df.isna().sum().sum()),
    "duplicates": int(df.duplicated().sum()),
}
df = df.drop_duplicates().reset_index(drop=True)
print(df.head(), "\n", df.describe(), "\n", info)
df.describe().round(2).to_csv("outputs/summary_statistics.csv")

# ------------------------------------------------------------------ 2. EDA
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df.charges, kde=True, ax=ax[0], color="#2a6f97")
ax[0].set_title("Distribution of Charges")
sns.histplot(np.log1p(df.charges), kde=True, ax=ax[1], color="#e07a5f")
ax[1].set_title("Distribution of log(Charges)")
plt.tight_layout(); plt.savefig("images/01_charges_distribution.png", dpi=150); plt.close()

fig, ax = plt.subplots(2, 3, figsize=(14, 8))
sns.boxplot(x="smoker", y="charges", data=df, ax=ax[0, 0], palette="Set2")
ax[0, 0].set_title("Charges vs Smoking Status")
sns.boxplot(x="region", y="charges", data=df, ax=ax[0, 1], palette="Set2")
ax[0, 1].set_title("Charges vs Region")
sns.boxplot(x="sex", y="charges", data=df, ax=ax[0, 2], palette="Set2")
ax[0, 2].set_title("Charges vs Sex")
sns.scatterplot(x="age", y="charges", hue="smoker", data=df, ax=ax[1, 0], alpha=.7)
ax[1, 0].set_title("Age vs Charges")
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, ax=ax[1, 1], alpha=.7)
ax[1, 1].set_title("BMI vs Charges")
sns.boxplot(x="children", y="charges", data=df, ax=ax[1, 2], palette="Set2")
ax[1, 2].set_title("Charges vs Children")
plt.tight_layout(); plt.savefig("images/02_eda_relationships.png", dpi=150); plt.close()

enc = df.copy()
enc["sex"] = (enc.sex == "male").astype(int)
enc["smoker"] = (enc.smoker == "yes").astype(int)
enc = pd.get_dummies(enc, columns=["region"], drop_first=False, dtype=int)
plt.figure(figsize=(9, 7))
sns.heatmap(enc.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig("images/03_correlation_heatmap.png", dpi=150); plt.close()
corr_charges = enc.corr()["charges"].drop("charges").sort_values(ascending=False).round(3).to_dict()

# ------------------------------------------------------------------ 3. FEATURES
df["bmi_category"] = pd.cut(df.bmi, [0, 18.5, 25, 30, 100],
                            labels=["underweight", "normal", "overweight", "obese"])
df["smoker_obese"] = ((df.smoker == "yes") & (df.bmi >= 30)).astype(int)  # interaction

X = df.drop(columns="charges")
y = df["charges"]
num = ["age", "bmi", "children", "smoker_obese"]
cat = ["sex", "smoker", "region", "bmi_category"]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(drop="first"), cat)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ------------------------------------------------------------------ 4. MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0, max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
kf = KFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    cv = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    rows.append({"Model": name, "CV R2 (mean)": cv.mean(), "Test R2": r2_score(y_test, p),
                 "MAE": mean_absolute_error(y_test, p),
                 "RMSE": np.sqrt(mean_squared_error(y_test, p))})
    print(rows[-1])
results = pd.DataFrame(rows).sort_values("Test R2", ascending=False).round(4)
results.to_csv("outputs/model_comparison.csv", index=False)

# ------------------------------------------------------------------ 5. TUNING
grid = {"model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4]}
gs = GridSearchCV(Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=SEED))]),
                  grid, cv=kf, scoring="r2", n_jobs=-1)
gs.fit(X_train, y_train)
best = gs.best_estimator_
pred = best.predict(X_test)
final = {"best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
         "cv_r2": float(gs.best_score_),
         "test_r2": float(r2_score(y_test, pred)),
         "mae": float(mean_absolute_error(y_test, pred)),
         "rmse": float(np.sqrt(mean_squared_error(y_test, pred)))}
print(final)

# ------------------------------------------------------------------ 6. PLOTS
plt.figure(figsize=(8, 5))
order = results.sort_values("Test R2")
plt.barh(order.Model, order["Test R2"], color="#2a6f97")
plt.xlabel("Test R2"); plt.title("Model Comparison"); plt.tight_layout()
plt.savefig("images/04_model_comparison.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_test, pred, alpha=.6, color="#2a6f97")
lim = [y.min(), y.max()]; ax[0].plot(lim, lim, "r--")
ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted"); ax[0].set_title("Actual vs Predicted")
res = y_test - pred
ax[1].scatter(pred, res, alpha=.6, color="#e07a5f"); ax[1].axhline(0, color="k", ls="--")
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Residual"); ax[1].set_title("Residual Plot")
plt.tight_layout(); plt.savefig("images/05_actual_vs_predicted.png", dpi=150); plt.close()

pi = permutation_importance(best, X_test, y_test, n_repeats=15, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
plt.figure(figsize=(8, 5)); imp.plot.barh(color="#2a6f97")
plt.title("Permutation Feature Importance (Test Set)"); plt.xlabel("Drop in R2")
plt.tight_layout(); plt.savefig("images/06_feature_importance.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 7. SAVE
joblib.dump(best, "models/best_model.joblib")
final.update(info=info, correlations=corr_charges,
             importance=imp.sort_values(ascending=False).round(4).to_dict(),
             n_train=len(X_train), n_test=len(X_test),
             smoker_mean=df.groupby("smoker").charges.mean().round(0).to_dict(),
             age_corr=float(df.age.corr(df.charges)), bmi_corr=float(df.bmi.corr(df.charges)))
json.dump(final, open("outputs/metrics.json", "w"), indent=2, default=str)

# demo prediction
sample = pd.DataFrame([{"age": 35, "sex": "male", "bmi": 28.5, "children": 2,
                        "smoker": "no", "region": "southwest"}])
sample["bmi_category"] = pd.cut(sample.bmi, [0, 18.5, 25, 30, 100],
                                labels=["underweight", "normal", "overweight", "obese"])
sample["smoker_obese"] = ((sample.smoker == "yes") & (sample.bmi >= 30)).astype(int)
print("Demo prediction:", round(float(best.predict(sample)[0]), 2))
