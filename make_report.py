"""
Builds Medical_Insurance_Cost_Prediction_Report.pdf from the results produced by
insurance_cost_prediction.py (outputs/metrics.json, model_comparison.csv, images/).
Edit the STUDENT DETAILS block below, then run:  python make_report.py
"""
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, KeepTogether)

# ================= STUDENT DETAILS (EDIT THESE) =================
NAME = "Dhanush S M"
REG_NO = "71052510004"
CLASS = "II CSE (AIML)"
COURSE = "U23AMT01 - Data Science"
GITHUB = "https://github.com/chiy-aan/medical-insurance-cost-prediction"
# ================================================================

m = json.load(open("outputs/metrics.json"))
res = pd.read_csv("outputs/model_comparison.csv")
stats = pd.read_csv("outputs/summary_statistics.csv", index_col=0)

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], keepWithNext=1, fontSize=15, textColor=colors.HexColor("#1b4965"), spaceBefore=10)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], keepWithNext=1, fontSize=12, textColor=colors.HexColor("#1b4965"))
cap = ParagraphStyle("c", parent=ss["Italic"], fontSize=9, alignment=TA_CENTER, spaceAfter=10)
ctr = ParagraphStyle("ctr", parent=ss["Title"], alignment=TA_CENTER)
mid = ParagraphStyle("mid", parent=body, alignment=TA_CENTER)
S = []
P = lambda t, s=body: S.append(Paragraph(t, s))
def fig(path, w=15.5, caption=""):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    S.append(Image(path, width=w * cm, height=w * cm * ih / iw))
    if caption: P(caption, cap)
def table(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b4965")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf2f8")])]))
    S.append(KeepTogether([t])); S.append(Spacer(1, 8))

best = res.iloc[0]
imp = m["importance"]; top = list(imp.items())[:3]
sm = m["smoker_mean"]
ratio = sm["yes"] / sm["no"]

# ---------------- Title page
S.append(Spacer(1, 3 * cm))
P("Medical Insurance Cost Prediction", ctr)
P("Using Machine Learning Regression", mid)
S.append(Spacer(1, 1 * cm))
P("<b>Data Science Mini Project (Assignment 2)</b>", mid)
P(f"{COURSE}<br/>{CLASS}", mid)
S.append(Spacer(1, 1 * cm))
P(f"<b>Name:</b> {NAME}<br/><b>Register Number:</b> {REG_NO}", mid)
S.append(Spacer(1, 1 * cm))
P(f"<b>GitHub Repository:</b><br/><a href='{GITHUB}' color='blue'>{GITHUB}</a>", mid)
S.append(PageBreak())

# ---------------- 1 Abstract / Intro
P("1. Abstract", h1)
P("Health insurers must estimate how much each policyholder is likely to cost so that premiums are fair and "
  "sustainable. This project builds a supervised machine-learning regression system that predicts individual "
  "medical insurance charges from age, sex, BMI, number of children, smoking status and region. Six regression "
  "models were compared using 5-fold cross-validation, and Gradient Boosting was then tuned with grid search. "
  f"The final tuned Gradient Boosting model achieves R<super>2</super> = {m['test_r2']:.3f}, "
  f"MAE = {m['mae']:,.0f} and RMSE = {m['rmse']:,.0f} on unseen test data.")
P("2. Problem Statement and Objectives", h1)
P("<b>Problem:</b> Develop a machine-learning regression system to predict individual medical insurance charges "
  "using age, BMI, smoking status, region and other relevant factors.")
for t in ["Explore the dataset and understand which factors influence charges.",
          "Pre-process the data (encoding, scaling, feature engineering).",
          "Train and compare multiple regression models.",
          "Tune the best model and evaluate it with R<super>2</super>, MAE and RMSE.",
          "Identify the most important cost drivers and provide a reusable prediction pipeline."]:
    P("&bull; " + t)

# ---------------- 3 Dataset
P("3. Dataset Description", h1)
P("<b>Source:</b> Kaggle - Medical Cost Personal Dataset (insurance.csv). "
  f"It contains <b>{m['info']['rows']}</b> records and <b>{m['info']['cols']}</b> columns. "
  f"Missing values: <b>{m['info']['missing']}</b>; duplicate rows found: <b>{m['info']['duplicates']}</b> "
  "(duplicates are removed).")
table([["Feature", "Type", "Description"],
       ["age", "Numeric", "Age of primary beneficiary"],
       ["sex", "Categorical", "male / female"],
       ["bmi", "Numeric", "Body mass index (kg/m2)"],
       ["children", "Numeric", "Number of dependents covered"],
       ["smoker", "Categorical", "yes / no"],
       ["region", "Categorical", "NE, NW, SE, SW (US region)"],
       ["charges", "Target", "Individual medical costs billed by insurance"]],
      [3.5 * cm, 3 * cm, 9 * cm])
P("<b>Summary statistics</b>", h2)
sd = stats[["age", "bmi", "children", "charges"]].round(2)
table([["stat"] + list(sd.columns)] + [[i] + [f"{v:,.2f}" for v in r] for i, r in sd.iterrows()],
      [3 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm])

# ---------------- 4 Methodology
P("4. Methodology", h1)
P("The workflow followed was: <b>Data loading -> Cleaning -> EDA -> Feature engineering -> Train/test split "
  "-> Model training with cross-validation -> Hyper-parameter tuning -> Evaluation -> Model saving.</b>")
P("4.1 Exploratory Data Analysis", h2)
fig("images/01_charges_distribution.png", 15, "Figure 1: Charges are right-skewed; the log transform is closer to normal.")
fig("images/02_eda_relationships.png", 16, "Figure 2: Relationship of each feature with insurance charges.")
P(f"Average charges for smokers are about <b>{sm['yes']:,.0f}</b> versus <b>{sm['no']:,.0f}</b> for non-smokers "
  f"(roughly <b>{ratio:.1f}x</b>). Correlation of charges with age is {m['age_corr']:.2f} and with BMI is {m['bmi_corr']:.2f}.")
fig("images/03_correlation_heatmap.png", 11, "Figure 3: Correlation heatmap of encoded features.")
P("4.2 Pre-processing and Feature Engineering", h2)
for t in ["Duplicate rows removed; no imputation required if no missing values were reported above.",
          "Numeric features (age, bmi, children) standardised with StandardScaler.",
          "Categorical features (sex, smoker, region) one-hot encoded.",
          "<b>bmi_category</b> created from WHO ranges (underweight, normal, overweight, obese).",
          "<b>smoker_obese</b> interaction flag (smoker AND BMI >= 30) added, since the combined effect is larger than either alone.",
          f"Data split 80/20 (train = {m['n_train']}, test = {m['n_test']}) with random_state = 42. All steps live inside a scikit-learn Pipeline to prevent data leakage."]:
    P("&bull; " + t)
P("4.3 Models Used", h2)
P("Linear Regression, Ridge, Lasso, Decision Tree (max depth 5), Random Forest (300 trees) and Gradient Boosting. "
  "Each model was evaluated with 5-fold cross-validation on the training set and then on the held-out test set.")

# ---------------- 5 Results
P("5. Results and Discussion", h1)
P("5.1 Model Comparison", h2)
table([["Model", "CV R2", "Test R2", "MAE", "RMSE"]] +
      [[r.Model, f"{r['CV R2 (mean)']:.3f}", f"{r['Test R2']:.3f}", f"{r.MAE:,.0f}", f"{r.RMSE:,.0f}"] for _, r in res.iterrows()],
      [5 * cm, 2.5 * cm, 2.5 * cm, 2.8 * cm, 2.8 * cm])
fig("images/04_model_comparison.png", 11, "Figure 4: Test R<super>2</super> of all models.")
P(f"The best untuned model on the test set was <b>{best.Model}</b> (R<super>2</super> = {best['Test R2']:.3f}).")
P("5.2 Hyper-parameter Tuning", h2)
bp = m["best_params"]
P(f"Gradient Boosting was tuned with GridSearchCV (5-fold) over n_estimators, learning_rate and max_depth. "
  f"Best parameters: <b>n_estimators = {bp['n_estimators']}, learning_rate = {bp['learning_rate']}, "
  f"max_depth = {bp['max_depth']}</b> (CV R<super>2</super> = {m['cv_r2']:.3f}).")
table([["Metric", "Final tuned model (test set)"],
       ["R2 score", f"{m['test_r2']:.4f}"], ["MAE", f"{m['mae']:,.2f}"], ["RMSE", f"{m['rmse']:,.2f}"]],
      [5 * cm, 6 * cm])
fig("images/05_actual_vs_predicted.png", 15, "Figure 5: Actual vs predicted charges and residual plot.")
P("5.3 Feature Importance", h2)
fig("images/06_feature_importance.png", 11, "Figure 6: Permutation importance on the test set.")
P(f"The most influential features were <b>{top[0][0]}</b>, <b>{top[1][0]}</b> and <b>{top[2][0]}</b>. "
  "Features with the largest importance values drive the predicted cost, while those near zero (see Figure 6) "
  "contribute little.")

# ---------------- 6 Conclusion
P("6. Conclusion", h1)
P(f"A complete ML regression pipeline was built to predict medical insurance charges. The tuned Gradient Boosting "
  f"model explains about {m['test_r2']*100:.1f}% of the variance in unseen data with an average error of about "
  f"{m['mae']:,.0f} per person. The most important cost drivers were {top[0][0]}, {top[1][0]} and {top[2][0]}. "
  "The saved pipeline (models/best_model.joblib) can predict charges for new customers directly from raw inputs.")
P("<b>Limitations:</b> small dataset (about 1,300 rows); no medical history or claim data; errors are larger "
  "for very high-cost customers.")
P("<b>Future work:</b> try XGBoost/LightGBM, model log(charges), add SHAP explanations, and deploy as a Streamlit web app.")
P("7. References", h1)
for t in ["Kaggle - Medical Cost Personal Dataset (Miri Choi), https://www.kaggle.com/datasets/mirichoi0218/insurance",
          "Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, 2011.",
          "Friedman, J. H., Greedy Function Approximation: A Gradient Boosting Machine, Annals of Statistics, 2001."]:
    P("&bull; " + t)
P(f"<b>Source code:</b> <a href='{GITHUB}' color='blue'>{GITHUB}</a>")

def footer(c, d):
    c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawCentredString(A4[0] / 2, 1 * cm, f"Medical Insurance Cost Prediction  |  {REG_NO}  |  Page {d.page}")
SimpleDocTemplate("Medical_Insurance_Cost_Prediction_Report.pdf", pagesize=A4, leftMargin=2.2 * cm,
                  rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title="Medical Insurance Cost Prediction", author=NAME).build(S, onFirstPage=footer, onLaterPages=footer)
print("Report written.")
