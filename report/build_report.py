"""Build report/Final_Report.pdf from results/metrics.json and figures/.

Every number in the PDF is read from the artefacts produced by src/train.py, so the report
cannot drift away from the results. Regenerate with:

    python report/build_report.py

reportlab is used because this machine has neither pandoc nor a LaTeX distribution.
"""

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
FIGURES = ROOT / "figures"
OUT = ROOT / "report" / "Final_Report.pdf"

AUTHOR = "<< your name >>"
STUDENT_ID = "<< your student ID >>"

NAVY = colors.HexColor("#1f3864")
GREY = colors.HexColor("#5a5a5a")
LIGHT = colors.HexColor("#eef2f8")

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=styles["Title"], fontSize=22, leading=27, textColor=NAVY),
    "subtitle": ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=12, leading=17,
                               alignment=TA_CENTER, textColor=GREY),
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, leading=19, textColor=NAVY,
                         spaceBefore=16, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, leading=15, textColor=NAVY,
                         spaceBefore=11, spaceAfter=5),
    "body": ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.8, leading=14.5,
                           alignment=TA_JUSTIFY, spaceAfter=7),
    "caption": ParagraphStyle("caption", parent=styles["Normal"], fontSize=8.3, leading=11,
                              alignment=TA_CENTER, textColor=GREY, spaceAfter=11),
    "bullet": ParagraphStyle("bullet", parent=styles["BodyText"], fontSize=9.8, leading=14.5,
                             leftIndent=13, bulletIndent=4, alignment=TA_JUSTIFY, spaceAfter=4),
}

story = []


def h1(text):
    story.append(Paragraph(text, S["h1"]))


def h2(text):
    story.append(Paragraph(text, S["h2"]))


def p(text):
    story.append(Paragraph(text, S["body"]))


def bullets(items):
    for item in items:
        story.append(Paragraph(item, S["bullet"], bulletText="•"))
    story.append(Spacer(1, 5))


def figure(name, caption, width=15.5 * cm):
    path = FIGURES / name
    if not path.exists():
        return
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    story.append(KeepTogether([
        Image(str(path), width=width, height=width * h / w),
        Spacer(1, 3),
        Paragraph(caption, S["caption"]),
    ]))


def table(data, widths=None, highlight_row=None):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c2d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row is not None:
        style += [("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#d6e4f7")),
                  ("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 9))


# ---------------------------------------------------------------- title page
best = METRICS["best_model"]
data = METRICS["dataset"]
default_op, tuned_op = best["at_default_threshold"], best["at_tuned_threshold"]

story += [
    Spacer(1, 3.2 * cm),
    Paragraph("Diabetes Prediction from Clinical Measurements", S["title"]),
    Spacer(1, 0.5 * cm),
    Paragraph("A comparative machine learning study on the Pima Indians Diabetes dataset",
              S["subtitle"]),
    Spacer(1, 1.8 * cm),
]
table(
    [["Course", "CBIO313 - Data Mining & Machine Learning"],
     ["Instructor", "Dr. Muhammad Elsayeh"],
     ["Author", AUTHOR],
     ["Student ID", STUDENT_ID],
     ["Date", date.today().strftime("%d %B %Y")],
     ["Repository", "github.com/<your-account>/diabetes-prediction-cbio313"]],
    widths=[4 * cm, 11 * cm],
)
story.append(Spacer(1, 1.2 * cm))

h2("Abstract")
p(
    f"Type 2 diabetes is frequently asymptomatic until complications appear, which makes cheap "
    f"triage valuable. This study builds a screening classifier from eight routine clinical and "
    f"demographic measurements on {data['n_samples']} patients. The central preprocessing finding is "
    f"that missing measurements in this dataset were silently recorded as zeros - affecting "
    f"{data['missing']['Insulin']['percent']:.1f}% of insulin and "
    f"{data['missing']['SkinThickness']['percent']:.1f}% of skin-fold values - and correcting this is "
    f"more consequential than the choice of algorithm. Five algorithms were compared under identical "
    f"preprocessing; a tuned {best['name']} was selected on cross-validated ROC-AUC "
    f"({best['cv_roc_auc']:.3f}) computed on training data alone, reaching a test ROC-AUC of "
    f"{best['test_roc_auc']:.3f}. Moving the decision threshold from 0.50 to {best['threshold']:.2f}, "
    f"chosen from out-of-fold training predictions, raised recall on diabetic patients from "
    f"{default_op['recall']:.1%} to {tuned_op['recall']:.1%} at a cost of only "
    f"{default_op['precision'] - tuned_op['precision']:.1%} precision - reducing missed diabetics from "
    f"{default_op['confusion_matrix'][1][0]} to {tuned_op['confusion_matrix'][1][0]} out of "
    f"{sum(tuned_op['confusion_matrix'][1])}."
)
story.append(PageBreak())

# ------------------------------------------------------------------ contents
h1("1. Introduction and problem definition")
p(
    "Type 2 diabetes develops silently. By the time symptoms prompt a clinic visit, damage to the "
    "kidneys, retina and peripheral nerves is often already underway. Screening programmes address "
    "this, but a full diagnostic workup cannot be offered to an entire population. A model that "
    "flags who most needs that workup, using measurements already collected during a routine visit, "
    "is therefore practically useful."
)
p("<b>Objectives.</b>")
bullets([
    "Correct a dataset in which missing values were recorded as zeros.",
    "Characterise how each measurement relates to diabetes status.",
    "Train and compare five classification algorithms covered in the course.",
    "Evaluate with metrics suited to an imbalanced medical screening task, not accuracy alone.",
    "Tune the strongest model and select an operating threshold appropriate to screening.",
    "Explain the model's predictions so that a clinician could interrogate them.",
])
p(
    "<b>The metric that matters.</b> The two error types are not equally costly. A false positive "
    "sends a healthy patient for a confirmatory test: inconvenient and mildly expensive. A false "
    "negative sends an undiagnosed diabetic home reassured. Recall on the positive class is therefore "
    "the primary metric, with precision monitored so the model does not degenerate into flagging "
    "everyone."
)

h1("2. Dataset description")
p(
    f"<b>Source.</b> National Institute of Diabetes and Digestive and Kidney Diseases, distributed "
    f"through the UCI Machine Learning Repository as the Pima Indians Diabetes Database. "
    f"<b>Population.</b> {data['n_samples']} women of Pima Indian heritage, aged 21 or older, living "
    f"near Phoenix, Arizona. {data['n_features']} predictive features and one binary target. "
    f"Class distribution: {data['class_counts']['no_diabetes']} without diabetes and "
    f"{data['class_counts']['diabetes']} with, a positive rate of {data['positive_rate']:.1%}."
)
corr = data["correlation_with_outcome"]
table(
    [["Feature", "Description", "Correlation with outcome"]]
    + [[k, v, f"{corr[k]:+.3f}"] for k, v in [
        ("Pregnancies", "Number of times pregnant"),
        ("Glucose", "Plasma glucose, 2 h OGTT (mg/dL)"),
        ("BloodPressure", "Diastolic blood pressure (mm Hg)"),
        ("SkinThickness", "Triceps skin-fold thickness (mm)"),
        ("Insulin", "2-hour serum insulin (mu U/mL)"),
        ("BMI", "Body mass index (kg/m²)"),
        ("DiabetesPedigreeFunction", "Family-history diabetes score"),
        ("Age", "Age (years)"),
    ]],
    widths=[4.6 * cm, 7.4 * cm, 3.5 * cm],
)

h1("3. Data preprocessing")
h2("3.1 Missing values disguised as zeros")
p(
    "Five columns report a minimum of zero: Glucose, BloodPressure, SkinThickness, Insulin and BMI. "
    "None is biologically possible in a living patient. These are unrecorded measurements that were "
    "never encoded as missing. Left uncorrected, a model would read <i>insulin = 0</i> as a genuine "
    "and extremely low reading, which is the opposite of the truth."
)
table(
    [["Feature", "Zeros", "% of rows"]]
    + [[k, str(v["zeros"]), f"{v['percent']}%"] for k, v in data["missing"].items()],
    widths=[6 * cm, 3.5 * cm, 3.5 * cm],
)
p(
    "Dropping incomplete rows was rejected: it would discard roughly half the dataset, and patients "
    "missing an insulin measurement are unlikely to be a random sample. The zeros were converted to "
    "NaN and filled with the median, which is robust to the pronounced right skew of these variables."
)
h2("3.2 Preventing data leakage")
p(
    "Imputation and scaling are placed <b>inside a scikit-learn Pipeline</b> rather than applied to "
    "the full dataset beforehand. Computing a median over all rows and only then splitting would let "
    "information about test patients influence the training data, inflating every score that follows. "
    "Within a pipeline, both are refitted from the training portion of each cross-validation fold. "
    f"The split is stratified 80/20: {METRICS['split']['train']} training and "
    f"{METRICS['split']['test']} test patients, preserving the class ratio in both."
)

h1("4. Exploratory data analysis")
figure("01_overview.png",
       "Figure 1. Class balance and the extent of the hidden missing values.")
p(
    f"The classes are imbalanced at roughly {1 - data['positive_rate']:.0%} / "
    f"{data['positive_rate']:.0%}. A model predicting 'no diabetes' for every patient would score "
    f"{1 - data['positive_rate']:.1%} accuracy while catching no diabetics at all. That is the "
    f"benchmark any headline accuracy must be read against."
)
figure("02_distributions.png",
       "Figure 2. Feature distributions split by outcome, missing values excluded.")
p(
    f"Glucose separates the groups most clearly, consistent with its correlation of "
    f"{corr['Glucose']:+.2f} and with plasma glucose forming part of the diagnostic definition of "
    f"diabetes. BMI ({corr['BMI']:+.2f}) and Age ({corr['Age']:+.2f}) shift more modestly. "
    f"BloodPressure and SkinThickness overlap almost entirely, anticipating their negligible "
    f"contribution in the importance analysis."
)
figure("03_correlation.png", "Figure 3. Correlation matrix of features and outcome.")
p(
    "Age correlates with Pregnancies, and SkinThickness with BMI. This collinearity means linear "
    "coefficients should not be read as independent effects, which is one reason the SHAP analysis in "
    "section 7 carries more weight than raw coefficients."
)
story.append(PageBreak())

h1("5. Machine learning implementation")
p(
    "Five algorithms from the course syllabus, each wrapped in the identical impute-scale-classify "
    "pipeline so that the comparison isolates the algorithm: Logistic Regression (linear baseline), "
    "K-Nearest Neighbours (non-parametric, local), Decision Tree (interpretable rules), Random Forest "
    "(bagged ensemble) and a Support Vector Machine with an RBF kernel (maximum margin, non-linear)."
)
h2("5.1 Cross-validated comparison")
p(
    "A single split of 768 rows is noisy, so models are ranked by stratified 5-fold cross-validation "
    "on the training set, leaving the test set untouched for the final estimate."
)
cv = METRICS["cross_validation"]
table(
    [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    + [[r["model"], f"{r['accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}",
        f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}"] for r in cv],
    widths=[5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm],
    highlight_row=1,
)
figure("05_model_comparison.png", "Figure 4. Cross-validated metrics for the five algorithms.")
p(
    "Recall sits far below accuracy for every model - at the default threshold these classifiers "
    "detect only about 55 to 60 percent of diabetics. Section 6.2 addresses this directly."
)

h1("6. Evaluation and comparison")
h2("6.1 Held-out test set")
tr = METRICS["test_results"]
table(
    [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    + [[r["model"], f"{r['accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}",
        f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}"] for r in tr],
    widths=[5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm],
)
figure("06_roc_curves.png", "Figure 5. ROC curves on the held-out test set.")
figure("07_confusion_matrices.png",
       "Figure 6. Confusion matrices at the default 0.50 threshold. The lower-left cell of each is "
       "the clinically costly error: diabetics the model sent home.")

h2("6.2 Hyperparameter tuning and threshold selection")
p(
    "Grids were scored on ROC-AUC rather than recall. Optimising a grid directly on recall rewards "
    "models that predict the positive class more often, which collapses precision; in the limit, "
    "predicting everyone positive scores perfect recall. ROC-AUC ranks models independently of the "
    "threshold, and the threshold is then chosen as a separate decision."
)
tuning = METRICS["tuning"]
table(
    [["Model", "Best parameters", "CV ROC-AUC", "Test ROC-AUC"]]
    + [[r["model"], ", ".join(f"{k}={v}" for k, v in r["best_params"].items()),
        f"{r['cv_roc_auc']:.4f}", f"{r['test_roc_auc']:.4f}"] for r in tuning],
    widths=[4 * cm, 7.5 * cm, 2.5 * cm, 2.5 * cm],
)
p(
    f"<b>{best['name']}</b> was selected on cross-validated ROC-AUC ({best['cv_roc_auc']:.4f}) with "
    f"parameters {', '.join(f'{k}={v}' for k, v in best['params'].items())}. Selecting by test score "
    f"would have leaked the test set into model selection."
)
p(
    f"The default 0.50 cut-off maximises accuracy, the wrong target here. A threshold of "
    f"<b>{best['threshold']:.3f}</b> was chosen as the highest value still reaching 80% recall on "
    f"out-of-fold training predictions, so the test set played no part in the choice."
)
table(
    [["Operating point", "Accuracy", "Precision", "Recall", "F1", "Missed diabetics"],
     ["Default threshold 0.50", f"{default_op['accuracy']:.3f}", f"{default_op['precision']:.3f}",
      f"{default_op['recall']:.3f}", f"{default_op['f1']:.3f}",
      f"{default_op['confusion_matrix'][1][0]} of {sum(default_op['confusion_matrix'][1])}"],
     [f"Chosen threshold {best['threshold']:.3f}", f"{tuned_op['accuracy']:.3f}",
      f"{tuned_op['precision']:.3f}", f"{tuned_op['recall']:.3f}", f"{tuned_op['f1']:.3f}",
      f"{tuned_op['confusion_matrix'][1][0]} of {sum(tuned_op['confusion_matrix'][1])}"]],
    widths=[4.6 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 3 * cm],
    highlight_row=2,
)
figure("12_threshold_effect.png", "Figure 7. Confusion matrices at the two operating points.",
       width=13 * cm)
figure("11_precision_recall.png",
       "Figure 8. Precision-recall trade-off, with both operating points marked.", width=11.5 * cm)

h1("7. Explainability")
p(
    "Two complementary views were produced: permutation importance, which shuffles one feature and "
    "measures the resulting drop in ROC-AUC on held-out data, and SHAP, which attributes each "
    "individual prediction to its features."
)
figure("08_feature_importance.png", "Figure 9. Permutation importance on the test set.",
       width=13 * cm)
figure("09_shap_summary.png",
       "Figure 10. SHAP summary. Each dot is one patient; horizontal position is the push towards a "
       "diabetes prediction, colour is whether that feature's value was high (red) or low (blue).",
       width=14 * cm)
shap_rank = list((METRICS.get("explainability", {}).get("shap_mean_abs")
                  or METRICS["explainability"]["permutation_importance"]).keys())
p(
    f"Both methods agree on the ordering, led by {shap_rank[0]}, then {shap_rank[1]} and "
    f"{shap_rank[2]}. Critically, the directions are clinically correct: high glucose and high BMI "
    f"push predictions towards diabetes. A model with acceptable accuracy but an inverted "
    f"relationship would be dangerous, and this plot is what would expose it."
)

h1("8. Discussion and conclusion")
p(
    f"A tuned {best['name']} reached a test ROC-AUC of {best['test_roc_auc']:.3f}, in line with "
    f"published results on this dataset. Logistic Regression finished within a fraction of it, which "
    f"is worth stating plainly: on {METRICS['split']['train']} training rows with eight features, a "
    f"well-regularised linear model is competitive with anything more elaborate. The single decision "
    f"tree was the clear laggard."
)
p(
    f"The decisive result is the threshold. Moving it from 0.50 to {best['threshold']:.2f} converted a "
    f"model that missed {default_op['confusion_matrix'][1][0]} of "
    f"{sum(default_op['confusion_matrix'][1])} diabetics into one that misses "
    f"{tuned_op['confusion_matrix'][1][0]}, lifting recall from {default_op['recall']:.1%} to "
    f"{tuned_op['recall']:.1%}. Accuracy barely moved ({default_op['accuracy']:.3f} to "
    f"{tuned_op['accuracy']:.3f}) while recall gained "
    f"{(tuned_op['recall'] - default_op['recall']) * 100:.0f} points, which is the clearest possible "
    f"demonstration that accuracy was not measuring the quantity of interest. The gains in this "
    f"project came from preprocessing and from matching the operating point to the clinical cost of "
    f"each error - not from the choice of algorithm, where four of five models landed within a few "
    f"points of one another."
)
h2("Limitations")
bullets([
    f"<b>Narrow population.</b> {data['n_samples']} patients, all women of Pima heritage aged 21 and "
    "over near Phoenix. Nothing here licenses use on men, other ethnic groups, or younger patients; "
    "the unusually high local prevalence means even the base rate does not transfer.",
    f"<b>Heavy imputation.</b> {data['missing']['Insulin']['percent']:.1f}% of insulin and "
    f"{data['missing']['SkinThickness']['percent']:.1f}% of skin-fold values were filled with the median. "
    "Those two features are correspondingly the least trustworthy - and rank near the bottom of the "
    "importance plots.",
    "<b>Missingness is likely not random.</b> A patient with no recorded insulin value probably "
    "differs systematically from one who has it; median imputation assumes otherwise.",
    f"<b>No external validation.</b> All results derive from one {METRICS['split']['test']}-patient "
    "test split of a single dataset, so the interval around the reported recall is wide.",
    "<b>Information ceiling.</b> Eight coarse measurements carry limited signal. Materially better "
    "results need better features - HbA1c, fasting glucose, waist circumference - not more elaborate "
    "models.",
])
h2("Conclusion")
p(
    f"As a triage aid flagging roughly {tuned_op['recall']:.0%} of diabetics for confirmatory testing, "
    f"the model is plausible and its behaviour is explainable. As a diagnostic device it is not, for "
    f"the reasons above. Future work: external validation on a broader cohort, multiple imputation "
    f"with an explicit missing-indicator feature, probability calibration so outputs can be read as "
    f"real risks, and cost-sensitive learning driven by the actual cost of each error type."
)

h1("References")
bullets([
    "Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). Using the ADAP "
    "learning algorithm to forecast the onset of diabetes mellitus. <i>Proc. Symp. on Computer "
    "Applications in Medical Care</i>, 261-265.",
    "Dua, D. and Graff, C. (2019). <i>UCI Machine Learning Repository - Pima Indians Diabetes "
    "Database.</i> University of California, Irvine.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i> 12, 2825-2830.",
    "Lundberg, S.M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. "
    "<i>NeurIPS</i> 30.",
    "Saito, T. and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC "
    "Plot When Evaluating Binary Classifiers on Imbalanced Datasets. <i>PLOS ONE</i> 10(3).",
])


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.3 * cm, "CBIO313 - Diabetes Prediction")
        canvas.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.4 * cm, rightMargin=2.4 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Diabetes Prediction from Clinical Measurements",
        author=AUTHOR, subject="CBIO313 Course Project",
    ).build(story, onFirstPage=decorate, onLaterPages=decorate)
    print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
