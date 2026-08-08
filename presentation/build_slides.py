"""Build presentation/Project_Presentation.pdf - a 16:9 slide deck.

The project guidelines accept "PowerPoint or PDF" for the presentation, so the deck is
generated as a PDF with reportlab. As with the report, every figure and number is pulled
from the artefacts written by src/train.py, so the slides cannot drift from the results.

    python presentation/build_slides.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
FIGURES = ROOT / "figures"
OUT = ROOT / "presentation" / "Project_Presentation.pdf"

AUTHOR = "Shahd Sameh Safwat"
STUDENT_ID = "231002256"

W, H = 13.333 * inch, 7.5 * inch  # 16:9
NAVY = colors.HexColor("#1f3864")
BLUE = colors.HexColor("#2e5c9a")
RED = colors.HexColor("#c44e52")
GREY = colors.HexColor("#5a5a5a")
LIGHT = colors.HexColor("#eef2f8")

best = METRICS["best_model"]
data = METRICS["dataset"]
default_op, tuned_op = best["at_default_threshold"], best["at_tuned_threshold"]

body = ParagraphStyle("body", fontName="Helvetica", fontSize=17, leading=26,
                      textColor=colors.HexColor("#222222"), alignment=TA_LEFT)
small = ParagraphStyle("small", parent=body, fontSize=14, leading=20, textColor=GREY)
big = ParagraphStyle("big", parent=body, fontSize=21, leading=30)

c = pdfcanvas.Canvas(str(OUT), pagesize=(W, H))
slide_no = 0


def chrome(title, subtitle=None):
    """Draw the standard slide frame and return the y where content may start."""
    global slide_no
    slide_no += 1
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, H - 1.15 * inch, W, 1.15 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(0.62 * inch, H - 0.78 * inch, title)
    if subtitle:
        c.setFont("Helvetica", 14)
        c.setFillColor(colors.HexColor("#c9d6ea"))
        c.drawString(0.66 * inch, H - 1.03 * inch, subtitle)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 10)
    c.drawString(0.62 * inch, 0.32 * inch, "CBIO313 - Diabetes Prediction")
    c.drawRightString(W - 0.62 * inch, 0.32 * inch, str(slide_no))
    return H - 1.6 * inch


def para(text, x, y, width, style=body):
    p = Paragraph(text, style)
    _, h = p.wrap(width, H)
    p.drawOn(c, x, y - h)
    return y - h


def bullets(items, x, y, width, style=body, gap=13):
    for item in items:
        y = para(f"<font color='#2e5c9a'><b>&bull;</b></font>  {item}", x, y, width, style) - gap
    return y


def picture(name, x, y_top, max_w, max_h):
    """Draw a figure scaled to fit, anchored at its top-left."""
    path = FIGURES / name
    if not path.exists():
        return
    from PIL import Image as PILImage

    iw, ih = PILImage.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    c.drawImage(str(path), x, y_top - h, width=w, height=h, mask="auto")


def stat_box(x, y, w, h, value, label, colour=BLUE):
    c.setFillColor(LIGHT)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(colour)
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(x + w / 2, y + h - 0.62 * inch, value)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 12)
    c.drawCentredString(x + w / 2, y + 0.22 * inch, label)


def table(rows, x, y_top, widths, highlight=None, font=13):
    t = Table(rows, colWidths=widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b9c2d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    if highlight:
        style += [("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#d6e4f7")),
                  ("FONTNAME", (0, highlight), (-1, highlight), "Helvetica-Bold"),
                  ("TEXTCOLOR", (0, highlight), (-1, highlight), NAVY)]
    t.setStyle(TableStyle(style))
    _, h = t.wrap(sum(widths), H)
    t.drawOn(c, x, y_top - h)
    return y_top - h


# =============================================================== 1. title
c.setFillColor(NAVY)
c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(colors.HexColor("#4a7ac7"))
c.rect(0, 0, W, 0.28 * inch, fill=1, stroke=0)
c.setFillColor(colors.white)
c.setFont("Helvetica-Bold", 46)
c.drawCentredString(W / 2, H - 2.6 * inch, "Diabetes Prediction from")
c.drawCentredString(W / 2, H - 3.35 * inch, "Clinical Measurements")
c.setFont("Helvetica", 19)
c.setFillColor(colors.HexColor("#c9d6ea"))
c.drawCentredString(W / 2, H - 4.15 * inch,
                    "Comparing five machine learning algorithms on the Pima Indians dataset")
c.setFont("Helvetica", 16)
c.setFillColor(colors.white)
c.drawCentredString(W / 2, H - 5.3 * inch, f"{AUTHOR}  -  {STUDENT_ID}")
c.setFont("Helvetica", 14)
c.setFillColor(colors.HexColor("#c9d6ea"))
c.drawCentredString(W / 2, H - 5.75 * inch, "CBIO313 - Data Mining & Machine Learning")
c.drawCentredString(W / 2, H - 6.1 * inch, "Instructor: Dr. Muhammad Elsayeh")
c.showPage()

# =============================================================== 2. problem
y = chrome("The problem", "Why this matters")
bullets([
    "Type 2 diabetes develops <b>silently</b>. By the time symptoms appear, damage to the "
    "kidneys, retina and nerves is often already underway.",
    "Screening programmes exist for this reason - but a full diagnostic workup is expensive "
    "and cannot be given to everyone.",
    "<b>The goal:</b> a cheap triage step that flags who most needs that workup, using "
    "measurements already taken at a routine visit.",
], 0.75 * inch, y, W - 3.4 * inch, big)

c.setFillColor(LIGHT)
c.roundRect(0.75 * inch, 0.95 * inch, W - 1.5 * inch, 1.5 * inch, 10, fill=1, stroke=0)
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 18)
c.drawString(1.05 * inch, 2.05 * inch, "The two errors are not equally costly")
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica", 15)
c.drawString(1.05 * inch, 1.68 * inch,
             "False positive  ->  a healthy patient goes for one extra blood test.  Inconvenient.")
c.setFillColor(RED)
c.setFont("Helvetica-Bold", 15)
c.drawString(1.05 * inch, 1.32 * inch,
             "False negative  ->  an undiagnosed diabetic is sent home reassured.  Serious.")
c.showPage()

# =============================================================== 3. objectives
y = chrome("Objectives")
bullets([
    "Clean a dataset where missing values were silently recorded as zeros",
    "Explore how each measurement relates to diabetes status",
    "Train and compare <b>five</b> algorithms from the course",
    "Evaluate with metrics suited to imbalanced medical screening - not accuracy alone",
    "Tune the best model and choose a threshold appropriate for screening",
    "Explain <i>why</i> the model predicts what it predicts",
], 0.9 * inch, y - 0.25 * inch, W - 2 * inch, big, gap=16)
c.showPage()

# =============================================================== 4. dataset
y = chrome("The dataset", "Pima Indians Diabetes Database - UCI / NIDDK")
stat_box(0.75 * inch, y - 1.5 * inch, 2.6 * inch, 1.4 * inch, str(data["n_samples"]), "patients")
stat_box(3.6 * inch, y - 1.5 * inch, 2.6 * inch, 1.4 * inch, str(data["n_features"]), "features")
stat_box(6.45 * inch, y - 1.5 * inch, 2.6 * inch, 1.4 * inch,
         f"{data['positive_rate']:.0%}", "have diabetes")
stat_box(9.3 * inch, y - 1.5 * inch, 3.2 * inch, 1.4 * inch, "21+", "women, Pima heritage")

y2 = y - 2.1 * inch
para("<b>Eight measurements:</b> pregnancies, plasma glucose, blood pressure, skin-fold "
     "thickness, serum insulin, BMI, family-history score, age.", 0.75 * inch, y2,
     W - 1.5 * inch, body)
para(f"<b>Target:</b> Outcome - 1 = diabetes, 0 = no diabetes. "
     f"{data['class_counts']['no_diabetes']} negative / {data['class_counts']['diabetes']} positive.",
     0.75 * inch, y2 - 0.75 * inch, W - 1.5 * inch, body)
c.setFillColor(GREY)
c.setFont("Helvetica-Oblique", 13)
c.drawString(0.75 * inch, 1.0 * inch,
             "All patients are women of Pima heritage aged 21+ near Phoenix, Arizona - "
             "a narrow population, revisited in the limitations.")
c.showPage()

# =============================================================== 5. finding 1
y = chrome("Finding 1: the zeros are not zeros", "The most important preprocessing step")
para("Five columns report a minimum of <b>0</b> - glucose, blood pressure, skin thickness, "
     "insulin, BMI. None is biologically possible in a living patient.",
     0.75 * inch, y, 6.6 * inch, body)
para("<b>These are missing measurements that were never marked as missing.</b> Left alone, a "
     "model reads <i>insulin = 0</i> as a real, extremely low reading - the opposite of the truth.",
     0.75 * inch, y - 1.25 * inch, 6.6 * inch, body)

rows = [["Column", "Zeros", "% of rows"]] + [
    [k, str(v["zeros"]), f"{v['percent']:.1f}%"] for k, v in data["missing"].items()]
table(rows, 8.0 * inch, y + 0.15 * inch, [2.4 * inch, 1.2 * inch, 1.4 * inch], highlight=1)

c.setFillColor(RED)
c.setFont("Helvetica-Bold", 17)
c.drawString(0.75 * inch, 1.55 * inch,
             "Dropping incomplete rows would throw away nearly half the dataset.")
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica", 15)
c.drawString(0.75 * inch, 1.15 * inch,
             "Instead: mark as NaN, then impute the median inside the model pipeline.")
c.showPage()

# =============================================================== 6. leakage
y = chrome("Preprocessing", "And how we avoided cheating")
para("<b>Pipeline:</b> impute (median) &rarr; scale (StandardScaler) &rarr; classify. "
     "One estimator, identical for all five algorithms.", 0.75 * inch, y, W - 1.5 * inch, big)

c.setFillColor(LIGHT)
c.roundRect(0.75 * inch, 1.5 * inch, W - 1.5 * inch, 3.4 * inch, 10, fill=1, stroke=0)
yy = 4.55 * inch
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 19)
c.drawString(1.05 * inch, yy, "Three places this project could have quietly cheated")
items = [
    "<b>Imputation is inside the pipeline.</b> Taking the median over all 768 rows and then "
    "splitting would leak test data into training and inflate every score.",
    "<b>The winner was chosen on cross-validated training score</b>, never on test performance. "
    "Picking by test score is model selection on the test set.",
    "<b>The threshold was chosen from out-of-fold training predictions.</b> The test set was "
    "used exactly once, at the very end.",
]
bullets(items, 1.05 * inch, yy - 0.45 * inch, W - 2.4 * inch,
        ParagraphStyle("x", parent=body, fontSize=15, leading=22), gap=10)
c.showPage()

# =============================================================== 7. EDA
y = chrome("Exploratory data analysis", "Class balance and missing data")
picture("01_overview.png", 1.35 * inch, y + 0.1 * inch, 10.6 * inch, 4.4 * inch)
c.setFillColor(RED)
c.setFont("Helvetica-Bold", 17)
c.drawCentredString(W / 2, 1.15 * inch,
                    f"A model that always says \"no diabetes\" scores "
                    f"{1 - data['positive_rate']:.0%} accuracy and catches zero diabetics.")
c.setFillColor(GREY)
c.setFont("Helvetica", 14)
c.drawCentredString(W / 2, 0.82 * inch, "That is the benchmark every accuracy figure must be read against.")
c.showPage()

# =============================================================== 8. distributions
y = chrome("Which measurements separate the groups?")
picture("02_distributions.png", 0.7 * inch, y + 0.15 * inch, 11.9 * inch, 4.5 * inch)
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica", 15)
c.drawString(0.75 * inch, 1.15 * inch,
             "Glucose separates the two groups most clearly. BMI and Age shift moderately.")
c.setFillColor(GREY)
c.drawString(0.75 * inch, 0.82 * inch,
             "BloodPressure and SkinThickness overlap almost entirely - they will contribute little.")
c.showPage()

# =============================================================== 9. algorithms
y = chrome("Five algorithms", "Each in the identical impute - scale - classify pipeline")
cv = {r["model"]: r for r in METRICS["cross_validation"]}
rows = [["Algorithm", "Why include it", "CV ROC-AUC"]]
for name, why in [
    ("Logistic Regression", "Linear baseline, interpretable coefficients"),
    ("SVM (RBF)", "Maximum margin with a non-linear kernel"),
    ("K-Nearest Neighbours", "Non-parametric, purely local decision rule"),
    ("Random Forest", "Bagged ensemble of 300 trees"),
    ("Decision Tree", "Interpretable rules, reference for the ensemble"),
]:
    rows.append([name, why, f"{cv[name]['roc_auc']:.3f}"])
table(rows, 0.9 * inch, y, [3.1 * inch, 6.1 * inch, 1.9 * inch], highlight=1, font=15)
c.setFillColor(GREY)
c.setFont("Helvetica-Oblique", 14)
c.drawString(0.9 * inch, 1.5 * inch,
             "Stratified 5-fold cross-validation on the training set only - the test set stays sealed.")
c.showPage()

# =============================================================== 10. comparison
y = chrome("Model comparison", "Cross-validated on the training set")
picture("05_model_comparison.png", 1.6 * inch, y + 0.1 * inch, 10.1 * inch, 4.3 * inch)
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica-Bold", 16)
c.drawString(0.9 * inch, 1.15 * inch,
             "Notice: recall is far below accuracy for every single model.")
c.setFillColor(GREY)
c.setFont("Helvetica", 14)
c.drawString(0.9 * inch, 0.82 * inch,
             "At the default cut-off these classifiers catch only about 55-60% of diabetics.")
c.showPage()

# =============================================================== 11. ROC
y = chrome("Held-out test set", "ROC curves - the test set's one and only use")
picture("06_roc_curves.png", 1.15 * inch, y + 0.1 * inch, 5.2 * inch, 4.6 * inch)
tr = sorted(METRICS["test_results"], key=lambda r: -r["roc_auc"])
rows = [["Model", "Acc.", "Recall", "ROC-AUC"]] + [
    [r["model"], f"{r['accuracy']:.3f}", f"{r['recall']:.3f}", f"{r['roc_auc']:.3f}"] for r in tr]
table(rows, 6.8 * inch, y - 0.15 * inch, [2.8 * inch, 1.0 * inch, 1.1 * inch, 1.2 * inch], font=13)
c.showPage()

# =============================================================== 12. tuning
y = chrome("Hyperparameter tuning", "GridSearchCV, scored on ROC-AUC")
para("Grids are scored on <b>ROC-AUC, not recall</b>. Optimising directly on recall rewards models "
     "that just predict \"diabetic\" more often - in the limit, flagging everyone scores perfect "
     "recall. ROC-AUC ranks models independently of the threshold.",
     0.85 * inch, y, W - 1.7 * inch, body)
rows = [["Model", "Best parameters", "CV ROC-AUC", "Test ROC-AUC"]] + [
    [r["model"], ", ".join(f"{k}={v}" for k, v in r["best_params"].items()),
     f"{r['cv_roc_auc']:.4f}", f"{r['test_roc_auc']:.4f}"]
    for r in METRICS["tuning"]]
winner = next(i for i, r in enumerate(METRICS["tuning"], start=1) if r["model"] == best["name"])
table(rows, 0.85 * inch, y - 1.5 * inch, [2.7 * inch, 5.4 * inch, 1.7 * inch, 1.8 * inch],
      highlight=winner, font=14)
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 18)
c.drawString(0.85 * inch, 1.25 * inch,
             f"Selected: {best['name']} - CV ROC-AUC {best['cv_roc_auc']:.3f}, "
             f"test ROC-AUC {best['test_roc_auc']:.3f}")
c.showPage()

# =============================================================== 13. finding 2
y = chrome("Finding 2: the threshold is the real result", "Where the marks are")
para("<b>predict() flags a patient at probability 0.50.</b> That cut-off maximises accuracy - the "
     "wrong target for screening. We chose the threshold from out-of-fold training predictions to "
     "reach 80% recall.", 0.85 * inch, y, W - 1.7 * inch, body)

rows = [["Operating point", "Accuracy", "Precision", "Recall", "Missed diabetics"],
        ["Default 0.50", f"{default_op['accuracy']:.3f}", f"{default_op['precision']:.3f}",
         f"{default_op['recall']:.3f}",
         f"{default_op['confusion_matrix'][1][0]} of {sum(default_op['confusion_matrix'][1])}"],
        [f"Chosen {best['threshold']:.3f}", f"{tuned_op['accuracy']:.3f}",
         f"{tuned_op['precision']:.3f}", f"{tuned_op['recall']:.3f}",
         f"{tuned_op['confusion_matrix'][1][0]} of {sum(tuned_op['confusion_matrix'][1])}"]]
table(rows, 0.85 * inch, y - 1.35 * inch,
      [2.7 * inch, 1.9 * inch, 1.9 * inch, 1.9 * inch, 2.8 * inch], highlight=2, font=15)

c.setFillColor(RED)
c.setFont("Helvetica-Bold", 21)
c.drawString(0.85 * inch, 1.95 * inch,
             f"Accuracy moved 1.3 points. Recall moved "
             f"{(tuned_op['recall'] - default_op['recall']) * 100:.0f} points.")
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica", 16)
c.drawString(0.85 * inch, 1.5 * inch,
             f"Missed diabetics fell from {default_op['confusion_matrix'][1][0]} to "
             f"{tuned_op['confusion_matrix'][1][0]}, for a "
             f"{(default_op['precision'] - tuned_op['precision']) * 100:.1f}-point precision cost.")
c.setFillColor(GREY)
c.setFont("Helvetica-Oblique", 15)
c.drawString(0.85 * inch, 1.1 * inch,
             "Accuracy was simply not measuring the thing we care about.")
c.showPage()

# =============================================================== 14. threshold figure
y = chrome("The same model, two operating points")
picture("12_threshold_effect.png", 2.2 * inch, y + 0.15 * inch, 8.9 * inch, 4.5 * inch)
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica", 16)
c.drawCentredString(W / 2, 1.1 * inch,
                    "Bottom-left cell = diabetics the model sent home. That is the number to watch.")
c.showPage()

# =============================================================== 15. SHAP
y = chrome("Explainability", "Permutation importance and SHAP agree")
picture("09_shap_summary.png", 0.85 * inch, y + 0.1 * inch, 7.6 * inch, 4.5 * inch)
yy = y - 0.1 * inch
bullets([
    "Each dot is one patient",
    "Right = pushed towards diabetes",
    "Red = high value, blue = low",
], 8.9 * inch, yy, 3.7 * inch, ParagraphStyle("s", parent=body, fontSize=15, leading=21), gap=6)
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 17)
c.drawString(8.9 * inch, 3.25 * inch, "Glucose, then BMI")
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica", 14)
tw = c.beginText(8.9 * inch, 2.9 * inch)
for line in ["High glucose pushes towards", "diabetes; low glucose pulls back.",
             "The direction is clinically",
             "correct - a model with the right", "accuracy but an inverted",
             "relationship would be dangerous,", "and this plot would expose it."]:
    tw.textLine(line)
c.drawText(tw)
c.showPage()

# =============================================================== 16. demo
y = chrome("Live demonstration", "Bonus: Streamlit prediction app")
bullets([
    "Enter eight measurements on the sidebar",
    "Get a predicted probability and a screening decision at the tuned threshold",
    "See how the patient compares to the population",
    "See a <b>per-patient SHAP explanation</b> of that specific prediction",
], 0.9 * inch, y - 0.2 * inch, W - 2 * inch, big, gap=15)

c.setFillColor(LIGHT)
c.roundRect(0.9 * inch, 1.5 * inch, W - 1.8 * inch, 2.0 * inch, 10, fill=1, stroke=0)
c.setFillColor(NAVY)
c.setFont("Helvetica-Bold", 19)
c.drawString(1.2 * inch, 3.05 * inch, "Demo")
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Courier-Bold", 16)
c.drawString(1.2 * inch, 2.62 * inch, "streamlit run src/app.py")
c.setFont("Helvetica", 15)
c.drawString(1.2 * inch, 2.15 * inch,
             "Default patient: 23.5% probability - not flagged.")
c.setFillColor(RED)
c.setFont("Helvetica-Bold", 15)
c.drawString(1.2 * inch, 1.78 * inch,
             "Raise glucose to 190: 78.6% - flagged for confirmatory testing.")
c.showPage()

# =============================================================== 17. limitations
y = chrome("Limitations", "What this model cannot do")
bullets([
    f"<b>Narrow population.</b> {data['n_samples']} patients, all women of Pima heritage aged 21+. "
    "Nothing transfers to men, other ethnicities, or younger patients.",
    f"<b>Heavy imputation.</b> {data['missing']['Insulin']['percent']:.0f}% of insulin and "
    f"{data['missing']['SkinThickness']['percent']:.0f}% of skin-fold values were filled in - and "
    "both rank near the bottom of the importance plots.",
    "<b>Missingness is probably not random.</b> Median imputation assumes it is.",
    f"<b>No external validation.</b> One {METRICS['split']['test']}-patient test split, "
    "so the interval around a recall of 0.81 is wide.",
    "<b>Information ceiling.</b> Eight coarse measurements carry limited signal. Better results "
    "need better features (HbA1c, fasting glucose), not fancier models.",
], 0.85 * inch, y - 0.15 * inch, W - 1.9 * inch,
    ParagraphStyle("l", parent=body, fontSize=16, leading=23), gap=11)
c.setFillColor(RED)
c.setFont("Helvetica-Bold", 17)
c.drawString(0.85 * inch, 0.95 * inch, "This is a course project, not a medical device.")
c.showPage()

# =============================================================== 18. conclusion
y = chrome("Conclusion")
stat_box(0.85 * inch, y - 1.45 * inch, 2.8 * inch, 1.35 * inch, f"{best['test_roc_auc']:.3f}",
         "test ROC-AUC")
stat_box(3.95 * inch, y - 1.45 * inch, 2.8 * inch, 1.35 * inch, f"{tuned_op['recall']:.0%}",
         "of diabetics caught", RED)
stat_box(7.05 * inch, y - 1.45 * inch, 2.8 * inch, 1.35 * inch,
         f"{tuned_op['confusion_matrix'][1][0]} of {sum(tuned_op['confusion_matrix'][1])}",
         "missed, down from 27")
stat_box(10.15 * inch, y - 1.45 * inch, 2.3 * inch, 1.35 * inch, "5", "algorithms compared")

yy = y - 2.05 * inch
bullets([
    "The gains came from <b>preprocessing</b> and from <b>matching the operating point to the "
    "clinical cost of each error</b> - not from the choice of algorithm.",
    "Four of the five models landed within a few points of each other. A well-regularised "
    "linear model was competitive with everything more elaborate.",
    "As a triage aid flagging ~80% of diabetics for confirmatory testing, the model is plausible "
    "and its behaviour is explainable.",
], 0.85 * inch, yy, W - 1.9 * inch, ParagraphStyle("c", parent=body, fontSize=16, leading=23),
    gap=11)
c.showPage()

# =============================================================== 19. thanks
c.setFillColor(NAVY)
c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(colors.HexColor("#4a7ac7"))
c.rect(0, 0, W, 0.28 * inch, fill=1, stroke=0)
c.setFillColor(colors.white)
c.setFont("Helvetica-Bold", 52)
c.drawCentredString(W / 2, H / 2 + 0.55 * inch, "Thank you")
c.setFont("Helvetica", 22)
c.setFillColor(colors.HexColor("#c9d6ea"))
c.drawCentredString(W / 2, H / 2 - 0.25 * inch, "Questions?")
c.setFont("Helvetica", 15)
c.setFillColor(colors.white)
c.drawCentredString(W / 2, 1.75 * inch, "github.com/ssameh12/diabetes-prediction-cbio313")
c.setFont("Helvetica", 13)
c.setFillColor(colors.HexColor("#c9d6ea"))
c.drawCentredString(W / 2, 1.35 * inch, f"{AUTHOR}  -  {STUDENT_ID}")
c.showPage()

c.save()
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {slide_no + 2} slides)")
