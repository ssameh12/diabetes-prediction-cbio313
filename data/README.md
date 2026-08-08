# Data

## `diabetes.csv`

The **Pima Indians Diabetes Database**, 768 rows × 9 columns. A header row was added to the
distributed file; the values are otherwise untouched.

| Property | Value |
|---|---|
| Rows | 768 |
| Features | 8 numeric |
| Target | `Outcome` — 1 = diabetes, 0 = no diabetes |
| Class split | 500 negative / 268 positive (34.9% positive) |
| Population | Women of Pima Indian heritage, aged ≥ 21, near Phoenix, Arizona |

### Original source

National Institute of Diabetes and Digestive and Kidney Diseases, published through the
UCI Machine Learning Repository.

> Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988).
> *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
> Proceedings of the Symposium on Computer Applications in Medical Care, 261–265.

### Columns

| Column | Meaning | Unit |
|---|---|---|
| `Pregnancies` | Number of times pregnant | count |
| `Glucose` | Plasma glucose, 2 h into an oral glucose tolerance test | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skin-fold thickness | mm |
| `Insulin` | 2-hour serum insulin | mu U/mL |
| `BMI` | Body mass index | kg/m² |
| `DiabetesPedigreeFunction` | Family-history-weighted diabetes likelihood score | — |
| `Age` | Age | years |
| `Outcome` | Target | 0 / 1 |

### ⚠️ Known data quality issue

**Missing values in this dataset are encoded as `0`, not as blanks.** Five columns are affected:

| Column | Zeros | % of rows |
|---|---|---|
| `Insulin` | 374 | 48.7% |
| `SkinThickness` | 227 | 29.6% |
| `BloodPressure` | 35 | 4.6% |
| `BMI` | 11 | 1.4% |
| `Glucose` | 5 | 0.7% |

None of these can biologically be zero in a living patient. `src/data_prep.py` converts them to
`NaN`; imputation then happens **inside the model pipeline** so that fill values are learned from
training folds only.

`Pregnancies` is excluded from this treatment — zero pregnancies is a real value.
