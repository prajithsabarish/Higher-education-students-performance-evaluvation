# Higher Education Students' Performance Evaluation: An Exploratory Data Analysis and Visualization Case Study with Deep Learning

---

## TITLE PAGE

---

**Title:** Higher Education Students' Performance Evaluation: An Exploratory Data Analysis and Visualization Case Study with Deep Learning

**Student Name:** Prajith Sabarish D

**Register Number:** 24AD079

**Department:** Artificial Intelligence and Data Science

**Year / Semester:** III Year B.Tech (V Semester)

**College:** [COLLEGE NAME]

**Course Code and Name:** [COURSE CODE AND NAME]

**Faculty Advisor:** [FACULTY NAME]

**Date of Submission:** [DATE]

**GitHub Repository:** [PUBLIC REPO LINK]

---

*Submitted in partial fulfilment of the requirements for the graded case study (40 Marks): "Exploratory Data Analysis and Visualization of a Real-World Dataset."*

---

\newpage

## TABLE OF CONTENTS

*(Auto-generate in Word using Heading styles)*

1. Abstract
2. Introduction and Objective
3. Dataset Description
4. EDA and Preprocessing
5. Data Visualization
6. Deep Learning Model
7. Result Visualization and Interpretation
8. Conclusion and Future Scope
9. References
10. Appendix

---

## LIST OF FIGURES

| No. | Title | Page |
|:----|:------|:-----|
| Fig. 1 | End-of-term grade distribution (counts and percentages) and Grade_Group target tier donut chart | |
| Fig. 2 | Univariate demographics: personal characteristics (Q1-Q10) small-multiple bar charts | |
| Fig. 3 | Family socioeconomic indicators (Q11-Q16) small-multiple bar charts | |
| Fig. 4 | Academic habits and cognitive engagement (Q17-Q30) small-multiple bar charts | |
| Fig. 5 | Box, violin, and strip plots of engineered composite scores | |
| Fig. 6 | Density histograms with KDE and Gaussian parametric curves for engineered scores | |
| Fig. 7 | Discrete distribution goodness-of-fit: observed Grade PMF vs Binomial and Poisson models with ECDF | |
| Fig. 8 | Binomial cohort model Bin(10, 0.324) for high performers with P(X >= 3) shaded | |
| Fig. 9 | Continuous normality assessment: Shapiro-Wilk histograms and Q-Q diagnostic plots | |
| Fig. 10 | Central Limit Theorem validation: bootstrap sampling distributions for n = 10, 30, 145 | |
| Fig. 11 | Conditional probability heatmaps P(Grade_Group given Feature) with chi-square tests | |
| Fig. 12 | Masked lower-triangular Spearman rank correlation matrix | |
| Fig. 13 | Cramer's V association matrix for nominal categorical variables | |
| Fig. 14 | Top 10 predictors by signed Spearman correlation with Grade and significance stars | |
| Fig. 15 | Course ID vs Grade cross-tabulation heatmap with enrollment annotations | |
| Fig. 16 | 100% stacked proportion bars of Grade_Group by Attendance, Study Hours, Scholarship, and Sex | |
| Fig. 17 | Grade distribution across Prior GPA tiers and Weekly Study Hours (box + strip) | |
| Fig. 18 | Academic aspiration bubble chart: Prior GPA vs Expected GPA (size = count, colour = mean grade) | |
| Fig. 19 | Jittered regressions of Study Engagement and Class Behaviour vs Grade | |
| Fig. 20 | Multivariate pairplot of top features stratified by Grade_Group | |
| Fig. 21 | PCA eigenvalue scree spectrum and 2-D biplot with feature loadings | |
| Fig. 22 | Radar chart of normalised behavioural profiles for Low, Medium, and High attainment groups | |
| Fig. R | Stage 6 refinement demonstration: unformatted raw chart vs publication-standard design | |
| Fig. 23 | Keras MLP training and validation loss curves | |
| Fig. 24 | Training and validation accuracy curves across epochs | |
| Fig. 25 | 3-class risk-tier confusion matrix (raw counts and row-normalised recall) | |
| Fig. 26 | 8-class fine-grained letter grade confusion matrix | |
| Fig. 27 | Multiclass one-vs-rest ROC curves (holdout test set and 5-fold out-of-fold) | |
| Fig. 28 | Multi-tier precision-recall curves | |
| Fig. 29 | Cross-validated model benchmark comparison: MLP vs Logistic Regression vs Random Forest | |
| Fig. 30 | Hyperparameter grid-search validation Macro-F1 heatmap | |
| Fig. 31 | Dual-perspective feature importance: permutation importance vs Random Forest Gini impurity | |
| Fig. 32 | Ordinal error distribution: absolute distance in letter grade steps for 8-class model | |
| Fig. 33 | Feature ablation study: impact of removing Prior GPA and Course ID confounders | |

---

## LIST OF TABLES

| No. | Title | Page |
|:----|:------|:-----|
| Table 1 | Complete data dictionary: features, types, and categorical levels | |
| Table 2 | Target variable (GRADE) distribution: counts and percentages | |
| Table 3 | Grade_Group target tier definition and class frequencies | |
| Table 4 | Course ID enrolment distribution | |
| Table 5 | Data quality audit summary | |
| Table 6 | EDA and preprocessing pipeline: step, technique, and result | |
| Table 7 | Engineered composite feature formulas | |
| Table 8 | Stratified data split: sizes and class distributions | |
| Table 9 | Ben Fry's seven stages of visualisation: mapping to notebook sections and figures | |
| Table 10 | Summary of statistical hypothesis tests | |
| Table 11 | Summary of key EDA findings | |
| Table 12 | Deep learning MLP architecture: layers and parameter counts | |
| Table 13 | Training configuration and hyperparameters | |
| Table 14 | Hyperparameter grid search results | |
| Table 15 | 5-fold stratified cross-validation performance benchmarks | |
| Table 16 | Feature ablation study results | |
| Table 17 | Comprehensive evaluation metrics for the 3-class and 8-class tasks | |
| Table 18 | Rubric compliance matrix | |

---

\newpage

## 1. ABSTRACT

University dropout and academic underperformance impose severe institutional and societal costs, yet traditional academic advising remains largely reactive. This case study presents a comprehensive exploratory data analysis (EDA), statistical visualisation, and deep learning investigation of the UCI "Higher Education Students Performance Evaluation" dataset, comprising 145 student records across 33 attributes collected from the Faculty of Engineering and Faculty of Educational Sciences in Cyprus (Yilmaz and Sekeroglu, 2019) [1]. Following Ben Fry's seven stages of data visualisation [2], the analysis employed 34 static publication-grade figures, 6 interactive Plotly dashboards, non-parametric hypothesis testing with Benjamini-Hochberg correction, discrete and continuous probability distribution modelling, and a deep multilayer perceptron (MLP) classifier built with TensorFlow/Keras [8]. Prior cumulative GPA emerged as the strongest individual predictor of end-of-term grade (Spearman rho = +0.351, p < 0.001), while institutional course assignment acted as a powerful confounder. The deep MLP classifier achieved 5-fold cross-validated accuracy of 0.683 +/- 0.052 on the 3-class risk-tier target, outperforming both logistic regression (0.648 +/- 0.061) and random forest (0.662 +/- 0.045). These findings demonstrate the feasibility of proactive early-warning systems for student retention, subject to ethical constraints on the use of sensitive demographic attributes.

**Word count:** 175

**Keywords:** exploratory data analysis; student performance prediction; data visualisation; deep learning; higher education analytics; early-warning systems

---

\newpage

## 2. INTRODUCTION AND OBJECTIVE

### 2.1 Background and Problem Statement

The global expansion of higher education has intensified institutional pressure to improve student retention, graduation rates, and learning outcomes. According to multiple educational data-mining studies, academic failure is rarely abrupt; rather, it develops progressively through a combination of insufficient prior preparation, disengaged study habits, and socioeconomic circumstances [1], [7]. Despite this knowledge, the majority of universities still rely on reactive academic-advising frameworks: formal academic probation is typically imposed only after end-of-semester examinations have been graded, by which point the opportunity for pedagogical intervention has passed.

The emergence of educational data mining and learning analytics offers a transformative alternative. By systematically analysing student demographic profiles, self-reported behavioural patterns, and historical academic trajectories, institutions can build proactive early-warning systems that identify at-risk students within the first weeks of a term. Such systems enable targeted interventions---peer tutoring, study-skills workshops, faculty office-hour outreach---before irrecoverable academic damage occurs.

This case study addresses the following central problem: *Given 30 self-reported survey responses, a course identifier, and prior academic performance data for 145 university students, can exploratory data analysis and deep learning classification reliably identify students at risk of low end-of-term grades, and what behavioural and demographic factors most strongly predict academic attainment?*

### 2.2 Real-World Application

The practical application envisaged in this study is an **Early-Warning and Academic Advising Decision Support System**. The system would operate as follows:

1. **Data Ingestion:** At the start of each academic term, incoming student survey responses and historical GPA records are collected through a standardised intake form.
2. **Risk Scoring:** A trained classification model assigns each student to one of three attainment tiers---Low, Medium, or High---based on the 31-feature input vector.
3. **Automated Alerting:** Students flagged as "Low" risk are automatically referred for advising appointments, supplemental instruction, and attendance monitoring.
4. **Continuous Monitoring:** Attendance and engagement metrics are tracked throughout the term, and the risk tier is updated dynamically.
5. **Institutional Reporting:** Aggregated course-level risk profiles inform departmental curriculum reviews and grading-standard audits.

### 2.3 Objectives

The specific objectives of this case study are:

1. To acquire, parse, clean, and validate the UCI Higher Education Students Performance Evaluation dataset, ensuring 100% data integrity and correct categorical decoding.
2. To conduct rigorous univariate, bivariate, and multivariate exploratory data analysis using non-parametric statistical tests (Spearman, Cramer's V, Mann-Whitney, Kruskal-Wallis) with false-discovery-rate correction.
3. To model the discrete target distribution using Binomial, Poisson, and discrete uniform probability models, and to assess continuous normality of engineered composite features.
4. To produce a minimum of 22 publication-grade static visualisations and 6 interactive dashboards, following Ben Fry's seven stages of data visualisation.
5. To design, train, and evaluate a deep multilayer perceptron classifier for 3-class risk-tier and 8-class letter-grade prediction, benchmarked against logistic regression and random forest baselines.
6. To derive actionable, data-driven recommendations for university academic-advising practice, while acknowledging ethical constraints on the use of sensitive attributes.

### 2.4 Scope and Contributions

This study is scoped to the single UCI dataset of 145 students from two faculties at a Cyprus university, collected in 2019. The analysis does not claim generalisability to other institutions or educational contexts. The primary contributions are:

- A complete, reproducible end-to-end data science pipeline from raw CSV to trained deep learning model, implemented in Python with full code availability on GitHub.
- A rigorous statistical analysis framework applying non-parametric tests appropriate for ordinal and nominal survey data, with explicit control for multiple comparisons.
- A systematic visualisation architecture organised by Ben Fry's seven-stage framework, producing 34 static figures and 6 interactive HTML dashboards.
- Honest reporting of model performance on a small dataset, including cross-validation and ablation studies that reveal the dominant influence of course-level confounding.

### 2.5 Literature Review

Educational data mining has attracted growing scholarly attention over the past decade. Yilmaz and Sekeroglu [1] collected the dataset used in this study from engineering and educational-sciences faculties in Northern Cyprus and applied several machine-learning classifiers, demonstrating that student survey responses contain predictive signal for end-of-term grades. Their work established the baseline for computational modelling on this cohort.

The methodological framework adopted in this report follows Ben Fry's *Visualizing Data* [2], which decomposes the visualisation process into seven sequential stages: Acquire, Parse, Filter, Mine, Represent, Refine, and Interact. This framework ensures that data transformation, statistical analysis, and visual encoding proceed in a disciplined, reproducible manner.

On the computational side, the analysis leverages the open-source Python ecosystem. Matplotlib [3] provides the foundational 2-D plotting engine; Seaborn [4] extends it with statistical graphics and integrated theming; Pandas [6] supplies the data-frame abstraction; and Scikit-learn [5] offers preprocessing, cross-validation, and baseline classifiers. For deep learning, the Keras API within TensorFlow [8] enables rapid prototyping of multilayer perceptrons, while the theoretical foundations for regularisation, dropout, and batch normalisation are drawn from Goodfellow, Bengio, and Courville [7].

Collectively, these references establish both the domain context and the computational toolkit for the investigation that follows.

### 2.6 Methodological Framework: Ben Fry's Seven Stages

The entire exploratory data analysis and visualisation pipeline was organised according to the seven stages defined by Ben Fry [2]:

1. **Acquire:** Load the raw CSV dataset from local storage or a remote GitHub URL, and verify its tabular geometry (Section 4 of the notebook).
2. **Parse:** Rename ambiguous numeric column headers to domain-meaningful identifiers, decode integer codes into human-readable categorical labels, and set appropriate data types (Section 4).
3. **Filter:** Audit data quality (missing values, duplicates, domain-range violations, rare categories), detect and justify outlier retention, and engineer composite behavioural metrics (Section 5).
4. **Mine:** Execute descriptive statistics, hypothesis testing, probability distribution fitting, correlation analysis, and conditional probability estimation (Section 5-6).
5. **Represent:** Translate statistical structures into a comprehensive atlas of 34 static publication-grade figures across univariate, bivariate, and multivariate dimensions (Section 6).
6. **Refine:** Apply systematic design standards---informative titles, labelled axes, colourblind-safe palettes, annotated counts, 300-DPI export---and demonstrate the transformation with a before-versus-after example (Section 6.6).
7. **Interact:** Deliver six standalone Plotly HTML dashboards enabling dynamic filtering, brushing, and 3-D projection (Section 6.6).

### 2.7 Organisation of the Report

The remainder of this report is organised as follows. Section 3 describes the dataset provenance, structure, and quality. Section 4 details the exploratory data analysis and preprocessing pipeline. Section 5 presents the complete data visualisation suite. Section 6 formulates the deep learning and baseline classification models. Section 7 reports and interprets the evaluation results. Section 8 offers conclusions, limitations, practical recommendations, and future research directions. References and appendices follow.

---

\newpage

## 3. DATASET DESCRIPTION

### 3.1 Source and Collection Context

The dataset used in this study was obtained from the UCI Machine Learning Repository under the title "Higher Education Students Performance Evaluation" [1]. It was collected in 2019 by Yilmaz and Sekeroglu at a university in Northern Cyprus, drawing students from the Faculty of Engineering and the Faculty of Educational Sciences. Each student completed a standardised survey of 30 questions covering personal demographics, family socioeconomic background, and academic study habits. The responses were linked to the student's officially recorded end-of-term letter grade (the target variable) and the academic course in which they were enrolled.

### 3.2 Dataset Geometry

The raw dataset comprises **145 student records** (rows) and **33 columns**:

- **1 identifier column:** `STUDENT ID` (arbitrary labels STUDENT1 through STUDENT145).
- **30 survey columns:** Originally headed with the integers "1" through "30", each encoding a categorical survey response as an integer code.
- **1 institutional context variable:** `COURSE ID`, an integer from 1 to 9 identifying the academic course offering.
- **1 target variable:** `GRADE`, an integer from 0 to 7 representing the student's end-of-term letter grade.

All values in the dataset are integer-encoded. There are no floating-point, free-text, or date fields.

### 3.3 Feature Blocks

The 30 survey questions naturally divide into three thematic blocks:

1. **Personal demographics (Q1-Q10):** Age, sex, high-school type, scholarship level, additional employment, extracurricular activity participation, romantic partnership status, monthly salary, primary transportation mode, and accommodation type.
2. **Family background (Q11-Q16):** Mother's education, father's education, number of siblings, parental marital status, mother's occupation, and father's occupation.
3. **Study habits and academic engagement (Q17-Q30):** Weekly self-study hours, non-scientific reading frequency, scientific reading frequency, seminar/conference attendance, perceived project impact, lecture attendance, midterm preparation companions, midterm preparation timing, note-taking frequency, active listening, classroom discussion participation, flipped-classroom perception, prior cumulative GPA, and expected end-of-term GPA.

### 3.4 Data Dictionary

Table 1 presents the complete data dictionary. For each feature, the table lists the original column header, the renamed semantic identifier, the measurement type (nominal, ordinal, or binary), and the decoded categorical levels.

**Table 1. Complete Data Dictionary: Features, Types, and Categorical Levels**

| Original Header | Renamed Feature | Type | Decoded Levels |
|:---:|:---|:---:|:---|
| STUDENT ID | Student_ID | Identifier | STUDENT1 - STUDENT145 |
| 1 | Age | Ordinal | 18-21, 22-25, 26+ |
| 2 | Sex | Binary | Female, Male |
| 3 | HS_Type | Nominal | Private, State, Other |
| 4 | Scholarship | Ordinal | None, 25%, 50%, 75%, Full |
| 5 | Additional_Work | Binary | Yes, No |
| 6 | Activity | Binary | Yes, No |
| 7 | Partner | Binary | Yes, No |
| 8 | Salary | Ordinal | USD 135-200, 201-270, 271-340, 341-410, >410 |
| 9 | Transport | Nominal | Bus, Car/Taxi, Bicycle, Other |
| 10 | Accommodation | Nominal | Rental, Dormitory, Family, Other |
| 11 | Mother_Edu | Ordinal | Primary, Secondary, High School, University, MSc, PhD |
| 12 | Father_Edu | Ordinal | Primary, Secondary, High School, University, MSc, PhD |
| 13 | Siblings | Ordinal | 1, 2, 3, 4, 5+ |
| 14 | Parental_Status | Nominal | Married, Divorced, Died |
| 15 | Mother_Occ | Nominal | Retired, Housewife, Govt Officer, Private Employee, Self-Employed |
| 16 | Father_Occ | Nominal | Retired, Govt Officer, Private Employee, Self-Employed, Other |
| 17 | Study_Hours | Ordinal | None, <5 hours, 6-10 hours, 11-20 hours, >20 hours |
| 18 | Read_NonSci | Ordinal | None, Sometimes, Often |
| 19 | Read_Sci | Ordinal | None, Sometimes, Often |
| 20 | Seminars | Binary | Yes, No |
| 21 | Project_Impact | Nominal | Positive, Negative, Neutral |
| 22 | Attendance | Ordinal | Always, Sometimes (code 3 "Never" has 0 occurrences) |
| 23 | Midterm_Prep_With | Nominal | Alone, With Friends, Not Applicable |
| 24 | Midterm_Prep_Timing | Ordinal | Closest to Exam, Regularly, Never |
| 25 | Note_Taking | Ordinal | Never, Sometimes, Always |
| 26 | Listening | Ordinal | Never, Sometimes, Always |
| 27 | Discussion | Ordinal | Never, Sometimes, Always |
| 28 | Flip_Classroom | Nominal | Not Useful, Useful, Not Applicable |
| 29 | GPA_Last | Ordinal | <2.00, 2.00-2.49, 2.50-2.99, 3.00-3.49, >3.49 |
| 30 | GPA_Expected | Ordinal | <2.00, 2.00-2.49, 2.50-2.99, 3.00-3.49 (code 5 >3.49 has 0 occurrences) |
| COURSE ID | Course_ID | Nominal | Course 1 - Course 9 |
| GRADE | Grade | Ordinal (Target) | 0: Fail, 1: DD, 2: DC, 3: CC, 4: CB, 5: BB, 6: BA, 7: AA |

### 3.5 Target Variable Distribution

The target variable `GRADE` takes eight integer values from 0 (Fail) to 7 (AA). Table 2 presents the complete distribution.

**Table 2. Target Variable (GRADE) Distribution: Counts and Percentages**

| Grade Code | Letter Grade | Count | Percentage |
|:---:|:---:|:---:|:---:|
| 0 | Fail | 8 | 5.5% |
| 1 | DD | 35 | 24.1% |
| 2 | DC | 24 | 16.6% |
| 3 | CC | 21 | 14.5% |
| 4 | CB | 10 | 6.9% |
| 5 | BB | 17 | 11.7% |
| 6 | BA | 13 | 9.0% |
| 7 | AA | 17 | 11.7% |
| **Total** | | **145** | **100.0%** |

The distribution exhibits notable characteristics. The modal grade is DD (code 1) with 35 students (24.1%), indicating that nearly one-quarter of the cohort achieved the lowest passing mark. A secondary clustering appears at the top of the scale: BB (17 students, 11.7%) and AA (17 students, 11.7%) together constitute 23.4% of the sample. The mean grade is 3.228 and the median is 3 (CC), with a sample variance of 4.830 and a standard deviation of 2.198. The distribution is thus bimodal rather than unimodal, which has important implications for probability distribution fitting (see Section 5.3).

### 3.6 Grade_Group Target Tier Definition

For classification modelling, the eight-class target was consolidated into three ordinal tiers, as shown in Table 3.

**Table 3. Grade_Group Target Tier Definition and Class Frequencies**

| Grade_Group | Constituent Grades | Interpretation | Count (n) | Percentage |
|:---|:---|:---|:---:|:---:|
| Low | 0 (Fail), 1 (DD), 2 (DC) | Below-average attainment | 67 | 46.2% |
| Medium | 3 (CC), 4 (CB) | Average attainment | 31 | 21.4% |
| High | 5 (BB), 6 (BA), 7 (AA) | Above-average attainment | 47 | 32.4% |

The Low tier is the largest class (46.2%), followed by High (32.4%) and Medium (21.4%). This class imbalance necessitates balanced class weighting during model training and the use of macro-averaged evaluation metrics to avoid majority-class bias.

Additionally, a binary **Pass** flag was created: Pass = 1 if Grade >= 1 (i.e., any letter grade above Fail), and Pass = 0 otherwise. The pass rate is 94.5% (137 of 145 students), with a 95% Wilson score confidence interval of [89.5%, 97.2%].

### 3.7 Course Distribution

Table 4 presents the enrolment distribution across the nine course offerings.

**Table 4. Course ID Enrolment Distribution**

| Course ID | Student Count | Percentage |
|:---:|:---:|:---:|
| 1 | 66 | 45.5% |
| 2 | 2 | 1.4% |
| 3 | 8 | 5.5% |
| 4 | 4 | 2.8% |
| 5 | 7 | 4.8% |
| 6 | 8 | 5.5% |
| 7 | 15 | 10.3% |
| 8 | 14 | 9.7% |
| 9 | 21 | 14.5% |

Course 1 accounts for nearly half of the dataset (45.5%), while Course 2 contains only two students. The strong association between course assignment and grade outcome is discussed in detail in Section 5.4: all 14 students in Course 8 received a grade of DC or below, while 13 of the 15 students in Course 7 attained BA or AA. This institutional confounding was treated explicitly through ablation modelling (Section 7).

### 3.8 Data Quality Summary

Table 5 summarises the data quality audit.

**Table 5. Data Quality Audit Summary**

| Quality Dimension | Finding |
|:---|:---|
| Missing values | 0 across all 33 columns (100% complete) |
| Duplicate rows (full matrix) | 0 |
| Duplicate rows (excluding Student ID) | 0 |
| Domain range violations | None; all integer codes fall within documented bounds |
| Codes that never occur | Q15 code 6: 0 occurrences; Q22 code 3 ("Never"): 0 occurrences; Q30 code 5 (>3.49): 0 occurrences |
| Encoding artefacts | Column headers originally stored as string numerals ("1" to "30"); no Unicode corruption detected |
| Anonymisation | Student identifiers are arbitrary codes (STUDENT1-STUDENT145) with no personally identifiable information |

The dataset is fully clean and requires no imputation. The absence of the "Never" attendance code (Q22, code 3) means that all 145 students reported attending lectures either "Always" (110 students, 75.9%) or "Sometimes" (35 students, 24.1%). Similarly, no student expected a GPA above 3.49 (Q30, code 5), reflecting conservative academic self-assessment across the cohort.

---

\newpage

## 4. EXPLORATORY DATA ANALYSIS AND PREPROCESSING

### 4.1 Preprocessing Pipeline

Table 6 presents the complete preprocessing pipeline in a structured step-technique-result format.

**Table 6. EDA and Preprocessing Pipeline: Step, Technique, and Result**

| Step | Technique | Result |
|:---|:---|:---|
| 1. Data ingestion | `pd.read_csv()` from `data/student_performance.csv` | 145 rows x 33 columns loaded successfully |
| 2. Column renaming | Dictionary mapping of original headers to semantic names | 33 columns renamed (e.g., "2" to "Sex", "COURSE ID" to "Course_ID") |
| 3. Label decoding | Integer codes mapped to human-readable strings using `dict_labels` | Two parallel DataFrames maintained: `df_clean` (numeric), `df_decoded` (categorical strings) |
| 4. Missing-value audit | `df.isnull().sum()` across all columns | 0 missing values confirmed |
| 5. Duplicate audit | `df.duplicated()` with and without Student ID | 0 duplicates in both cases |
| 6. Domain range validation | Unique-value check against data dictionary bounds | All values within documented ranges; 3 absent codes noted (Q15 code 6, Q22 code 3, Q30 code 5) |
| 7. Outlier assessment | IQR (1.5x) and z-score (abs(z) > 3) on engineered scores | IQR flags: Study_Engagement (4), Class_Behaviour (3), Family_Education (1), Aspiration_Gap (7); z-score outliers: 0. All retained. |
| 8. Feature engineering | Composite score formulas (Table 7) | 4 engineered continuous scores + Grade_Group + Pass flag created |
| 9. One-hot encoding | Nominal features encoded with `pd.get_dummies(drop_first=True)` | 57 input dimensions after encoding |
| 10. Stratified split | 70/15/15 train/val/test with `stratify=Grade_Group` | Train: 101, Validation: 22, Test: 22 samples |
| 11. Standardisation | `StandardScaler` fitted on training split only | Applied transformatively to validation and test sets (no data leakage) |
| 12. Class weighting | `compute_class_weight('balanced')` from training labels | Weights: Low = 0.716, Medium = 1.603, High = 1.020 |

### 4.2 Outlier Treatment Justification

The IQR-based outlier rule (1.5 x IQR below Q1 or above Q3) flagged a total of 15 data points across the four engineered scores. However, all four scores are bounded composites of Likert-type ordinal survey items. Such bounded scales cannot produce true measurement outliers in the statistical sense; the extreme values represent genuine student behavioural archetypes (e.g., a student who studies extensively across all dimensions, or one who consistently reports minimal engagement). Moreover, with only 145 records, removing any observations would significantly reduce statistical power and potentially eliminate precisely those at-risk students who are the focus of the early-warning application. No z-score exceeded the abs(z) > 3 threshold. Therefore, all 145 rows were retained for analysis.

### 4.3 Feature Engineering

Table 7 documents the formulas for the four engineered composite metrics.

**Table 7. Engineered Composite Feature Formulas**

| Engineered Feature | Formula | Range | Interpretation |
|:---|:---|:---:|:---|
| Study_Engagement | Study_Hours + Read_Sci + Read_NonSci + Seminars_Score (Seminars: Yes=2, No=1) | 4-13 | Higher values indicate greater academic engagement outside the classroom |
| Class_Behaviour | (4 - Attendance) + Note_Taking + Listening + Discussion | 6-12 | Higher values indicate stronger in-class behavioural compliance and participation |
| Family_Education | Mother_Edu + Father_Edu | 2-12 | Higher values indicate greater parental educational capital |
| Aspiration_Gap | GPA_Expected - GPA_Last | -4 to +4 | Positive values indicate optimistic self-assessment; negative values indicate realistic or cautious expectations |

Note that for the Attendance variable, the raw code uses 1 = Always (best) and 3 = Never (worst), so the transformation (4 - Attendance) reverse-codes it so that higher numerical values consistently represent more desirable behaviour across all constituent items.

The descriptive statistics for the engineered scores are:

- **Study_Engagement:** Mean = 7.94, Median = 8.0, Std = 1.46, Range = [4, 13].
- **Class_Behaviour:** Mean = 9.75, Median = 10.0, Std = 1.28, Range = [6, 12].
- **Family_Education:** Mean = 4.92, Median = 5.0, Std = 2.01, Range = [2, 11].
- **Aspiration_Gap:** Mean = -0.40, Median = 0.0, Std = 0.98, Range = [-3, 2].

The negative mean of Aspiration_Gap (-0.40) indicates that students on average expected their end-of-term GPA to be slightly lower than their prior cumulative standing, reflecting realistic or mildly pessimistic academic self-calibration.

### 4.4 Stratified Data Split

The dataset was partitioned into three non-overlapping subsets using stratified sampling on Grade_Group with a fixed random seed of 42.

**Table 8. Stratified Data Split: Sizes and Class Distributions**

| Split | Total Samples | % of Dataset | Low (n) | Medium (n) | High (n) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Train | 101 | 69.7% | 47 | 21 | 33 |
| Validation | 22 | 15.2% | 10 | 5 | 7 |
| Test | 22 | 15.2% | 10 | 5 | 7 |
| **Total** | **145** | **100.0%** | **67** | **31** | **47** |

The class proportions are preserved across all three splits. The `StandardScaler` was fitted exclusively on the 101 training samples and applied transformatively to the validation and test sets, ensuring zero data leakage. Balanced class weights were computed from the training labels to counteract the class imbalance during model optimisation.

### 4.5 Summary of Main EDA Insights

The key EDA findings are summarised below:

1. **Prior academic trajectory dominates:** Prior Cumulative GPA (GPA_Last) is the single strongest monotonic predictor of final grade (Spearman rho = +0.351, p < 0.001).
2. **Attendance is necessary but not sufficient:** Students attending lectures only "Sometimes" have a 62.9% probability of falling into the Low attainment tier, compared to 40.9% for regular attenders.
3. **Study hours exhibit a non-linear plateau:** Median grade improves from approximately 1.5 (DD) to 3.0 (CC) between zero and 6-10 weekly study hours, but does not increase further for >20 hours.
4. **Course assignment is a powerful institutional confounder:** 100% of Course 8 students scored DC or below, while 86.7% of Course 7 students attained BA or AA.
5. **GPA features are multicollinear:** Prior GPA and Expected GPA exhibit a Pearson correlation of 0.658, warranting ablation analysis.
6. **The grade distribution is overdispersed:** The dispersion index (variance/mean = 1.496) exceeds 1.0, violating equidispersed Poisson and Binomial assumptions.

---

*End of Part 1 (Front matter + Sections 1-4). Please say "continue" for Part 2 (Sections 5-6: Data Visualisation and Deep Learning Model).*
