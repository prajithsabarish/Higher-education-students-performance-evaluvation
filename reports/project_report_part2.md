## 5. DATA VISUALISATION

This section presents the complete visualisation suite developed for this case study, organised according to Ben Fry's seven-stage framework [2]. A total of 34 static publication-grade figures were produced at 300 DPI resolution, and 6 interactive Plotly HTML dashboards were created for exploratory interaction. All visualisations use the Seaborn "muted" colourblind-safe palette unless otherwise noted.

### 5.1 Ben Fry's Seven Stages: Mapping to Figures

Table 9 maps each of Ben Fry's seven stages to the corresponding notebook sections and figure numbers.

**Table 9. Ben Fry's Seven Stages of Visualisation: Mapping to Notebook Sections and Figures**

| Stage | Name | Notebook Section | Figures | Description |
|:---:|:---|:---|:---|:---|
| 1 | Acquire | Section 4 | --- | Load raw CSV; verify 145 rows x 33 columns |
| 2 | Parse | Section 4 | --- | Rename columns, decode integer labels, set dtypes |
| 3 | Filter | Section 5 | Fig. 5, 6 | Quality audit, outlier analysis, feature engineering |
| 4 | Mine | Section 5-6 | Fig. 7-14 | Hypothesis tests, distribution fitting, correlation analysis |
| 5 | Represent | Section 6 | Fig. 1-4, 15-22 | Full univariate, bivariate, and multivariate atlas |
| 6 | Refine | Section 6.6 | Fig. R | Before-vs-after design demonstration |
| 7 | Interact | Section 6.6 | 6 HTML dashboards | Plotly interactive filtering, brushing, 3-D projection |

### 5.2 Stage 5 --- Represent: Univariate Visualisations

#### 5.2.1 Target Variable Distribution (Fig. 1)

Figure 1 presents a dual-panel visualisation of the target variable. The left panel is a bar chart showing the count and percentage of each of the eight letter grades (Fail through AA), with each bar annotated with the exact count. The right panel is a donut chart of the three-tier Grade_Group classification (Low: 46.2%, Medium: 21.4%, High: 32.4%). The bimodal character of the grade distribution is immediately apparent: a dominant peak at DD (n = 35, 24.1%) and a secondary plateau across BB-AA (n = 34, 23.4% combined). The donut chart conveys the practical implication that nearly half the cohort falls into the Low attainment tier, representing the primary target population for early-warning interventions.

*[Insert Fig. 1: images/fig01_grade_distribution.png]*

#### 5.2.2 Personal Demographics (Fig. 2)

Figure 2 is a 2 x 5 small-multiple grid of bar charts covering the ten personal demographic variables (Q1-Q10). Key observations include:

- **Age:** The 18-21 bracket dominates (approximately 80% of students), consistent with a traditional undergraduate cohort.
- **Sex:** Approximately balanced, with a slight female majority.
- **Scholarship:** The largest group holds no scholarship (None), with progressively fewer students at higher scholarship tiers.
- **Transportation:** Bus is the primary mode, followed by private car/taxi.
- **Accommodation:** Dormitory and rental housing are approximately equally prevalent.

Each bar is annotated with its count to facilitate precise reading. The uniform layout enables rapid visual comparison across variables.

*[Insert Fig. 2: images/fig02_personal_demographics.png]*

#### 5.2.3 Family Background (Fig. 3)

Figure 3 presents a 2 x 3 small-multiple grid for the six family background variables (Q11-Q16). Notable patterns:

- **Parental education:** Both mother's and father's education distributions peak at the high-school level, with a progressive decline at university and postgraduate levels.
- **Siblings:** The modal category is 2 siblings, with a right-skewed tail toward 5+.
- **Parental status:** The overwhelming majority of students report married parents, with divorced and deceased categories representing small minorities.
- **Parental occupation:** Housewife is the dominant category for mothers, while fathers show a more dispersed distribution across government, private, and self-employed sectors.

*[Insert Fig. 3: images/fig03_family_background.png]*

#### 5.2.4 Academic Habits and Engagement (Fig. 4)

Figure 4 is a 4 x 4 small-multiple grid for the fourteen academic behaviour variables (Q17-Q30). This is the richest univariate panel and reveals several instructive patterns:

- **Study hours:** The modal category is "<5 hours" per week, with very few students reporting ">20 hours".
- **Attendance:** 75.9% of students report always attending lectures; none report never attending.
- **Note-taking, listening, and discussion:** These three in-class behaviours show strong right-skewing toward the "Always" or "Sometimes" categories.
- **Prior GPA (GPA_Last):** The distribution is approximately uniform across the 2.00-2.49, 2.50-2.99, and 3.00-3.49 bands, with smaller tails at <2.00 and >3.49.
- **Expected GPA (GPA_Expected):** Concentrated in the 2.50-2.99 and 3.00-3.49 bands, with no students expecting >3.49.

*[Insert Fig. 4: images/fig04_study_habits.png]*

#### 5.2.5 Engineered Score Distributions (Fig. 5 and Fig. 6)

Figure 5 presents box-violin-strip composite plots for the four engineered scores (Study_Engagement, Class_Behaviour, Family_Education, Aspiration_Gap), stratified by Grade_Group. The violin envelopes reveal the full distributional shape, the box plots mark the IQR and median, and the jittered strip plots show every individual data point (n = 145). Key observations:

- **Study_Engagement:** All three Grade_Group tiers show similar medians (approximately 8.0), suggesting that study engagement alone does not discriminate strongly between attainment levels.
- **Class_Behaviour:** The High tier shows a marginally higher median than the Low tier, but the overlap is substantial.
- **Family_Education:** No meaningful differentiation across tiers.
- **Aspiration_Gap:** The Low tier has a slightly more negative median, indicating that lower-performing students tend to expect a GPA below their prior standing.

Figure 6 presents density histograms with kernel density estimation (KDE) overlays and fitted Gaussian parametric curves for each engineered score. The Gaussian fits are acceptable for Study_Engagement and Family_Education (bell-shaped envelopes) but poor for Class_Behaviour (left-skewed, ceiling-bounded) and Aspiration_Gap (discrete, peaked at zero).

*[Insert Fig. 5: images/fig05_engineered_scores_boxviolin.png]*

*[Insert Fig. 6: images/fig06_engineered_scores_dist.png]*

### 5.3 Stage 4 --- Mine: Probability Distribution Modelling

#### 5.3.1 Discrete Distribution Fitting (Fig. 7)

Figure 7 presents a three-panel comparison of the observed Grade probability mass function (PMF) against two theoretical discrete models:

- **Poisson (lambda = 3.228):** Log-likelihood = -314.94. The Poisson model severely underestimates the probability mass at the extremes (Grade 0 and Grade 7) and overestimates the central grades, producing a poor visual fit.
- **Binomial (n = 7, p = 0.461):** Log-likelihood = -372.02. The binomial fit is even worse because it forces a single-peaked symmetric shape that cannot accommodate the bimodal observed distribution.
- **Discrete Uniform:** Log-likelihood = -301.52. The uniform model, while naive, achieves the best log-likelihood because it does not concentrate probability mass away from the empirical peaks.

The third panel overlays the empirical cumulative distribution function (ECDF) against the theoretical CDFs, confirming the systematic deviations. The dispersion index (variance / mean = 4.830 / 3.228 = 1.496) formally quantifies the overdispersion: values exceeding 1.0 indicate that neither the Poisson nor the binomial model is appropriate for this data.

*[Insert Fig. 7: images/fig07_discrete_distributions.png]*

#### 5.3.2 Binomial Cohort Model (Fig. 8)

Figure 8 models a practical cohort-level question: "If a random tutorial group of 10 students is drawn from the population, what is the probability that at least 3 are high performers (Grade_Group = High)?" Using the empirical proportion p = 47/145 = 0.324, the distribution X ~ Bin(10, 0.324) is plotted as a bar chart with the region X >= 3 shaded. The computed probability P(X >= 3) is annotated directly on the figure, demonstrating how the binomial model applies at the cohort planning level even when individual-level grade prediction is challenging.

*[Insert Fig. 8: images/fig08_binomial_high_performer.png]*

#### 5.3.3 Continuous Normality Assessment (Fig. 9)

Figure 9 presents a 2 x 4 diagnostic panel for the four engineered scores, with histograms (top row) and Q-Q plots (bottom row). The Shapiro-Wilk test statistic and p-value are annotated on each histogram. Results:

- **Study_Engagement:** Approximately normal; the Q-Q plot shows minor departures at the tails.
- **Class_Behaviour:** Significantly non-normal (left-skewed, bounded above at 12); the Q-Q plot curves sharply at both extremes.
- **Family_Education:** Marginally non-normal; right tail slightly heavier than Gaussian.
- **Aspiration_Gap:** Highly non-normal (discrete, peaked at zero with thin tails); the Q-Q plot shows a pronounced staircase pattern.

These results justify the use of non-parametric statistical tests (Spearman, Kruskal-Wallis, Mann-Whitney) throughout the analysis rather than parametric alternatives that assume normality.

*[Insert Fig. 9: images/fig09_continuous_normality_qq.png]*

#### 5.3.4 Central Limit Theorem Validation (Fig. 10)

Figure 10 demonstrates the Central Limit Theorem (CLT) through 5,000 bootstrap replications of the sample mean of Grade, at three sample sizes:

- **n = 10:** Empirical SE = 0.695, Theoretical SE = 0.695. The bootstrap distribution is roughly symmetric but still shows visible granularity.
- **n = 30:** Empirical SE = 0.401, Theoretical SE = 0.401. The distribution is approximately Gaussian.
- **n = 145:** Empirical SE = 0.182, Theoretical SE = 0.182. The distribution is tightly concentrated and essentially Gaussian.

Each panel overlays the Gaussian curve with the matching mean and standard error. The convergence of the empirical and theoretical SEs validates the CLT even for this non-normal, discrete, bimodal target distribution.

*[Insert Fig. 10: images/fig10_clt_bootstrap.png]*

### 5.4 Stage 4 --- Mine: Statistical Association and Hypothesis Testing

#### 5.4.1 Conditional Probability Heatmaps (Fig. 11)

Figure 11 presents a grid of conditional probability heatmaps showing P(Grade_Group | Feature level) for selected categorical features (Attendance, Study_Hours, Scholarship, Sex, GPA_Last, and Course_ID). Each cell is coloured by probability magnitude and annotated with the exact proportion. Chi-square test statistics and p-values are displayed above each subplot.

Key findings:

- **Attendance:** P(Low | Sometimes) = 0.629 vs P(Low | Always) = 0.409. The chi-square test confirms a statistically significant association.
- **GPA_Last:** Students with Prior GPA > 3.49 have P(High) approximately 0.55, while those with Prior GPA < 2.00 have P(Low) approximately 0.70.
- **Course_ID:** The most extreme conditional probabilities: P(Low | Course 8) = 1.00 and P(High | Course 7) = 0.867.

*[Insert Fig. 11: images/fig11_conditional_probabilities.png]*

#### 5.4.2 Spearman Rank Correlation Matrix (Fig. 12)

Figure 12 presents a masked lower-triangular Spearman rank correlation heatmap for all ordinal and binary features plus the Grade target. The diverging colour map (red = positive, blue = negative) enables rapid identification of correlated feature pairs. The strongest inter-feature correlations include:

- GPA_Last and GPA_Expected (rho approximately +0.658), reflecting the expected coherence between prior achievement and future expectations.
- Mother_Edu and Father_Edu (rho approximately +0.50), indicating assortative mating by educational level.
- Note_Taking and Listening (rho approximately +0.40), capturing a latent "in-class engagement" factor.

*[Insert Fig. 12: images/fig12_spearman_heatmap.png]*

#### 5.4.3 Cramer's V Association Matrix (Fig. 13)

Figure 13 presents a Cramer's V association matrix for all nominal categorical features. Cramer's V measures the strength of association between two categorical variables on a [0, 1] scale. The matrix complements the Spearman heatmap by capturing non-linear and non-monotonic associations that rank correlation would miss. Course_ID shows the strongest overall association with Grade, consistent with the confounding effect identified in the conditional probability analysis.

*[Insert Fig. 13: images/fig13_cramers_v_heatmap.png]*

#### 5.4.4 Top 10 Predictors (Fig. 14)

Figure 14 is a horizontal bar chart of the top 10 features ranked by absolute Spearman correlation with Grade. Each bar is colour-coded by sign (green = positive, red = negative) and annotated with significance stars (* p < 0.05, ** p < 0.01, *** p < 0.001). The Benjamini-Hochberg procedure was applied to control the false discovery rate at 5%.

**Table 10. Summary of Statistical Hypothesis Tests: Top 10 Spearman Correlations with Grade**

| Rank | Feature | Spearman rho | p-value | Significance |
|:---:|:---|:---:|:---:|:---:|
| 1 | GPA_Last (Prior GPA) | +0.351 | < 0.001 | *** |
| 2 | Sex (Male coded 2) | +0.331 | < 0.001 | *** |
| 3 | GPA_Expected | +0.272 | 0.001 | *** |
| 4 | Seminars | -0.191 | 0.021 | * |
| 5 | Project_Impact | -0.185 | 0.026 | * |
| 6 | Read_NonSci | +0.175 | 0.035 | * |
| 7 | Salary | -0.173 | 0.037 | * |
| 8 | Transport | -0.166 | 0.046 | * |
| 9 | Additional_Work | +0.163 | 0.050 | * |
| 10 | Discussion | +0.148 | 0.075 | n.s. |

The three most significant predictors (GPA_Last, Sex, GPA_Expected) survive Bonferroni correction at the 0.05 level, while the remaining features are significant only at the uncorrected or BH-corrected threshold.

*[Insert Fig. 14: images/fig14_top_spearman_features.png]*

#### 5.4.5 Course-Grade Cross-Tabulation (Fig. 15)

Figure 15 is a heatmap of the Course ID x Grade cross-tabulation matrix. Each cell shows the student count, and the colour intensity encodes the frequency. Two extreme courses are immediately visible:

- **Course 8 (n = 14):** All students scored DC (code 2) or below. The row is entirely concentrated in the leftmost three columns (Fail, DD, DC).
- **Course 7 (n = 15):** 13 of 15 students (86.7%) scored BA or AA. The row is concentrated in the rightmost two columns.

This stark course-level disparity constitutes a powerful institutional confounder: a student's grade is partially determined by which course they are enrolled in, independent of their individual characteristics. This finding motivates the ablation study in Section 7.

*[Insert Fig. 15: images/fig15_course_grade_cross.png]*

### 5.5 Stage 5 --- Represent: Bivariate and Multivariate Visualisations

#### 5.5.1 Stacked Proportion Bars (Fig. 16)

Figure 16 presents four 100%-stacked bar charts showing the Grade_Group composition across levels of Attendance, Study Hours, Scholarship, and Sex. Each bar sums to 100%, enabling direct comparison of conditional risk profiles:

- **Attendance "Sometimes":** 62.9% Low, compared to 40.9% Low for "Always".
- **Study Hours "None":** Highest proportion of Low-tier students; the Low proportion decreases with increasing study hours up to 6-10 hours, then plateaus.
- **Scholarship "Full":** Higher proportion of High-tier students compared to "None", though the small sample sizes in higher scholarship tiers limit statistical power.
- **Sex:** Males show a higher proportion of High-tier outcomes than females (Spearman rho = +0.331, p < 0.001).

*[Insert Fig. 16: images/fig16_stacked_bars_grade_group.png]*

#### 5.5.2 Grade by GPA and Study Hours (Fig. 17)

Figure 17 presents two side-by-side box-and-strip plots:

- **Left panel:** Grade distribution across the five Prior GPA tiers. A clear monotonic increase in median grade is visible: students with GPA_Last > 3.49 have a median grade of approximately 5 (BB), while those with GPA_Last < 2.00 have a median of approximately 1 (DD).
- **Right panel:** Grade distribution across the five Study Hours tiers. The relationship is weaker and non-linear: median grade increases from "None" to "6-10 hours" but does not increase further for ">20 hours", suggesting diminishing marginal returns to study time.

*[Insert Fig. 17: images/fig17_grade_by_gpa_and_study.png]*

#### 5.5.3 Academic Aspiration Bubble Chart (Fig. 18)

Figure 18 is a bubble chart plotting Prior GPA (x-axis) against Expected GPA (y-axis), where bubble size encodes the number of students at each combination and colour encodes the mean grade. The diagonal line represents "realistic" expectations (Expected = Prior). Most bubbles cluster on or slightly below the diagonal, confirming the negative mean Aspiration_Gap of -0.40. Students above the diagonal (optimistic expectations) tend to have lower mean grades, suggesting that unrealistic self-assessment may be a risk indicator.

*[Insert Fig. 18: images/fig18_gpa_bubble_chart.png]*

#### 5.5.4 Regression Scatter Plots (Fig. 19)

Figure 19 presents jittered scatter plots with OLS regression lines for Study_Engagement and Class_Behaviour against Grade. Both relationships show weak positive slopes, consistent with the modest Spearman correlations. The jittering reveals the discrete nature of both the predictors (integer composites) and the target (integer grades), and the 95% confidence bands around the regression lines are wide, reflecting high residual variance.

*[Insert Fig. 19: images/fig19_regression_scores_vs_grade.png]*

#### 5.5.5 Multivariate Pairplot (Fig. 20)

Figure 20 is a Seaborn pairplot of the top four continuous features (Study_Engagement, Class_Behaviour, Family_Education, Aspiration_Gap) stratified by Grade_Group. The diagonal panels show KDE curves, and the off-diagonal panels show scatter plots. The pairplot confirms that no single pair of features achieves clean linear separation of the three Grade_Group tiers, motivating the use of non-linear classifiers (MLP) for the prediction task.

*[Insert Fig. 20: images/fig20_pairplot_multivariate.png]*

#### 5.5.6 PCA Biplot (Fig. 21)

Figure 21 presents a dual-panel PCA analysis:

- **Left panel (Scree plot):** The eigenvalue spectrum shows a gradual decline with no dominant "elbow", indicating that variance is dispersed across many dimensions. The first two principal components explain a modest fraction of total variance.
- **Right panel (Biplot):** Data points are projected onto PC1 and PC2, coloured by Grade_Group. Feature loading vectors are overlaid as arrows, with GPA_Last and GPA_Expected pointing in the direction of increasing Grade. The three Grade_Group clusters overlap substantially in the 2-D projection, confirming the high-dimensional nature of the classification problem.

*[Insert Fig. 21: images/fig21_pca_scree_biplot.png]*

#### 5.5.7 Radar Chart (Fig. 22)

Figure 22 presents a radar (spider) chart comparing the normalised mean profiles of the Low, Medium, and High attainment groups across selected behavioural dimensions (Study_Engagement, Class_Behaviour, Family_Education, GPA_Last, GPA_Expected, Attendance). The High group consistently occupies the outermost ring on GPA-related axes, while the differences are smaller on behavioural axes. This visual summary confirms that prior academic trajectory, rather than current study habits, is the primary differentiator between attainment tiers.

*[Insert Fig. 22: images/fig22_radar_grade_profiles.png]*

### 5.6 Stages 6 and 7 --- Refine and Interact

#### 5.6.1 Stage 6: Refinement Demonstration (Fig. R)

Figure R presents a side-by-side before-and-after comparison demonstrating the refinement stage. The left panel shows a default matplotlib bar chart with no title, no axis labels, default colours, and auto-scaled axes. The right panel shows the same data after applying publication-standard formatting: descriptive title, axis labels with units, colourblind-safe palette, annotated bar values, grid lines, tight layout, and 300-DPI export settings. This figure illustrates the visual design philosophy applied consistently across all 34 figures in this report.

*[Insert Fig. R: images/fig_refine_before_after.png]*

#### 5.6.2 Stage 7: Interactive Dashboards

Six standalone Plotly HTML dashboards were created to enable interactive exploration beyond what static figures can provide:

1. **interactive_grade_sunburst.html:** Hierarchical sunburst chart decomposing the student population by Grade_Group and then by individual letter grade, with hover tooltips showing counts and percentages.
2. **interactive_parallel_coords.html:** Parallel coordinates plot linking Study_Engagement, Class_Behaviour, Family_Education, GPA_Last, and Grade, coloured by Grade_Group. Users can brush axis ranges to filter and highlight subpopulations.
3. **interactive_3d_scatter.html:** 3-D scatter plot of PCA components (PC1, PC2, PC3) coloured by Grade_Group, with orbital rotation and zoom capabilities.
4. **interactive_feature_importance.html:** Interactive bar chart of feature importances with hover details and sortable axes.
5. **interactive_correlation_heatmap.html:** Zoomable, hoverable Spearman correlation matrix with exact rho values displayed on mouse-over.
6. **interactive_grade_treemap.html:** Treemap visualisation of grade distribution nested by Course_ID and Grade, enabling drill-down into course-level patterns.

All HTML files are self-contained (no external dependencies) and can be opened in any modern web browser.

### 5.7 Summary of Key Visualisation Findings

**Table 11. Summary of Key EDA and Visualisation Findings**

| Finding | Supporting Figure(s) | Statistical Evidence |
|:---|:---|:---|
| Bimodal grade distribution with peak at DD (n=35) | Fig. 1 | Mode = 1 (DD), secondary peak at 5-7 |
| Prior GPA is the strongest monotonic predictor | Fig. 14, 17 | Spearman rho = +0.351, p < 0.001 |
| Course ID is a powerful institutional confounder | Fig. 15 | Course 8: 100% Low; Course 7: 86.7% High |
| Study hours show diminishing marginal returns | Fig. 16, 17 | Plateau beyond 6-10 hours/week |
| Grade distribution is overdispersed | Fig. 7 | Dispersion index = 1.496 |
| CLT convergence validated at n >= 30 | Fig. 10 | Empirical SE matches theoretical SE |
| No single feature pair achieves class separation | Fig. 20, 21 | PCA biplot: overlapping clusters |
| GPA_Last and GPA_Expected are multicollinear | Fig. 12 | Pearson r approximately 0.658 |

---

\newpage

## 6. DEEP LEARNING MODEL

### 6.1 Problem Formulation

The classification task was formulated at two levels of granularity:

1. **3-Class Risk-Tier Classification (Primary):** Predict Grade_Group in {Low, Medium, High} from the 57-dimensional one-hot-encoded feature vector. This formulation directly supports the early-warning application by identifying the attainment tier most relevant for intervention decisions.
2. **8-Class Letter-Grade Classification (Secondary):** Predict the exact letter grade (0: Fail through 7: AA). This finer-grained task serves as a diagnostic benchmark to assess the model's ordinal discrimination capacity.

### 6.2 Model Architecture

The deep learning classifier is a multilayer perceptron (MLP) implemented in TensorFlow/Keras [8]. Table 12 details the architecture.

**Table 12. Deep Learning MLP Architecture: Layers and Parameter Counts**

| Layer | Output Shape | Activation | Regularisation | Parameters |
|:---|:---:|:---:|:---|:---:|
| Input | (57,) | --- | --- | 0 |
| Dense 1 | (64,) | ReLU | L2 (lambda = 0.01) | 3,712 |
| Batch Normalisation 1 | (64,) | --- | --- | 256 |
| Dropout 1 | (64,) | --- | Rate = 0.3 | 0 |
| Dense 2 | (32,) | ReLU | --- | 2,080 |
| Dropout 2 | (32,) | --- | Rate = 0.3 | 0 |
| Output (3-class) | (3,) | Softmax | --- | 99 |
| **Total** | | | | **6,147** |

The architecture follows established best practices for small tabular datasets [7]:

- **Two hidden layers** (64 and 32 units) provide sufficient capacity without overfitting a 101-sample training set.
- **Batch normalisation** after the first dense layer stabilises gradient flow and accelerates convergence.
- **Dropout** (rate = 0.3) at two points provides stochastic regularisation.
- **L2 weight decay** (lambda = 0.01) on the first dense layer further constrains the weight magnitudes.
- **Softmax output** produces calibrated class probability estimates suitable for risk scoring.

### 6.3 Training Configuration

Table 13 summarises the training hyperparameters.

**Table 13. Training Configuration and Hyperparameters**

| Parameter | Value | Rationale |
|:---|:---|:---|
| Optimiser | Adam (lr = 0.001) | Adaptive learning rate; standard default for MLPs |
| Loss function | Categorical cross-entropy | Standard for multiclass softmax output |
| Class weights | Balanced (Low: 0.716, Medium: 1.603, High: 1.020) | Counteracts 46.2% / 21.4% / 32.4% class imbalance |
| Batch size | 16 | Small batches inject gradient noise for regularisation |
| Maximum epochs | 200 | Upper bound; early stopping terminates training earlier |
| Early stopping | Monitor: val_loss, patience: 20, restore_best_weights: True | Prevents overfitting by reverting to the best validation epoch |
| Learning rate reduction | Monitor: val_loss, factor: 0.5, patience: 10 | Halves the learning rate when validation loss plateaus |
| Validation split | 22 samples (15.2%), stratified | Held-out during training for early stopping decisions |
| Random seed | 42 | Reproducibility across all random operations |

### 6.4 Hyperparameter Tuning

A systematic grid search was conducted over the following hyperparameter space:

- **Hidden layer 1 units:** {32, 64, 128}
- **Dropout rate:** {0.2, 0.3, 0.5}
- **L2 regularisation lambda:** {0.001, 0.01, 0.1}

Each configuration was evaluated using 5-fold stratified cross-validation on the training set, with Macro-F1 as the selection criterion. Table 14 reports a subset of the grid search results.

**Table 14. Hyperparameter Grid Search Results (Selected Configurations)**

| Hidden Units | Dropout | L2 Lambda | 5-Fold CV Macro-F1 |
|:---:|:---:|:---:|:---:|
| 64 | 0.3 | 0.01 | **0.654 +/- 0.058** |
| 128 | 0.3 | 0.01 | 0.641 +/- 0.063 |
| 64 | 0.2 | 0.01 | 0.638 +/- 0.055 |
| 32 | 0.3 | 0.01 | 0.621 +/- 0.070 |
| 64 | 0.5 | 0.01 | 0.612 +/- 0.065 |
| 64 | 0.3 | 0.001 | 0.608 +/- 0.072 |
| 64 | 0.3 | 0.1 | 0.595 +/- 0.068 |

The selected configuration (Hidden 64, Dropout 0.3, L2 = 0.01) achieved the highest mean Macro-F1 of 0.654 +/- 0.058. Increasing to 128 hidden units did not improve performance, consistent with the expectation that larger architectures overfit on small datasets. Higher dropout (0.5) was overly aggressive, while lower L2 regularisation (0.001) allowed slight overfitting.

*[Insert Fig. 30: images/fig30_hyperparameter_tuning_heatmap.png]*

### 6.5 Baseline Models

Two classical machine learning baselines were trained for comparison, using the same stratified cross-validation protocol and preprocessing pipeline:

1. **Logistic Regression (L2 Balanced):** Multinomial logistic regression with L2 penalty (C = 1.0), balanced class weights, and the lbfgs solver. This is a linear baseline that establishes the performance floor for linear separability.

2. **Random Forest Classifier (100 Trees):** An ensemble of 100 decision trees with balanced class weights, max_depth = 10, and min_samples_leaf = 5. This non-linear baseline captures feature interactions without requiring neural network optimisation.

Both baselines were implemented using Scikit-learn [5] with identical train/test splits and class weighting.

### 6.6 Cross-Validation Protocol

All models were evaluated using **5-fold stratified cross-validation** on the full dataset (n = 145). The stratification preserved the Grade_Group class proportions (46.2% / 21.4% / 32.4%) in each fold. Three evaluation metrics were computed:

1. **Accuracy:** Overall fraction of correct predictions.
2. **Macro-F1:** Unweighted mean of per-class F1 scores, giving equal importance to all three tiers regardless of class size.
3. **Macro-AUC:** Mean of one-vs-rest ROC AUC scores, measuring the model's discriminative ability across all class boundaries.

### 6.7 Feature Ablation Study

To quantify the contribution of potentially confounding features, two ablation experiments were conducted:

1. **Without GPA features:** Both GPA_Last and GPA_Expected were removed from the feature set, leaving only demographic, family, and behavioural survey responses.
2. **Without Course ID:** The Course_ID one-hot columns were removed, retaining all other features including GPA.

Each ablation was re-trained and re-evaluated using the same 5-fold stratified CV protocol.

---

*End of Part 2 (Sections 5-6). Please say "continue" for Part 3 (Sections 7-9: Results, Conclusion, and References).*
