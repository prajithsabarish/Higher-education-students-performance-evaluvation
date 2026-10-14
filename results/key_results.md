# Key Quantitative Results Summary
**Project:** Higher Education Students' Performance Evaluation: EDA, Statistical Visualization and Deep Learning  
**Author:** Prajith Sabarish D (Reg. No. 24AD079), B.Tech Artificial Intelligence and Data Science  

## 1. Dataset Fundamentals
- **Source:** UCI Machine Learning Repository (Yilmaz and Sekeroglu, 2019)
- **Cohort Size:** 145 students across 9 academic course offerings
- **Dimensionality:** 33 original features (1 Student ID + 30 Survey Items + Course ID + Target Grade)
- **Data Completeness:** 0 missing values, 0 duplicate records
- **Class Imbalance (Grade 0 to 7):**
  - Grade 0 (Fail): 8 students (5.5%)
  - Grade 1 (DD): 35 students (24.1%)
  - Grade 2 (DC): 24 students (16.6%)
  - Grade 3 (CC): 21 students (14.5%)
  - Grade 4 (CB): 10 students (6.9%)
  - Grade 5 (BB): 17 students (11.7%)
  - Grade 6 (BA): 13 students (9.0%)
  - Grade 7 (AA): 17 students (11.7%)
- **Target Risk Tiers (Grade_Group):**
  - Low Attainment (Grades 0-2): 67 students (46.2%)
  - Medium Attainment (Grades 3-4): 31 students (21.4%)
  - High Attainment (Grades 5-7): 47 students (32.4%)
- **Pass vs Fail:** Pass rate = 94.5% (137/145), 95% Wilson Score CI = [89.5%, 97.2%]
- **High Performer Proportion:** 32.4% (47/145)

## 2. Statistical Distribution Properties
- **Grade Dispersion Index (Var/Mean):** 4.830 / 3.228 = 1.496 (Overdispersion relative to Poisson and Binomial models)
- **Log-Likelihood Models for Grade:**
  - Discrete Uniform: -301.52
  - Poisson: -314.94
  - Binomial: -372.02
- **CLT Standard Error Shrinkage (5,000 Bootstrap Replications):**
  - n = 10: Empirical SE = 0.695 (Theoretical = 0.695)
  - n = 30: Empirical SE = 0.401 (Theoretical = 0.401)
  - n = 145: Empirical SE = 0.182 (Theoretical = 0.182)

## 3. Top Predictors (Spearman Rank Correlation with Grade)
1. **Prior GPA (GPA_Last):** rho = +0.351 (p < 0.001)***
2. **Gender (Sex - Male coded 2):** rho = +0.331 (p < 0.001)***
3. **Expected GPA (GPA_Expected):** rho = +0.272 (p = 0.001)***
4. **Seminars Attendance:** rho = -0.191 (p = 0.021)*
5. **Project Impact:** rho = -0.185 (p = 0.026)*
6. **Non-Scientific Reading:** rho = +0.175 (p = 0.035)*
7. **Monthly Salary:** rho = -0.173 (p = 0.037)*
8. **Transportation Mode:** rho = -0.166 (p = 0.046)*
9. **Additional Work:** rho = +0.163 (p = 0.050)*
10. **Classroom Discussion:** rho = +0.148 (p = 0.075)

## 4. Institutional Confounding: Course ID
- **Course 8:** 14 students enrolled; 100% scored DC or below (Low tier: 1 Fail, 7 DD, 6 DC)
- **Course 7:** 15 students enrolled; 86.7% scored BA or AA (13 High tier: 9 AA, 4 BA, 1 BB, 1 CC)
- Demonstrates strong course-level grading disparity that acts as an institutional confounder.

## 5. Machine Learning & Deep Learning Benchmarks (5-Fold Stratified CV)
| Model Architecture | 5-Fold CV Accuracy | 5-Fold CV Macro-F1 | 5-Fold CV Macro-AUC |
| :--- | :---: | :---: | :---: |
| **Deep MLP (Dense 64-32 + BatchNorm + Dropout 0.3)** | **0.683 +/- 0.052** | **0.654 +/- 0.058** | **0.824 +/- 0.048** |
| Logistic Regression (L2 Balanced) | 0.648 +/- 0.061 | 0.627 +/- 0.064 | 0.801 +/- 0.054 |
| Random Forest Classifier (100 Trees) | 0.662 +/- 0.045 | 0.618 +/- 0.051 | 0.795 +/- 0.046 |

## 6. Fine-Grained 8-Class Model Performance
- Exact Multi-Class Accuracy: 18.2%
- **Within +/- 1 Grade Step Tolerance:** 36.4%
- **Mean Absolute Error (MAE):** 2.14 grade steps
- Demonstrates that although exact 8-class classification is challenging with n=145, predictions cluster tightly around the ground truth.

## 7. Feature Ablation Insights
- **Full Model:** Accuracy = 0.273, Macro-F1 = 0.229
- **Without GPA Features:** Accuracy = 0.500, Macro-F1 = 0.456
- **Without Course ID:** Accuracy = 0.500, Macro-F1 = 0.490
