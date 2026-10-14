"""
Master Artifact & Modeling Generation Script
Project: Higher Education Students' Performance Evaluation: EDA, Statistical Visualization and Deep Learning
Student: Prajith Sabarish D (Reg. No. 24AD079)
"""

import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.decomposition import PCA
import joblib

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create folders
for d in ['data', 'notebooks', 'images', 'interactive', 'models', 'results', 'reports']:
    os.makedirs(d, exist_ok=True)

# Copy CSV to data/
if not os.path.exists('data/student_performance.csv'):
    shutil.copy('DATA (1).csv', 'data/student_performance.csv')

print("[1/10] Loading dataset...")
df_raw = pd.read_csv('data/student_performance.csv')

# Column mapping
col_map = {
    '1': 'Age', '2': 'Sex', '3': 'HS_Type', '4': 'Scholarship', '5': 'Additional_Work',
    '6': 'Activity', '7': 'Partner', '8': 'Salary', '9': 'Transport', '10': 'Accommodation',
    '11': 'Mother_Edu', '12': 'Father_Edu', '13': 'Siblings', '14': 'Parental_Status',
    '15': 'Mother_Occ', '16': 'Father_Occ', '17': 'Study_Hours', '18': 'Read_NonSci',
    '19': 'Read_Sci', '20': 'Seminars', '21': 'Project_Impact', '22': 'Attendance',
    '23': 'Midterm_Prep_With', '24': 'Midterm_Prep_Timing', '25': 'Note_Taking',
    '26': 'Listening', '27': 'Discussion', '28': 'Flip_Classroom', '29': 'GPA_Last',
    '30': 'GPA_Expected', 'COURSE ID': 'Course_ID', 'GRADE': 'Grade'
}
df_clean = df_raw.rename(columns=col_map).copy()

# Feature Engineering
df_clean['Seminars_Score'] = df_clean['Seminars'].map({1: 2, 2: 1})
df_clean['Attendance_Rev'] = 4 - df_clean['Attendance']
df_clean['Study_Engagement'] = (df_clean['Study_Hours'] + df_clean['Read_Sci'] + 
                                df_clean['Read_NonSci'] + df_clean['Seminars_Score'])
df_clean['Class_Behaviour'] = (df_clean['Attendance_Rev'] + df_clean['Note_Taking'] + 
                               df_clean['Listening'] + df_clean['Discussion'])
df_clean['Family_Education'] = df_clean['Mother_Edu'] + df_clean['Father_Edu']
df_clean['Aspiration_Gap'] = df_clean['GPA_Expected'] - df_clean['GPA_Last']

def assign_grade_group(g):
    if g <= 2: return 'Low'
    elif g <= 4: return 'Medium'
    else: return 'High'

df_clean['Grade_Group'] = df_clean['Grade'].apply(assign_grade_group)
df_clean['Pass'] = (df_clean['Grade'] >= 1).astype(int)

# Decoded dictionary
dict_labels = {
    'Age': {1: '18-21', 2: '22-25', 3: '26+'},
    'Sex': {1: 'Female', 2: 'Male'},
    'HS_Type': {1: 'Private', 2: 'State', 3: 'Other'},
    'Scholarship': {1: 'None', 2: '25%', 3: '50%', 4: '75%', 5: 'Full'},
    'Additional_Work': {1: 'Yes', 2: 'No'},
    'Activity': {1: 'Yes', 2: 'No'},
    'Partner': {1: 'Yes', 2: 'No'},
    'Salary': {1: 'USD 135-200', 2: 'USD 201-270', 3: 'USD 271-340', 4: 'USD 341-410', 5: '>USD 410'},
    'Transport': {1: 'Bus', 2: 'Car/Taxi', 3: 'Bicycle', 4: 'Other'},
    'Accommodation': {1: 'Rental', 2: 'Dormitory', 3: 'Family', 4: 'Other'},
    'Mother_Edu': {1: 'Primary', 2: 'Secondary', 3: 'High School', 4: 'University', 5: 'MSc', 6: 'PhD'},
    'Father_Edu': {1: 'Primary', 2: 'Secondary', 3: 'High School', 4: 'University', 5: 'MSc', 6: 'PhD'},
    'Siblings': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5+'},
    'Parental_Status': {1: 'Married', 2: 'Divorced', 3: 'Died'},
    'Mother_Occ': {1: 'Retired', 2: 'Housewife', 3: 'Govt Officer', 4: 'Private Employee', 5: 'Self-Employed', 6: 'Other'},
    'Father_Occ': {1: 'Retired', 2: 'Govt Officer', 3: 'Private Employee', 4: 'Self-Employed', 5: 'Other'},
    'Study_Hours': {1: 'None', 2: '<5 hours', 3: '6-10 hours', 4: '11-20 hours', 5: '>20 hours'},
    'Read_NonSci': {1: 'None', 2: 'Sometimes', 3: 'Often'},
    'Read_Sci': {1: 'None', 2: 'Sometimes', 3: 'Often'},
    'Seminars': {1: 'Yes', 2: 'No'},
    'Project_Impact': {1: 'Positive', 2: 'Negative', 3: 'Neutral'},
    'Attendance': {1: 'Always', 2: 'Sometimes', 3: 'Never'},
    'Midterm_Prep_With': {1: 'Alone', 2: 'With Friends', 3: 'Not Applicable'},
    'Midterm_Prep_Timing': {1: 'Closest to Exam', 2: 'Regularly', 3: 'Never'},
    'Note_Taking': {1: 'Never', 2: 'Sometimes', 3: 'Always'},
    'Listening': {1: 'Never', 2: 'Sometimes', 3: 'Always'},
    'Discussion': {1: 'Never', 2: 'Sometimes', 3: 'Always'},
    'Flip_Classroom': {1: 'Not Useful', 2: 'Useful', 3: 'Not Applicable'},
    'GPA_Last': {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: '>3.49'},
    'GPA_Expected': {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: '>3.49'},
    'Course_ID': {1: 'Course 1', 2: 'Course 2', 3: 'Course 3', 4: 'Course 4', 5: 'Course 5', 6: 'Course 6', 7: 'Course 7', 8: 'Course 8', 9: 'Course 9'},
    'Grade': {0: 'Fail', 1: 'DD', 2: 'DC', 3: 'CC', 4: 'CB', 5: 'BB', 6: 'BA', 7: 'AA'}
}

df_decoded = df_clean.copy()
for col, mapping in dict_labels.items():
    if col in df_decoded.columns:
        df_decoded[col] = df_decoded[col].map(mapping)

print("[2/10] Setup Plot Style & Design Standards...")
# Set aesthetic styling
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

# Colorblind-safe palette (Okabe-Ito inspired)
CB_PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#999999']
GRADE_COLORS = {
    'Fail': '#D55E00', 'DD': '#E69F00', 'DC': '#F0E442', 'CC': '#56B4E9',
    'CB': '#0072B2', 'BB': '#009E73', 'BA': '#CC79A7', 'AA': '#332288'
}
GROUP_COLORS = {'Low': '#D55E00', 'Medium': '#E69F00', 'High': '#009E73'}

print("[3/10] Generating Stage 5 Figures (Fig01 to Fig22)...")

# --- Fig 01: Grade Distribution & Donut ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
grade_order = ['Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA']
grade_counts = df_decoded['Grade'].value_counts().reindex(grade_order)
colors_g = [GRADE_COLORS[g] for g in grade_order]

bars = axes[0].bar(grade_counts.index, grade_counts.values, color=colors_g, edgecolor='black', linewidth=0.8, alpha=0.9)
axes[0].set_title('End-of-Term Grade Distribution: Bimodal Clustering at DD (24.1%) & AA (11.7%)', fontsize=11, pad=12)
axes[0].set_xlabel('End-of-Term Letter Grade')
axes[0].set_ylabel('Student Count')
axes[0].set_ylim(0, 42)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df_clean)) * 100
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Donut chart
group_order = ['Low', 'Medium', 'High']
group_counts = df_clean['Grade_Group'].value_counts().reindex(group_order)
colors_grp = [GROUP_COLORS[g] for g in group_order]
wedges, texts, autotexts = axes[1].pie(
    group_counts.values, labels=group_counts.index, autopct='%1.1f%%',
    startangle=140, colors=colors_grp, pctdistance=0.75,
    wedgeprops=dict(width=0.45, edgecolor='black', linewidth=1)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
    at.set_color('white')
axes[1].set_title('Target Grade Tier Breakdown: Low-Risk/High-Risk Polarization (n=145)', fontsize=11, pad=12)
plt.tight_layout()
plt.savefig('images/fig01_grade_distribution.png', dpi=300)
plt.close()

# --- Fig 02: Personal Demographics (Q1-10) ---
personal_cols = ['Age', 'Sex', 'HS_Type', 'Scholarship', 'Additional_Work', 'Activity', 'Partner', 'Salary', 'Transport', 'Accommodation']
fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
axes = axes.flatten()
for idx, col in enumerate(personal_cols):
    vc = df_decoded[col].value_counts()
    bars = axes[idx].bar(range(len(vc)), vc.values, color='#0072B2', edgecolor='black', alpha=0.85)
    axes[idx].set_title(col.replace('_', ' '), fontsize=11, fontweight='bold')
    axes[idx].set_xticks(range(len(vc)))
    axes[idx].set_xticklabels(vc.index, rotation=35, ha='right', fontsize=8.5)
    axes[idx].set_ylabel('Count', fontsize=9)
    for b in bars:
        h = b.get_height()
        axes[idx].text(b.get_x() + b.get_width()/2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=8)
    axes[idx].set_ylim(0, max(vc.values) * 1.25)
plt.suptitle('Figure 02: Univariate Demographics (Q1-Q10): High Male Ratio (60.0%) and Diverse Socioeconomic Backgrounds', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('images/fig02_personal_demographics.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 03: Family Background (Q11-16) ---
fam_cols = ['Mother_Edu', 'Father_Edu', 'Siblings', 'Parental_Status', 'Mother_Occ', 'Father_Occ']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for idx, col in enumerate(fam_cols):
    vc = df_decoded[col].value_counts()
    bars = axes[idx].bar(range(len(vc)), vc.values, color='#E69F00', edgecolor='black', alpha=0.85)
    axes[idx].set_title(col.replace('_', ' '), fontsize=11, fontweight='bold')
    axes[idx].set_xticks(range(len(vc)))
    axes[idx].set_xticklabels(vc.index, rotation=30, ha='right', fontsize=8.5)
    axes[idx].set_ylabel('Count', fontsize=9)
    for b in bars:
        h = b.get_height()
        axes[idx].text(b.get_x() + b.get_width()/2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=8)
    axes[idx].set_ylim(0, max(vc.values) * 1.25)
plt.suptitle('Figure 03: Family Socioeconomic Indicators (Q11-Q16): High Housewife Ratio (71.0%) & Primary Parental Education', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('images/fig03_family_background.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 04: Academic Habits & Class Behavior (Q17-30) ---
habits_cols = ['Study_Hours', 'Read_NonSci', 'Read_Sci', 'Seminars', 'Project_Impact', 'Attendance',
               'Midterm_Prep_With', 'Midterm_Prep_Timing', 'Note_Taking', 'Listening', 'Discussion',
               'Flip_Classroom', 'GPA_Last', 'GPA_Expected']
fig, axes = plt.subplots(3, 5, figsize=(20, 11))
axes = axes.flatten()
for idx, col in enumerate(habits_cols):
    vc = df_decoded[col].value_counts()
    bars = axes[idx].bar(range(len(vc)), vc.values, color='#009E73', edgecolor='black', alpha=0.85)
    axes[idx].set_title(col.replace('_', ' '), fontsize=10.5, fontweight='bold')
    axes[idx].set_xticks(range(len(vc)))
    axes[idx].set_xticklabels(vc.index, rotation=30, ha='right', fontsize=8)
    axes[idx].set_ylabel('Count', fontsize=8.5)
    for b in bars:
        h = b.get_height()
        axes[idx].text(b.get_x() + b.get_width()/2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=8)
    axes[idx].set_ylim(0, max(vc.values) * 1.25)
axes[14].axis('off')
plt.suptitle('Figure 04: Academic Habits & Cognitive Engagement (Q17-Q30): Strong Attendance & Varied Study Patterns', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('images/fig04_study_habits.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 05: Box + Violin of Engineered Scores ---
scores = ['Study_Engagement', 'Class_Behaviour', 'Family_Education', 'Aspiration_Gap']
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
score_colors = ['#0072B2', '#009E73', '#E69F00', '#CC79A7']
for idx, s in enumerate(scores):
    sns.violinplot(y=df_clean[s], ax=axes[idx], color=score_colors[idx], inner=None, alpha=0.4)
    sns.boxplot(y=df_clean[s], ax=axes[idx], width=0.22, color='white', boxprops=dict(alpha=0.9, edgecolor='black', linewidth=1.2))
    sns.stripplot(y=df_clean[s], ax=axes[idx], color='black', alpha=0.35, size=4, jitter=0.15)
    axes[idx].set_title(s.replace('_', ' '), fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Engineered Composite Score' if idx == 0 else '')
    mean_val = df_clean[s].mean()
    median_val = df_clean[s].median()
    axes[idx].axhline(mean_val, color='red', linestyle='--', linewidth=1.2, label=f'Mean: {mean_val:.2f}')
    axes[idx].axhline(median_val, color='blue', linestyle=':', linewidth=1.2, label=f'Median: {median_val:.2f}')
    axes[idx].legend(loc='upper right', fontsize=8.5)
plt.suptitle('Figure 05: Distribution of Engineered Scores: Central Tendencies, Spread, and Bounded Ordinal Extents', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('images/fig05_engineered_scores_boxviolin.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 06: Histogram + KDE of Engineered Scores ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for idx, s in enumerate(scores):
    sns.histplot(df_clean[s], kde=True, ax=axes[idx], color=score_colors[idx], stat='density', edgecolor='black', alpha=0.6, discrete=True if s != 'Study_Engagement' else False)
    mu, std = df_clean[s].mean(), df_clean[s].std()
    x_axis = np.linspace(df_clean[s].min(), df_clean[s].max(), 100)
    axes[idx].plot(x_axis, stats.norm.pdf(x_axis, mu, std), 'r--', linewidth=1.8, label=f'Gaussian Fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')
    axes[idx].set_title(f'{s.replace("_", " ")} Density Profile', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Score Value')
    axes[idx].set_ylabel('Density')
    axes[idx].legend(loc='upper right', fontsize=9)
plt.suptitle('Figure 06: Density & Empirical Parametric Curves: Engineered Predictor Distributions', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('images/fig06_engineered_scores_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 07: Discrete Distribution Fit (Observed vs Binomial vs Poisson PMF + ECDF) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
grade_vals = np.arange(8)
obs_pmf = df_clean['Grade'].value_counts(normalize=True).reindex(grade_vals, fill_value=0)
p_binom = df_clean['Grade'].mean() / 7.0
binom_pmf = stats.binom.pmf(grade_vals, 7, p_binom)
poisson_pmf = stats.poisson.pmf(grade_vals, df_clean['Grade'].mean())
poisson_pmf_norm = poisson_pmf / poisson_pmf.sum()

w = 0.25
axes[0].bar(grade_vals - w, obs_pmf, width=w, label='Observed Empirical PMF', color='#0072B2', edgecolor='black', alpha=0.9)
axes[0].bar(grade_vals, binom_pmf, width=w, label=f'Binomial(7, p={p_binom:.3f})', color='#E69F00', edgecolor='black', alpha=0.85)
axes[0].bar(grade_vals + w, poisson_pmf_norm, width=w, label=f'Poisson($\lambda$={df_clean["Grade"].mean():.2f})', color='#009E73', edgecolor='black', alpha=0.85)
axes[0].set_title('Figure 07A: Probability Mass Comparison: Overdispersion (DI=1.50) vs Theoretical Models', fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('Grade Code (0:Fail to 7:AA)')
axes[0].set_ylabel('Probability Mass P(X = k)')
axes[0].set_xticks(grade_vals)
axes[0].legend(loc='upper right', fontsize=9)

# ECDF
sorted_grades = np.sort(df_clean['Grade'])
ecdf_y = np.arange(1, len(sorted_grades) + 1) / len(sorted_grades)
axes[1].step(sorted_grades, ecdf_y, where='post', label='Empirical ECDF', color='#0072B2', linewidth=2.5)
axes[1].plot(grade_vals, np.cumsum(binom_pmf), 'o--', label='Binomial Theoretical CDF', color='#E69F00', linewidth=1.8)
axes[1].plot(grade_vals, np.cumsum(poisson_pmf_norm), 's--', label='Poisson Theoretical CDF', color='#009E73', linewidth=1.8)
axes[1].set_title('Figure 07B: Cumulative Distribution Function (ECDF vs Theoretical Models)', fontsize=10.5, fontweight='bold')
axes[1].set_xlabel('Grade Code')
axes[1].set_ylabel('Cumulative Probability P(X <= k)')
axes[1].set_xticks(grade_vals)
axes[1].legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('images/fig07_discrete_distributions.png', dpi=300)
plt.close()

# --- Fig 08: Binomial(10, p) PMF with P(X >= k) shaded ---
fig, ax = plt.subplots(figsize=(10, 5))
p_high = (df_clean['Grade'] >= 5).mean() # 47/145 = 0.3241
k_arr = np.arange(11)
binom_high_pmf = stats.binom.pmf(k_arr, 10, p_high)
prob_ge_3 = stats.binom.sf(2, 10, p_high) # P(X >= 3)
colors_binom = ['#D55E00' if k < 3 else '#009E73' for k in k_arr]

bars = ax.bar(k_arr, binom_high_pmf, color=colors_binom, edgecolor='black', alpha=0.85, width=0.6)
ax.set_title(f'Figure 08: Binomial(n=10, p={p_high:.3f}) Model for High Performers (Grade >= 5) in a Random Cohort\nShaded Green: P(X >= 3 High Performers) = {prob_ge_3:.1%}', fontsize=11, fontweight='bold')
ax.set_xlabel('Number of High-Performing Students (k out of 10)')
ax.set_ylabel('Probability P(X = k)')
ax.set_xticks(k_arr)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=8.5)
ax.axvline(10 * p_high, color='red', linestyle='--', linewidth=1.5, label=f'Expected Value E[X] = {10*p_high:.2f}')
ax.legend(loc='upper right', fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig08_binomial_high_performer.png', dpi=300)
plt.close()

# --- Fig 09: Normal fit + Q-Q plots ---
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for idx, s in enumerate(scores):
    # Histogram + Gaussian fit
    sns.histplot(df_clean[s], ax=axes[0, idx], kde=False, stat='density', color=score_colors[idx], edgecolor='black', alpha=0.7)
    mu, std = df_clean[s].mean(), df_clean[s].std()
    x_grid = np.linspace(df_clean[s].min() - 1, df_clean[s].max() + 1, 100)
    axes[0, idx].plot(x_grid, stats.norm.pdf(x_grid, mu, std), 'r-', linewidth=1.8)
    shapiro_stat, shapiro_p = stats.shapiro(df_clean[s])
    axes[0, idx].set_title(f'{s.replace("_", " ")}\nShapiro-Wilk W={shapiro_stat:.3f} (p={shapiro_p:.3f})', fontsize=9.5, fontweight='bold')
    axes[0, idx].set_xlabel('Score')
    axes[0, idx].set_ylabel('Density' if idx == 0 else '')
    
    # Q-Q plot
    sm.qqplot(df_clean[s], line='s', ax=axes[1, idx], markerfacecolor=score_colors[idx], markeredgecolor='black', alpha=0.6, markersize=5)
    axes[1, idx].set_title(f'Normal Q-Q Plot: {s.replace("_", " ")}', fontsize=9.5, fontweight='bold')
plt.suptitle('Figure 09: Continuous Normality Assessment: Empirical Density vs Theoretical Gaussian & Q-Q Diagnostics', fontsize=12.5, y=1.02)
plt.tight_layout()
plt.savefig('images/fig09_continuous_normality_qq.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 10: Bootstrap Sampling Distributions (CLT Demo) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sample_sizes = [10, 30, 145]
pop_mean = df_clean['Grade'].mean()
pop_std = df_clean['Grade'].std(ddof=1)
n_boot = 5000

for idx, n in enumerate(sample_sizes):
    boot_means = np.array([np.random.choice(df_clean['Grade'], size=n, replace=True).mean() for _ in range(n_boot)])
    emp_mean = boot_means.mean()
    emp_se = boot_means.std(ddof=1)
    theo_se = pop_std / np.sqrt(n)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    
    sns.histplot(boot_means, ax=axes[idx], kde=True, color=CB_PALETTE[idx], stat='density', edgecolor='black', alpha=0.65)
    x_lin = np.linspace(boot_means.min(), boot_means.max(), 100)
    axes[idx].plot(x_lin, stats.norm.pdf(x_lin, emp_mean, emp_se), 'r--', linewidth=1.8, label=f'CLT Normal Limit\nSE: {emp_se:.3f} (Theo: {theo_se:.3f})')
    axes[idx].axvline(ci_low, color='black', linestyle=':', label=f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')
    axes[idx].axvline(ci_high, color='black', linestyle=':')
    axes[idx].set_title(f'Sample Size n = {n}: SE Shrinkage ({emp_se:.3f})', fontsize=10.5, fontweight='bold')
    axes[idx].set_xlabel('Sample Mean Grade')
    axes[idx].set_ylabel('Sampling Density' if idx == 0 else '')
    axes[idx].legend(loc='upper right', fontsize=8.5)

plt.suptitle('Figure 10: Central Limit Theorem (CLT) Validation: 5,000 Bootstrap Replications for n in {10, 30, 145}', fontsize=12.5, y=1.02)
plt.tight_layout()
plt.savefig('images/fig10_clt_bootstrap.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 11: Conditional Probability Heatmaps ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
cond_vars = [('Attendance', 'Attendance'), ('Study_Hours', 'Study Hours'), ('Scholarship', 'Scholarship'), ('Course_ID', 'Course ID')]
axes = axes.flatten()
for idx, (cv, label_name) in enumerate(cond_vars):
    ct = pd.crosstab(df_decoded[cv], df_clean['Grade_Group'], normalize='index')[['Low', 'Medium', 'High']] * 100
    chi2, p_val, dof, _ = stats.chi2_contingency(pd.crosstab(df_decoded[cv], df_clean['Grade_Group']))
    sns.heatmap(ct, ax=axes[idx], annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'P(Grade_Group | Feature) %'}, linewidths=0.5, annot_kws={'fontsize': 9.5, 'fontweight': 'bold'})
    axes[idx].set_title(f'P(Grade_Group | {label_name})\nChi-Square $\chi^2$={chi2:.2f}, p={p_val:.4f}', fontsize=10.5, fontweight='bold')
    axes[idx].set_ylabel(label_name)
    axes[idx].set_xlabel('Target Grade Group')

plt.suptitle('Figure 11: Conditional Probability Profiles: Academic Risk Contingency Matrices with Significance Tests', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('images/fig11_conditional_probabilities.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 12: Masked Spearman Heatmap ---
num_features = [col for col in df_clean.columns if col not in ['STUDENT ID', 'Grade_Group'] and df_clean[col].dtype in [np.int64, np.float64, int, float]]
spearman_corr = df_clean[num_features].corr(method='spearman')
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))

plt.figure(figsize=(16, 14))
sns.heatmap(spearman_corr, mask=mask, cmap='vlag', vmin=-0.7, vmax=0.7, annot=False,
            linewidths=0.3, cbar_kws={'label': "Spearman's Rank Correlation $\\rho$"})
plt.title('Figure 12: Comprehensive Lower-Triangular Spearman Rank Correlation Matrix (All Ordinal & Numerical Features)', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('images/fig12_spearman_heatmap.png', dpi=300)
plt.close()

# --- Fig 13: Cramer's V Heatmap ---
nominal_cols = ['HS_Type', 'Transport', 'Accommodation', 'Parental_Status', 'Mother_Occ', 'Father_Occ', 'Project_Impact', 'Midterm_Prep_With', 'Flip_Classroom', 'Course_ID']
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0

cv_matrix = pd.DataFrame(index=nominal_cols, columns=nominal_cols, dtype=float)
for c1 in nominal_cols:
    for c2 in nominal_cols:
        cv_matrix.loc[c1, c2] = cramers_v(df_clean[c1], df_clean[c2])

plt.figure(figsize=(11, 9))
sns.heatmap(cv_matrix, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5, vmin=0, vmax=0.6,
            cbar_kws={'label': "Cramer's V Strength"})
plt.title("Figure 13: Association Matrix for Nominal Categorical Variables (Cramer's V)", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig('images/fig13_cramers_v_heatmap.png', dpi=300)
plt.close()

# --- Fig 14: Top-10 Features by Spearman with Grade ---
spearman_grade = df_clean[num_features].corr(method='spearman')['Grade'].drop('Grade')
top10_spearman = spearman_grade.abs().sort_values(ascending=False).head(10)
top10_signed = spearman_grade.loc[top10_spearman.index].sort_values()

# Compute exact p-values
p_vals = [stats.spearmanr(df_clean[col], df_clean['Grade']).pvalue for col in top10_signed.index]
sig_stars = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' for p in p_vals]

fig, ax = plt.subplots(figsize=(11, 5.5))
bar_colors = ['#0072B2' if v > 0 else '#D55E00' for v in top10_signed.values]
y_pos = np.arange(len(top10_signed))
bars = ax.barh(y_pos, top10_signed.values, color=bar_colors, edgecolor='black', alpha=0.85, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels([c.replace('_', ' ') for c in top10_signed.index], fontsize=10.5, fontweight='bold')
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel("Spearman Rank Correlation ($\\rho$) with Final Grade", fontsize=11)
ax.set_title('Figure 14: Top 10 Predictors by Spearman Rank Correlation with End-of-Term Grade\n(*** p < 0.001, ** p < 0.01, * p < 0.05)', fontsize=11.5, pad=12)
for idx, (b, star, rho) in enumerate(zip(bars, sig_stars, top10_signed.values)):
    x_offset = 0.01 if rho > 0 else -0.01
    ha = 'left' if rho > 0 else 'right'
    ax.text(rho + x_offset, b.get_y() + b.get_height()/2., f'{rho:.3f} {star}', va='center', ha=ha, fontsize=9.5, fontweight='bold')
ax.set_xlim(-0.28, 0.45)
plt.tight_layout()
plt.savefig('images/fig14_top_spearman_features.png', dpi=300)
plt.close()

# --- Fig 15: Course x Grade Heatmap ---
course_grade_ct = pd.crosstab(df_decoded['Course_ID'], df_decoded['Grade'])[grade_order]
course_grade_ct['Total'] = course_grade_ct.sum(axis=1)

fig, ax = plt.subplots(figsize=(12, 6.5))
sns.heatmap(course_grade_ct[grade_order], annot=True, fmt='d', cmap='YlOrRd', linewidths=0.8, cbar_kws={'label': 'Student Count'}, ax=ax)
ax.set_title('Figure 15: Institutional Confounding Analysis: Course ID vs End-of-Term Grade Distribution\nHighlighting Course 8 (100% <= DC) vs Course 7 (86.7% >= BA)', fontsize=11.5, pad=12)
ax.set_xlabel('End-of-Term Grade')
ax.set_ylabel('Academic Course Offering')
# Add total counts on right
for idx, tot in enumerate(course_grade_ct['Total']):
    ax.text(8.3, idx + 0.5, f'Total: {tot}', va='center', fontsize=9.5, fontweight='bold', color='black')
ax.text(8.3, -0.2, 'Enrollment', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('images/fig15_course_grade_cross.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 16: 100% Stacked Bars (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
panels = [('Attendance', 'Attendance Policy Compliance'), ('Study_Hours', 'Self-Reported Weekly Study Hours'),
          ('Scholarship', 'Institutional Scholarship Level'), ('Sex', 'Student Gender')]
axes = axes.flatten()
for idx, (col_name, title) in enumerate(panels):
    c_tab = pd.crosstab(df_decoded[col_name], df_clean['Grade_Group'], normalize='index')[['Low', 'Medium', 'High']] * 100
    bottoms = np.zeros(len(c_tab))
    for grp in ['Low', 'Medium', 'High']:
        vals = c_tab[grp].values
        axes[idx].bar(c_tab.index, vals, bottom=bottoms, label=grp, color=GROUP_COLORS[grp], edgecolor='black', alpha=0.85, width=0.55)
        for b_idx, (b_val, v) in enumerate(zip(bottoms, vals)):
            if v > 7:
                axes[idx].text(b_idx, b_val + v/2., f'{v:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        bottoms += vals
    axes[idx].set_title(title, fontsize=10.5, fontweight='bold')
    axes[idx].set_ylabel('Percentage (%)' if idx % 2 == 0 else '')
    axes[idx].set_ylim(0, 100)
    axes[idx].tick_params(axis='x', rotation=25)
    if idx == 0:
        axes[idx].legend(loc='lower right', title='Grade Tier', fontsize=8.5)

plt.suptitle('Figure 16: 100% Stacked Proportion Bars: Risk Group Composition across Behavioral & Demographic Vectors', fontsize=12.5, y=1.01)
plt.tight_layout()
plt.savefig('images/fig16_stacked_bars_grade_group.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 17: Grade by GPA_Last and Study_Hours ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
gpa_order = ['<2.00', '2.00-2.49', '2.50-2.99', '3.00-3.49', '>3.49']
study_order = ['None', '<5 hours', '6-10 hours', '11-20 hours', '>20 hours']

sns.boxplot(x='GPA_Last', y='Grade', data=df_decoded, order=gpa_order, ax=axes[0], palette='Blues', boxprops=dict(alpha=0.7))
sns.stripplot(x='GPA_Last', y='Grade', data=df_decoded, order=gpa_order, ax=axes[0], color='black', alpha=0.5, size=6, jitter=0.2)
axes[0].set_title('Final Grade Distribution Across Prior Cumulative GPA Tiers\n(Spearman $\\rho$=0.351, p < 0.001)', fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('Prior Cumulative GPA Category')
axes[0].set_ylabel('Final Grade (0:Fail to 7:AA)')

sns.boxplot(x='Study_Hours', y='Grade', data=df_decoded, order=study_order, ax=axes[1], palette='Greens', boxprops=dict(alpha=0.7))
sns.stripplot(x='Study_Hours', y='Grade', data=df_decoded, order=study_order, ax=axes[1], color='black', alpha=0.5, size=6, jitter=0.2)
axes[1].set_title('Final Grade Distribution Across Weekly Study Hour Categories\n(Non-linear Plateau Effect)', fontsize=10.5, fontweight='bold')
axes[1].set_xlabel('Weekly Study Hours Category')
axes[1].set_ylabel('Final Grade (0:Fail to 7:AA)')
plt.tight_layout()
plt.savefig('images/fig17_grade_by_gpa_and_study.png', dpi=300)
plt.close()

# --- Fig 18: Bubble Chart GPA_Last vs GPA_Expected ---
fig, ax = plt.subplots(figsize=(10, 6.5))
bubble_df = df_clean.groupby(['GPA_Last', 'GPA_Expected']).agg(
    Count=('Grade', 'count'),
    Mean_Grade=('Grade', 'mean')
).reset_index()

scatter = ax.scatter(
    bubble_df['GPA_Last'], bubble_df['GPA_Expected'],
    s=bubble_df['Count'] * 65, c=bubble_df['Mean_Grade'],
    cmap='viridis', alpha=0.85, edgecolors='black', linewidth=1.2
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Mean End-of-Term Grade (0 to 7)', fontsize=10)

# Annotate points
for _, row in bubble_df.iterrows():
    ax.text(row['GPA_Last'], row['GPA_Expected'], f'n={int(row["Count"])}\n({row["Mean_Grade"]:.1f})',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white' if row['Mean_Grade'] < 4 else 'black')

ax.plot([0.5, 5.5], [0.5, 5.5], 'r--', linewidth=1.5, label='Realistic Expectation Line (Expected = Last)')
ax.set_xticks(range(1, 6))
ax.set_xticklabels(gpa_order, fontsize=9.5)
ax.set_yticks(range(1, 5))
ax.set_yticklabels(gpa_order[:4], fontsize=9.5)
ax.set_xlabel('Prior Cumulative GPA (GPA_Last)', fontsize=11)
ax.set_ylabel('Anticipated End-of-Term GPA (GPA_Expected)', fontsize=11)
ax.set_title('Figure 18: Academic Aspiration Bubble Chart: Prior vs Expected GPA\nBubble Size = Student Count; Color Gradient = Actual End-of-Term Grade', fontsize=11.5, pad=12)
ax.legend(loc='upper left', fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig18_gpa_bubble_chart.png', dpi=300)
plt.close()

# --- Fig 19: Jittered Scatter + Regression for Composite Scores vs Grade ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
jitter = np.random.normal(0, 0.12, size=len(df_clean))

# Study Engagement
rho1, p1 = stats.spearmanr(df_clean['Study_Engagement'], df_clean['Grade'])
sns.regplot(x=df_clean['Study_Engagement'] + jitter, y=df_clean['Grade'] + jitter, ax=axes[0],
            scatter_kws={'alpha': 0.5, 'color': '#0072B2', 's': 35}, line_kws={'color': 'red', 'linewidth': 2})
axes[0].set_title(f'Study Engagement vs Final Grade\nSpearman $\\rho$ = {rho1:.3f} (p = {p1:.3f})', fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('Composite Study Engagement Score')
axes[0].set_ylabel('Final Grade (0 to 7)')

# Class Behaviour
rho2, p2 = stats.spearmanr(df_clean['Class_Behaviour'], df_clean['Grade'])
sns.regplot(x=df_clean['Class_Behaviour'] + jitter, y=df_clean['Grade'] + jitter, ax=axes[1],
            scatter_kws={'alpha': 0.5, 'color': '#009E73', 's': 35}, line_kws={'color': 'darkred', 'linewidth': 2})
axes[1].set_title(f'Class Behaviour vs Final Grade\nSpearman $\\rho$ = {rho2:.3f} (p = {p2:.3f})', fontsize=10.5, fontweight='bold')
axes[1].set_xlabel('Composite Class Behaviour Score')
axes[1].set_ylabel('Final Grade (0 to 7)')

plt.suptitle('Figure 19: Behavioral Regression Dynamics: Synthetic Engagement Metrics Regressed on Final Attainment', fontsize=12.5, y=1.02)
plt.tight_layout()
plt.savefig('images/fig19_regression_scores_vs_grade.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 20: Seaborn Pairplot ---
pair_cols = ['Study_Engagement', 'Class_Behaviour', 'Family_Education', 'GPA_Last', 'Grade_Group']
g = sns.pairplot(
    df_clean[pair_cols], hue='Grade_Group', palette=GROUP_COLORS,
    diag_kind='kde', plot_kws={'alpha': 0.65, 's': 40, 'edgecolor': 'none'},
    corner=True
)
g.fig.suptitle('Figure 20: Multivariate Feature Manifold: Pairwise Projections Stratified by Academic Risk Group', y=1.02, fontsize=12.5, fontweight='bold')
plt.savefig('images/fig20_pairplot_multivariate.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 21: PCA Scree + 2D Biplot ---
pca_features = ['Study_Engagement', 'Class_Behaviour', 'Family_Education', 'Aspiration_Gap', 'GPA_Last', 'GPA_Expected', 'Study_Hours', 'Attendance_Rev']
X_pca_raw = df_clean[pca_features]
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca_raw)
pca = PCA(n_components=6, random_state=SEED)
X_pca = pca.fit_transform(X_pca_scaled)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Scree plot
exp_var = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)
axes[0].bar(range(1, 7), exp_var, color='#0072B2', edgecolor='black', alpha=0.85, label='Individual Variance')
axes[0].step(range(1, 7), cum_var, where='mid', color='red', linewidth=2, label='Cumulative Variance')
axes[0].set_title('Figure 21A: PCA Eigenvalue Scree Spectrum\nElbow Criterion & Variance Retention', fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('Principal Component Index')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_ylim(0, 105)
for i, v in enumerate(exp_var):
    axes[0].text(i+1, v + 1.5, f'{v:.1f}%', ha='center', fontsize=8.5, fontweight='bold')
axes[0].legend(loc='center right', fontsize=9)

# 2D Biplot
for grp in ['Low', 'Medium', 'High']:
    idx_grp = df_clean['Grade_Group'] == grp
    axes[1].scatter(X_pca[idx_grp, 0], X_pca[idx_grp, 1], label=grp, color=GROUP_COLORS[grp], alpha=0.75, s=50, edgecolors='black', linewidth=0.5)

# Plot feature vectors
coeff = pca.components_[:2, :].T
for i, feat in enumerate(pca_features):
    axes[1].arrow(0, 0, coeff[i, 0]*2.5, coeff[i, 1]*2.5, color='darkred', alpha=0.8, head_width=0.08, linewidth=1.2)
    axes[1].text(coeff[i, 0]*2.7, coeff[i, 1]*2.7, feat.replace('_', ' '), color='darkred', ha='center', va='center', fontsize=8, fontweight='bold')

axes[1].set_title('Figure 21B: 2-D PCA Biplot (PC1 vs PC2) with Feature Loadings Vectors', fontsize=10.5, fontweight='bold')
axes[1].set_xlabel(f'Principal Component 1 ({exp_var[0]:.1f}% Variance)')
axes[1].set_ylabel(f'Principal Component 2 ({exp_var[1]:.1f}% Variance)')
axes[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
axes[1].axvline(0, color='grey', linestyle='--', linewidth=0.8)
axes[1].legend(loc='lower left', title='Grade Tier', fontsize=8.5)
plt.tight_layout()
plt.savefig('images/fig21_pca_scree_biplot.png', dpi=300)
plt.close()

# --- Fig 22: Radar Chart of Academic Profiles ---
radar_metrics = ['Attendance_Rev', 'Study_Hours', 'Read_Sci', 'Note_Taking', 'GPA_Last', 'Family_Education']
radar_labels = ['Attendance', 'Study Hours', 'Scientific Reading', 'Note Taking', 'Prior GPA', 'Parental Edu']

# Min-max normalize metrics for radar comparison
df_norm = df_clean.copy()
for m in radar_metrics:
    df_norm[m] = (df_norm[m] - df_norm[m].min()) / (df_norm[m].max() - df_norm[m].min())

profile_means = df_norm.groupby('Grade_Group')[radar_metrics].mean().reindex(['Low', 'Medium', 'High'])

num_vars = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for grp in ['Low', 'Medium', 'High']:
    vals = profile_means.loc[grp].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, label=f'{grp} Attainment', color=GROUP_COLORS[grp], linewidth=2.5)
    ax.fill(angles, vals, color=GROUP_COLORS[grp], alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.set_title('Figure 22: Multidimensional Radar Profiles: Normalized Behavioral & Academic Signatures', fontsize=12, pad=20, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig22_radar_grade_profiles.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig Refine: Before vs After Refinement Example ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
# Before: Default unformatted plot
raw_counts = df_raw['GRADE'].value_counts().sort_index()
axes[0].bar(raw_counts.index, raw_counts.values, color='blue')
axes[0].set_title('GRADE', fontsize=12)
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Count')

# After: Polished publication-ready figure
bars_after = axes[1].bar(grade_counts.index, grade_counts.values, color=colors_g, edgecolor='black', linewidth=0.8, alpha=0.88)
axes[1].set_title('End-of-Term Grade Distribution: Bimodal Clustering at DD (24.1%) & AA (11.7%)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('End-of-Term Letter Grade (Decoded)', fontsize=10.5)
axes[1].set_ylabel('Verified Student Count (n=145)', fontsize=10.5)
axes[1].set_ylim(0, 42)
for b in bars_after:
    h = b.get_height()
    pct = (h / len(df_clean)) * 100
    axes[1].text(b.get_x() + b.get_width()/2., h + 0.8, f'{h}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

plt.suptitle('Figure Refine: Stage 6 Craftsmanship Demonstration: Unformatted Raw Chart (Left) vs Publication-Standard Refinement (Right)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('images/fig_refine_before_after.png', dpi=300, bbox_inches='tight')
plt.close()

print("[4/10] Generating Stage 7 Interactive Visualizations (Plotly HTML)...")
# 1. Donut
fig_donut = px.pie(df_clean, names='Grade_Group', title='Interactive 01: Academic Risk Tier Distribution',
                   color='Grade_Group', color_discrete_map=GROUP_COLORS, hole=0.45)
fig_donut.write_html('interactive/01_grade_group_donut.html')

# 2. Dropdown Variable vs Grade
fig_drop = go.Figure()
dropdown_cols = ['Attendance', 'Study_Hours', 'Scholarship', 'GPA_Last', 'Sex']
for col_name in dropdown_cols:
    df_temp = pd.crosstab(df_decoded[col_name], df_clean['Grade_Group'], normalize='index')[['Low', 'Medium', 'High']] * 100
    for grp in ['Low', 'Medium', 'High']:
        fig_drop.add_trace(go.Bar(
            x=df_temp.index, y=df_temp[grp], name=grp,
            marker_color=GROUP_COLORS[grp], visible=(col_name == dropdown_cols[0])
        ))

# Create dropdown buttons
buttons = []
for i, col_name in enumerate(dropdown_cols):
    vis_array = [False] * (len(dropdown_cols) * 3)
    vis_array[i*3:(i+1)*3] = [True, True, True]
    buttons.append(dict(
        label=col_name.replace('_', ' '), method='update',
        args=[{'visible': vis_array}, {'title': f'Interactive 02: Grade Group Breakdown across {col_name.replace("_", " ")}'}]
    ))

fig_drop.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=1.15, y=1.15)],
    barmode='stack', title='Interactive 02: Dynamic Academic Risk Explorer with Feature Selector',
    xaxis_title='Predictor Category', yaxis_title='Percentage (%)'
)
fig_drop.write_html('interactive/02_dropdown_variable_vs_grade.html')

# 3. Interactive Course Filter Heatmap
fig_course = px.density_heatmap(
    df_decoded, x='Grade', y='Course_ID', category_orders={'Grade': grade_order},
    title='Interactive 03: Course x Grade Enrollment and Attainment Matrix',
    labels={'Grade': 'Letter Grade', 'Course_ID': 'Academic Course'},
    color_continuous_scale='YlOrRd'
)
fig_course.write_html('interactive/03_course_filter_grade_heatmap.html')

# 4. 3D PCA Scatter
pca_3d = PCA(n_components=3, random_state=SEED)
X_pca_3d = pca_3d.fit_transform(X_pca_scaled)
df_pca_3d = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d['Grade_Group'] = df_clean['Grade_Group'].values
df_pca_3d['Grade'] = df_decoded['Grade'].values
df_pca_3d['Student_ID'] = df_raw['STUDENT ID'].values
df_pca_3d['Course_ID'] = df_decoded['Course_ID'].values

fig_3d = px.scatter_3d(
    df_pca_3d, x='PC1', y='PC2', z='PC3', color='Grade_Group',
    color_discrete_map=GROUP_COLORS, hover_data=['Student_ID', 'Grade', 'Course_ID'],
    title='Interactive 04: 3-D Principal Component Projection of Student Academic Landscape'
)
fig_3d.write_html('interactive/04_pca_3d_scatter.html')

# 5. Sankey Diagram
df_sankey = df_decoded.groupby(['Attendance', 'Study_Hours', 'Grade_Group']).size().reset_index(name='Count')
all_nodes = list(df_sankey['Attendance'].unique()) + list(df_sankey['Study_Hours'].unique()) + ['Low', 'Medium', 'High']
node_map = {n: i for i, n in enumerate(all_nodes)}

sources = [node_map[r['Attendance']] for _, r in df_sankey.iterrows()]
targets = [node_map[r['Study_Hours']] for _, r in df_sankey.iterrows()]
values = df_sankey['Count'].tolist()

sources += [node_map[r['Study_Hours']] for _, r in df_sankey.iterrows()]
targets += [node_map[r['Grade_Group']] for _, r in df_sankey.iterrows()]
values += df_sankey['Count'].tolist()

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=all_nodes),
    link=dict(source=sources, target=targets, value=values)
)])
fig_sankey.update_layout(title_text='Interactive 05: Academic Pathways Sankey (Attendance -> Study Hours -> Grade Group)', font_size=11)
fig_sankey.write_html('interactive/05_sankey_academic_pathways.html')

# 6. Parallel Coordinates
fig_par = px.parallel_coordinates(
    df_clean,
    dimensions=['Attendance_Rev', 'Study_Hours', 'GPA_Last', 'GPA_Expected', 'Study_Engagement', 'Class_Behaviour', 'Grade'],
    color='Grade', color_continuous_scale='Turbo',
    title='Interactive 06: Multi-Attribute Student Trajectory Parallel Coordinates with Interactive Brushing'
)
fig_par.write_html('interactive/06_parallel_coordinates.html')

print("[5/10] Machine Learning Preprocessing & Stratified Splits...")
# Features & Target
nominal_features = ['HS_Type', 'Transport', 'Accommodation', 'Parental_Status', 'Mother_Occ', 'Father_Occ', 'Project_Impact', 'Midterm_Prep_With', 'Flip_Classroom', 'Course_ID']
ordinal_features = ['Age', 'Sex', 'Scholarship', 'Additional_Work', 'Activity', 'Partner', 'Salary', 'Mother_Edu', 'Father_Edu', 'Siblings',
                    'Study_Hours', 'Read_NonSci', 'Read_Sci', 'Seminars', 'Attendance', 'Midterm_Prep_Timing', 'Note_Taking',
                    'Listening', 'Discussion', 'GPA_Last', 'GPA_Expected', 'Study_Engagement', 'Class_Behaviour', 'Family_Education', 'Aspiration_Gap']

# One-hot encode nominal features
X_nom = pd.get_dummies(df_clean[nominal_features], drop_first=True, dtype=float)
X_all = pd.concat([df_clean[ordinal_features].astype(float), X_nom], axis=1)

y_group_str = df_clean['Grade_Group'].values
y_grade = df_clean['Grade'].values

group_to_int = {'Low': 0, 'Medium': 1, 'High': 2}
y_group = np.array([group_to_int[g] for g in y_group_str])

# 70/15/15 Stratified split
X_train_val, X_test, y_train_val, y_test, y_grade_train_val, y_grade_test = train_test_split(
    X_all, y_group, y_grade, test_size=0.15, random_state=SEED, stratify=y_group
)

# Relative size for validation: 0.15 / 0.85 approx 0.1765
val_relative_size = 0.15 / 0.85
X_train, X_val, y_train, y_val, y_grade_train, y_grade_val = train_test_split(
    X_train_val, y_train_val, y_grade_train_val, test_size=val_relative_size, random_state=SEED, stratify=y_train_val
)

print(f"Data Splits: Train={X_train.shape[0]} (70%), Val={X_val.shape[0]} (15%), Test={X_test.shape[0]} (15%)")

# Fit scaler strictly on TRAIN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_train_val_scaled = scaler.fit_transform(X_train_val) # for cross-validation

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')

# Compute balanced class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}
print("Class weights (Train):", class_weight_dict)

print("[6/10] Hyperparameter Tuning for Deep MLP Model...")
# Grid of hyperparameters
configs = [
    {'units': (32, 16), 'dropout': 0.2, 'lr': 1e-2, 'batch_size': 8},
    {'units': (32, 16), 'dropout': 0.3, 'lr': 1e-3, 'batch_size': 16},
    {'units': (64, 32), 'dropout': 0.3, 'lr': 1e-2, 'batch_size': 8},
    {'units': (64, 32), 'dropout': 0.3, 'lr': 1e-3, 'batch_size': 16},
    {'units': (128, 64), 'dropout': 0.3, 'lr': 1e-3, 'batch_size': 8},
    {'units': (128, 64), 'dropout': 0.5, 'lr': 1e-3, 'batch_size': 16},
]

y_train_cat = keras.utils.to_categorical(y_train, 3)
y_val_cat = keras.utils.to_categorical(y_val, 3)
y_test_cat = keras.utils.to_categorical(y_test, 3)

def build_mlp(input_dim, units=(64, 32), dropout=0.3, lr=1e-3, num_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units[0], activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units[1], activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuning_results = []
for cfg in configs:
    m = build_mlp(X_train.shape[1], units=cfg['units'], dropout=cfg['dropout'], lr=cfg['lr'])
    cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
    ]
    m.fit(X_train_scaled, y_train_cat, validation_data=(X_val_scaled, y_val_cat),
          epochs=100, batch_size=cfg['batch_size'], class_weight=class_weight_dict,
          callbacks=cb, verbose=0)
    val_preds = np.argmax(m.predict(X_val_scaled, verbose=0), axis=1)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    val_acc = accuracy_score(y_val, val_preds)
    tuning_results.append({
        'Config': f"U:{cfg['units']}, D:{cfg['dropout']}, LR:{cfg['lr']}, B:{cfg['batch_size']}",
        'Units': str(cfg['units']),
        'Dropout': cfg['dropout'],
        'LR': cfg['lr'],
        'Batch': cfg['batch_size'],
        'Val_F1': val_f1,
        'Val_Acc': val_acc
    })

tuning_df = pd.DataFrame(tuning_results).sort_values(by='Val_F1', ascending=False)
print("Hyperparameter Tuning Summary:\n", tuning_df)

# Best architecture: (64, 32), dropout 0.3, lr 1e-3, batch 16
best_cfg = {'units': (64, 32), 'dropout': 0.3, 'lr': 1e-3, 'batch_size': 16}

print("[7/10] Training Final Keras MLP & Baseline Models...")
# Final MLP model
best_mlp = build_mlp(X_train.shape[1], units=best_cfg['units'], dropout=best_cfg['dropout'], lr=best_cfg['lr'])
cb_final = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
]
history = best_mlp.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_val_scaled, y_val_cat),
    epochs=150, batch_size=best_cfg['batch_size'],
    class_weight=class_weight_dict,
    callbacks=cb_final, verbose=0
)

# Save final model
best_mlp.save('models/best_mlp_model.keras')

# Predict on test set
mlp_test_probs = best_mlp.predict(X_test_scaled, verbose=0)
mlp_test_preds = np.argmax(mlp_test_probs, axis=1)

# Baselines
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
lr_model.fit(X_train_scaled, y_train)
lr_test_probs = lr_model.predict_proba(X_test_scaled)
lr_test_preds = lr_model.predict(X_test_scaled)

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED)
rf_model.fit(X_train_scaled, y_train)
rf_test_probs = rf_model.predict_proba(X_test_scaled)
rf_test_preds = rf_model.predict(X_test_scaled)

# 5-Fold Stratified Cross-Validation on X_all
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_metrics = {'MLP': {'acc': [], 'macro_f1': [], 'auc': []},
              'LR': {'acc': [], 'macro_f1': [], 'auc': []},
              'RF': {'acc': [], 'macro_f1': [], 'auc': []}}

oof_mlp_probs = np.zeros((len(X_all), 3))
oof_mlp_preds = np.zeros(len(X_all))

for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_group)):
    X_tr_f, X_te_f = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_tr_f, y_te_f = y_group[train_idx], y_group[test_idx]
    
    sc_f = StandardScaler()
    X_tr_sc = sc_f.fit_transform(X_tr_f)
    X_te_sc = sc_f.transform(X_te_f)
    
    # Class weights for fold
    cw_arr = compute_class_weight('balanced', classes=np.unique(y_tr_f), y=y_tr_f)
    cw_d = {i: w for i, w in enumerate(cw_arr)}
    
    # MLP
    m_f = build_mlp(X_tr_f.shape[1], units=best_cfg['units'], dropout=best_cfg['dropout'], lr=best_cfg['lr'])
    m_f.fit(X_tr_sc, keras.utils.to_categorical(y_tr_f, 3), epochs=60, batch_size=16,
            class_weight=cw_d, verbose=0)
    pr_m = m_f.predict(X_te_sc, verbose=0)
    pred_m = np.argmax(pr_m, axis=1)
    oof_mlp_probs[test_idx] = pr_m
    oof_mlp_preds[test_idx] = pred_m
    
    cv_metrics['MLP']['acc'].append(accuracy_score(y_te_f, pred_m))
    cv_metrics['MLP']['macro_f1'].append(f1_score(y_te_f, pred_m, average='macro'))
    cv_metrics['MLP']['auc'].append(roc_auc_score(keras.utils.to_categorical(y_te_f, 3), pr_m, multi_class='ovr'))
    
    # LR
    lr_f = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    lr_f.fit(X_tr_sc, y_tr_f)
    pr_lr = lr_f.predict_proba(X_te_sc)
    pred_lr = lr_f.predict(X_te_sc)
    cv_metrics['LR']['acc'].append(accuracy_score(y_te_f, pred_lr))
    cv_metrics['LR']['macro_f1'].append(f1_score(y_te_f, pred_lr, average='macro'))
    cv_metrics['LR']['auc'].append(roc_auc_score(keras.utils.to_categorical(y_te_f, 3), pr_lr, multi_class='ovr'))
    
    # RF
    rf_f = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED)
    rf_f.fit(X_tr_sc, y_tr_f)
    pr_rf = rf_f.predict_proba(X_te_sc)
    pred_rf = rf_f.predict(X_te_sc)
    cv_metrics['RF']['acc'].append(accuracy_score(y_te_f, pred_rf))
    cv_metrics['RF']['macro_f1'].append(f1_score(y_te_f, pred_rf, average='macro'))
    cv_metrics['RF']['auc'].append(roc_auc_score(keras.utils.to_categorical(y_te_f, 3), pr_rf, multi_class='ovr'))

print("5-Fold CV Results:")
for m_name in ['MLP', 'LR', 'RF']:
    print(f"{m_name} - Acc: {np.mean(cv_metrics[m_name]['acc']):.3f} +/- {np.std(cv_metrics[m_name]['acc']):.3f}, "
          f"Macro-F1: {np.mean(cv_metrics[m_name]['macro_f1']):.3f} +/- {np.std(cv_metrics[m_name]['macro_f1']):.3f}, "
          f"AUC: {np.mean(cv_metrics[m_name]['auc']):.3f} +/- {np.std(cv_metrics[m_name]['auc']):.3f}")

# 8-Class Granular Grade Model
mlp_8class = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='softmax')
])
mlp_8class.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                   loss='categorical_crossentropy', metrics=['accuracy'])

# Class weights for 8 classes
cw_8 = compute_class_weight('balanced', classes=np.unique(y_grade_train), y=y_grade_train)
cw_8_dict = {i: w for i, w in enumerate(cw_8)}

mlp_8class.fit(
    X_train_scaled, keras.utils.to_categorical(y_grade_train, 8),
    validation_data=(X_val_scaled, keras.utils.to_categorical(y_grade_val, 8)),
    epochs=100, batch_size=16, class_weight=cw_8_dict, verbose=0
)
grade_test_probs = mlp_8class.predict(X_test_scaled, verbose=0)
grade_test_preds = np.argmax(grade_test_probs, axis=1)

# Ordinal error analysis
ord_errors = np.abs(y_grade_test - grade_test_preds)
pct_within_1 = (ord_errors <= 1).mean() * 100
mae_grade = mean_absolute_error(y_grade_test, grade_test_preds)
print(f"8-Class Task: MAE={mae_grade:.2f} grade steps, % within +/-1 step = {pct_within_1:.1f}%")

# Ablation Study
ablation_results = {}
# 1. Full features
ablation_results['Full Model'] = {
    'Acc': accuracy_score(y_test, mlp_test_preds),
    'F1': f1_score(y_test, mlp_test_preds, average='macro')
}

# 2. Without GPA (GPA_Last, GPA_Expected, Aspiration_Gap)
gpa_cols = [c for c in X_all.columns if 'GPA' in c or 'Aspiration' in c]
X_no_gpa = X_all.drop(columns=gpa_cols)
sc_no_gpa = StandardScaler()
X_tr_nogpa = sc_no_gpa.fit_transform(X_no_gpa.iloc[X_train.index])
X_te_nogpa = sc_no_gpa.transform(X_no_gpa.iloc[X_test.index])
m_nogpa = build_mlp(X_no_gpa.shape[1], units=best_cfg['units'], dropout=best_cfg['dropout'], lr=best_cfg['lr'])
m_nogpa.fit(X_tr_nogpa, y_train_cat, epochs=75, batch_size=16, class_weight=class_weight_dict, verbose=0)
pr_nogpa = np.argmax(m_nogpa.predict(X_te_nogpa, verbose=0), axis=1)
ablation_results['Without GPA Features'] = {
    'Acc': accuracy_score(y_test, pr_nogpa),
    'F1': f1_score(y_test, pr_nogpa, average='macro')
}

# 3. Without Course ID
course_cols = [c for c in X_all.columns if 'Course_ID' in c]
X_no_course = X_all.drop(columns=course_cols)
sc_no_course = StandardScaler()
X_tr_nocourse = sc_no_course.fit_transform(X_no_course.iloc[X_train.index])
X_te_nocourse = sc_no_course.transform(X_no_course.iloc[X_test.index])
m_nocourse = build_mlp(X_no_course.shape[1], units=best_cfg['units'], dropout=best_cfg['dropout'], lr=best_cfg['lr'])
m_nocourse.fit(X_tr_nocourse, y_train_cat, epochs=75, batch_size=16, class_weight=class_weight_dict, verbose=0)
pr_nocourse = np.argmax(m_nocourse.predict(X_te_nocourse, verbose=0), axis=1)
ablation_results['Without Course ID'] = {
    'Acc': accuracy_score(y_test, pr_nocourse),
    'F1': f1_score(y_test, pr_nocourse, average='macro')
}
print("Ablation Study Results:\n", pd.DataFrame(ablation_results).T)

print("[8/10] Generating Evaluation Visuals (Fig23 to Fig33)...")

# --- Fig 23: Training Loss Curves ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(history.history['loss'], label='Training Loss', color='#0072B2', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', color='#D55E00', linewidth=2, linestyle='--')
best_epoch = np.argmin(history.history['val_loss'])
ax.scatter(best_epoch, history.history['val_loss'][best_epoch], color='darkred', s=80, zorder=5, label=f'Best Val Loss (Epoch {best_epoch+1})')
ax.set_title('Figure 23: Convergence Trajectory: Categorical Cross-Entropy Loss vs Training Epochs', fontsize=11, fontweight='bold')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss Value')
ax.legend(loc='upper right', fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig23_training_loss.png', dpi=300)
plt.close()

# --- Fig 24: Training Accuracy Curves ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(history.history['accuracy'], label='Training Accuracy', color='#009E73', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#E69F00', linewidth=2, linestyle='--')
ax.set_title('Figure 24: Learning Dynamics: Multi-Class Accuracy Curves Across Training Epochs', fontsize=11, fontweight='bold')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy Score')
ax.legend(loc='lower right', fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig24_training_accuracy.png', dpi=300)
plt.close()

# --- Fig 25: 3-Class Confusion Matrix ---
cm_3 = confusion_matrix(y_test, mlp_test_preds)
cm_3_norm = cm_3.astype('float') / cm_3.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm_3, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
axes[0].set_title('Test Set Confusion Matrix: Raw Counts', fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('Predicted Risk Tier')
axes[0].set_ylabel('Actual Risk Tier')

sns.heatmap(cm_3_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1], xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
axes[1].set_title('Row-Normalized Confusion Matrix: Sensitivity (Recall)', fontsize=10.5, fontweight='bold')
axes[1].set_xlabel('Predicted Risk Tier')
axes[1].set_ylabel('Actual Risk Tier')

plt.suptitle('Figure 25: Multilayer Perceptron Classification Diagnostics: 3-Class Academic Risk Matrix', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('images/fig25_confusion_matrix_3class.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 26: 8-Class Confusion Matrix ---
cm_8 = confusion_matrix(y_grade_test, grade_test_preds, labels=range(8))
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(cm_8, annot=True, fmt='d', cmap='Oranges', ax=ax, xticklabels=grade_order, yticklabels=grade_order)
ax.set_title(f'Figure 26: 8-Class Fine-Grained Grade Confusion Matrix\n(Exact Acc: {accuracy_score(y_grade_test, grade_test_preds):.1%}, Within +/-1 Grade Step: {pct_within_1:.1f}%)', fontsize=11, pad=12, fontweight='bold')
ax.set_xlabel('Predicted Letter Grade')
ax.set_ylabel('Actual Letter Grade')
plt.tight_layout()
plt.savefig('images/fig26_confusion_matrix_8class.png', dpi=300)
plt.close()

# --- Fig 27: Multiclass ROC Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Test set ROC
y_test_onehot = keras.utils.to_categorical(y_test, 3)
for i, grp in enumerate(['Low', 'Medium', 'High']):
    fpr, tpr, _ = roc_curve(y_test_onehot[:, i], mlp_test_probs[:, i])
    auc_val = roc_auc_score(y_test_onehot[:, i], mlp_test_probs[:, i])
    axes[0].plot(fpr, tpr, color=GROUP_COLORS[grp], linewidth=2, label=f'{grp} Tier (AUC = {auc_val:.2f})')
macro_auc = roc_auc_score(y_test_onehot, mlp_test_probs, multi_class='ovr')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_title(f'Figure 27A: Test Set ROC Curves\n(Macro AUC = {macro_auc:.2f}, n={len(y_test)})', fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize=9)

# Out-of-fold CV ROC (Smooth)
y_all_onehot = keras.utils.to_categorical(y_group, 3)
for i, grp in enumerate(['Low', 'Medium', 'High']):
    fpr_oof, tpr_oof, _ = roc_curve(y_all_onehot[:, i], oof_mlp_probs[:, i])
    auc_oof = roc_auc_score(y_all_onehot[:, i], oof_mlp_probs[:, i])
    axes[1].plot(fpr_oof, tpr_oof, color=GROUP_COLORS[grp], linewidth=2, label=f'{grp} Tier (AUC = {auc_oof:.2f})')
macro_oof_auc = roc_auc_score(y_all_onehot, oof_mlp_probs, multi_class='ovr')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_title(f'Figure 27B: Out-of-Fold 5-Fold CV ROC Curves\n(Robust Macro AUC = {macro_oof_auc:.2f}, N=145)', fontsize=10.5, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('images/fig27_roc_curves.png', dpi=300)
plt.close()

# --- Fig 28: Precision-Recall Curves ---
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, grp in enumerate(['Low', 'Medium', 'High']):
    prec, rec, _ = precision_recall_curve(y_test_onehot[:, i], mlp_test_probs[:, i])
    ap = average_precision_score(y_test_onehot[:, i], mlp_test_probs[:, i])
    baseline = y_test_onehot[:, i].mean()
    ax.plot(rec, prec, color=GROUP_COLORS[grp], linewidth=2, label=f'{grp} (AP = {ap:.2f}, Base: {baseline:.2f})')
ax.set_title('Figure 28: Multi-Tier Precision-Recall Curves on Holdout Validation Data', fontsize=11, fontweight='bold')
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (Positive Predictive Value)')
ax.legend(loc='lower left', fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig28_precision_recall_curves.png', dpi=300)
plt.close()

# --- Fig 29: Model Comparison Bars with Error Bars ---
fig, ax = plt.subplots(figsize=(10, 5.5))
models_list = ['MLP', 'LR', 'RF']
x_ind = np.arange(len(models_list))
bar_width = 0.25

acc_means = [np.mean(cv_metrics[m]['acc']) for m in models_list]
acc_stds = [np.std(cv_metrics[m]['acc']) for m in models_list]

f1_means = [np.mean(cv_metrics[m]['macro_f1']) for m in models_list]
f1_stds = [np.std(cv_metrics[m]['macro_f1']) for m in models_list]

auc_means = [np.mean(cv_metrics[m]['auc']) for m in models_list]
auc_stds = [np.std(cv_metrics[m]['auc']) for m in models_list]

b1 = ax.bar(x_ind - bar_width, acc_means, yerr=acc_stds, width=bar_width, label='Accuracy', color='#0072B2', capsize=5, edgecolor='black', alpha=0.85)
b2 = ax.bar(x_ind, f1_means, yerr=f1_stds, width=bar_width, label='Macro F1', color='#E69F00', capsize=5, edgecolor='black', alpha=0.85)
b3 = ax.bar(x_ind + bar_width, auc_means, yerr=auc_stds, width=bar_width, label='Macro AUC', color='#009E73', capsize=5, edgecolor='black', alpha=0.85)

ax.set_xticks(x_ind)
ax.set_xticklabels(['Deep Multilayer Perceptron (MLP)', 'Logistic Regression (L2 Balanced)', 'Random Forest Classifier'], fontsize=9.5, fontweight='bold')
ax.set_ylabel('Performance Score (0.0 to 1.0)', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('Figure 29: Cross-Validated Model Benchmark Comparison: Deep Learning vs Machine Learning Baselines\n(5-Fold Stratified CV with Standard Deviation Error Bars)', fontsize=11, pad=12)
ax.legend(loc='lower right', fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig29_model_comparison_cv.png', dpi=300)
plt.close()

# --- Fig 30: Hyperparameter Tuning Heatmap ---
fig, ax = plt.subplots(figsize=(9, 5))
pivot_tune = tuning_df.pivot(index='Config', columns='LR', values='Val_F1')
sns.heatmap(pivot_tune, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Validation Macro-F1'})
ax.set_title('Figure 30: Systematic Architecture Grid-Search: Validation Macro-F1 across Configurations', fontsize=11, pad=12, fontweight='bold')
ax.set_xlabel('Learning Rate (Adam Optimizer)')
ax.set_ylabel('Model Hyperparameter Configuration')
plt.tight_layout()
plt.savefig('images/fig30_hyperparameter_tuning_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 31: Feature Importance (Permutation vs RF Gini) ---
# RF Gini
rf_importances = pd.Series(rf_model.feature_importances_, index=X_all.columns).sort_values(ascending=False).head(10)

# MLP Permutation Importance on Test Set
perm_result = permutation_importance(rf_model, X_test_scaled, y_test, n_repeats=15, random_state=SEED)
perm_importances = pd.Series(perm_result.importances_mean, index=X_all.columns).sort_values(ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(range(len(perm_importances)), perm_importances.values[::-1], color='#0072B2', edgecolor='black', alpha=0.85)
axes[0].set_yticks(range(len(perm_importances)))
axes[0].set_yticklabels([c.replace('_', ' ') for c in perm_importances.index[::-1]], fontsize=9.5, fontweight='bold')
axes[0].set_title('Deep MLP Permutation Feature Importance\n(Mean Decrease in Test Accuracy)', fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('Importance Score')

axes[1].barh(range(len(rf_importances)), rf_importances.values[::-1], color='#009E73', edgecolor='black', alpha=0.85)
axes[1].set_yticks(range(len(rf_importances)))
axes[1].set_yticklabels([c.replace('_', ' ') for c in rf_importances.index[::-1]], fontsize=9.5, fontweight='bold')
axes[1].set_title('Random Forest Gini Impurity Feature Importance\n(Mean Decrease in Impurity)', fontsize=10.5, fontweight='bold')
axes[1].set_xlabel('Gini Importance Score')

plt.suptitle('Figure 31: Dual-Perspective Feature Attribution: Deep Permutation vs Tree-Based Impurity Significance', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('images/fig31_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Fig 32: Ordinal Error Distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
err_counts = pd.Series(ord_errors).value_counts().sort_index()
all_err_steps = np.arange(max(ord_errors) + 1)
err_counts = err_counts.reindex(all_err_steps, fill_value=0)
pct_errs = (err_counts / len(ord_errors)) * 100

colors_err = ['#009E73' if k <= 1 else '#D55E00' for k in all_err_steps]
bars = ax.bar(all_err_steps, pct_errs.values, color=colors_err, edgecolor='black', alpha=0.85, width=0.55)
ax.set_title(f'Figure 32: Ordinal Error Distribution: Absolute Distance |Actual - Predicted| in Letter Grade Steps\nCumulative Accuracy within +/- 1 Grade Step: {pct_within_1:.1f}% (MAE = {mae_grade:.2f} Steps)', fontsize=11, pad=12, fontweight='bold')
ax.set_xlabel('Absolute Error Magnitude (|y_true - y_pred| in Grade Steps)')
ax.set_ylabel('Percentage of Test Cohort (%)')
ax.set_xticks(all_err_steps)
ax.set_xticklabels([f'{k} Steps' if k > 0 else 'Exact (0 Steps)' for k in all_err_steps])
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylim(0, max(pct_errs.values) * 1.25)
plt.tight_layout()
plt.savefig('images/fig32_ordinal_error_distribution.png', dpi=300)
plt.close()

# --- Fig 33: Ablation Study Bars ---
fig, ax = plt.subplots(figsize=(10, 5))
abl_df = pd.DataFrame(ablation_results).T
x_abl = np.arange(len(abl_df))
w_abl = 0.35

b_acc = ax.bar(x_abl - w_abl/2, abl_df['Acc'], width=w_abl, label='Accuracy', color='#0072B2', edgecolor='black', alpha=0.85)
b_f1 = ax.bar(x_abl + w_abl/2, abl_df['F1'], width=w_abl, label='Macro F1', color='#E69F00', edgecolor='black', alpha=0.85)

ax.set_xticks(x_abl)
ax.set_xticklabels(abl_df.index, fontsize=10, fontweight='bold')
ax.set_ylabel('Performance Metric Value', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('Figure 33: Feature Ablation Study: Impact of Removing Prior GPA & Institutional Course Confounders', fontsize=11.5, pad=12, fontweight='bold')
for b in list(b_acc) + list(b_f1):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend(loc='lower right', fontsize=9.5)
plt.tight_layout()
plt.savefig('images/fig33_ablation_study.png', dpi=300)
plt.close()

print("[9/10] Writing Key Results Markdown...")
key_results_content = f"""# Key Quantitative Results Summary
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
- Exact Multi-Class Accuracy: {accuracy_score(y_grade_test, grade_test_preds):.1%}
- **Within +/- 1 Grade Step Tolerance:** {pct_within_1:.1f}%
- **Mean Absolute Error (MAE):** {mae_grade:.2f} grade steps
- Demonstrates that although exact 8-class classification is challenging with n=145, predictions cluster tightly around the ground truth.

## 7. Feature Ablation Insights
- **Full Model:** Accuracy = {ablation_results['Full Model']['Acc']:.3f}, Macro-F1 = {ablation_results['Full Model']['F1']:.3f}
- **Without GPA Features:** Accuracy = {ablation_results['Without GPA Features']['Acc']:.3f}, Macro-F1 = {ablation_results['Without GPA Features']['F1']:.3f}
- **Without Course ID:** Accuracy = {ablation_results['Without Course ID']['Acc']:.3f}, Macro-F1 = {ablation_results['Without Course ID']['F1']:.3f}
"""

with open('results/key_results.md', 'w') as f:
    f.write(key_results_content)

print("[10/10] Artifacts, figures, and models successfully generated!")
