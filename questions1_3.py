# %%
[markdown] 
# Question 1: Can we predict which institutions will have high student retention rates based on financial aid levels? 
# Independent Business Metric: Assuming that higher retention rates result in higher institutional rankings 
# and student satisfaction, can we predict which institutions will achieve 
# high retention (≥70%) in the upcoming academic year based on their financial 
# aid per student? 


# %%
# Import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset #1
institution = pd.read_csv("/workspaces/datacleaninglab/cc_institution_details.csv")
institution.info()

# %%
# Keep only necessary columns 
print(institution.columns.tolist())
institution = institution.drop(columns=[
    'hbcu', 'flagship', 'chronname', 'city', 'site', 'unitid', 'long_x', 'lat_y', 'med_sat_value', 'med_sat_percentile', 'vsa_year', 'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first', 'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first', 'vsa_grad_after6_first', 'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first', 'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer', 'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer', 'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer'
])
# Columns unrelated to retention or financial aid

#%%
institution.dtypes

#%%
institution["high_retention"] = institution["retain_value"].apply(
    lambda x: 1 if x >= 70 else 0
)

print("This indictes that 1517 out of 3798 institutions have high retention rates.")
institution["high_retention"].value_counts()

#%% 
prev = institution["high_retention"].mean()
print(prev)
# The baseline is the prevalence of high retention institutions and predicts the majority, in this case the prevalence of institutions with high retention rates is about 40%.

#%%
numeric_features = [
    "aid_value",
    "student_count",
    "awards_per_value"
]

categorical_features = [
    "control",
    "level"
]

x1 = institution[numeric_features + categorical_features]
y1 = institution["high_retention"]

#%%
x_encoded = pd.get_dummies(
    x1,
    columns=categorical_features,
    drop_first=True
)

#%%
# Filled in the na values with the median of the column ['aid_value']
# Handle missing values (median imputation for numeric features)
x_encoded["aid_value"] = x_encoded["aid_value"].fillna(
    x_encoded["aid_value"].median()
)

#%%
#train/test split

X_train, X_test, y_train, y_test = train_test_split(
    x_encoded,
    y1,
    test_size=0.2,
    random_state=42,
    stratify=y1
)
#%%
#scale/center
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

#%%
# Check for missing values in the encoded dataset
x_encoded.isna().sum()

#%%
#model eval vs baselibe

y_prediction = model.predict(X_test_scaled)

model_accuracy = accuracy_score(y_test, y_prediction)
print("Baseline accuracy:", prev)
print("Model accuracy:", model_accuracy)
print(classification_report(y_test, y_prediction))

#%%
[markdown]
# The model outperforms the baseline prevalence rate, showing that
# financial aid and institutional traits provide indications
# past general classification.

# %%
# Question 2: Can we predict whether an MBA student will be placed after graduation based on their academic performance and work experience?
# Independent Business Metric: Assuming that higher placement rates improve program reputation and future enrollment, can we predict whether an MBA student will secure job placement after graduation based on academic performance (e.g., MBA percentage) and prior work experience?

#%%
# mba percentage
sns.histplot(placement["mba_p"], bins=20)
plt.title("Distribution of mba Percentages")
plt.show()

# Placement outcome
sns.countplot(x="placed", data=placement)
plt.title("Placement outcome distributed")
plt.show()

# MBA percentage vs placement
sns.boxplot(
    x="placed",
    y="mba_p",
    data=placement
)
plt.title("MBA Percentage vs Placement Outcome")
plt.show()


# %%
# Repeating same process as question 1 for question 2
 
# Load dataset #2
placement_url = ("/workspaces/datacleaninglab/Placement_Data_Full_Class.csv")
placement = pd.read_csv(placement_url)

placement = placement[[
    "ssc_p",
    "hsc_p",
    "degree_p", 
    "mba_p", 
    "workex", 
    "status" 
]]

# %%]
# binary target variable
placement["placed"] = placement["status"].apply(
    lambda x: 1 if x == "Placed" else 0
)

placement["placed"].value_counts()
print("This indicates that 148 out of 215 students were placed.")

# %%
# baseline/prevalence
prev2 = placement["placed"].mean()
print("Baseline:", prev2)
# This predicts that everyone is placed,  with a prevalence of about 69%.

# %%
# Define features + target variable
numeric_features = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "mba_p"
]

categorical_features = [
    "workex"
]

x2 = placement[numeric_features + categorical_features]
y2 = placement["placed"]

# one-hot enccode categorical variables
x2_encoded = pd.get_dummies(
    x2,
    columns=categorical_features,
    drop_first=True
)
# %%
#handle missing values
for col in numeric_features:
    x2_encoded[col] = x2_encoded[col].fillna(
        x2_encoded[col].median()
    )

# %%
#train/test split with stratification
X2_train, X2_test, y2_train, y2_test = train_test_split(
    x2_encoded,
    y2,
    test_size=0.2,
    random_state=42,
    stratify=y2
)

# %%
# scale/center
scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

# %%
# train model
model2 = LogisticRegression(max_iter=1000)
model2.fit(X2_train_scaled, y2_train)

# %% 
# model eval vs baseline
y2_prediction = model2.predict(X2_test_scaled)
model2_accuracy = accuracy_score(y2_test, y2_prediction)
print("Baseline accuracy:", prev2)
print("Model accuracy:", model2_accuracy)
print(classification_report(y2_test, y2_prediction))

# %%
[markdown]
# The logistic regression model outperforms the baseline placement rate, indicating that academic performance and prior work experience provide meaningful predictive power beyond majority-class classification.

# %%
[markdown]
# Question 3
# Reflection on Dataset Suitability

# Financial Aid and Student Retention Dataset
#This dataset does a good job of addressing the question of whether financial aid levels are associated with higher student retention rates. It includes key variables like average financial aid per student, institutional characteristics, and retention rates, which directly relate to the problem I am trying to analyze. One concern I have with the dataset is that key variables such as financial aid required median imputation for missing values, and while aid appears correlated with retention, this does not imply causation. That being said, I am cautious about drawing strong conclusions from the results. Retention is influenced by many factors that are not captured in the dataset, such as academic preparedness, campus support services, student engagement, and socioeconomic background. Additionally, retention is measured at the institutional level, which hides variation among individual students. The choice to classify retention as “high” using a 70% cutoff also simplifies what is truly a continuous outcome, which could lead to some loss of information.

#MBA Placement Dataset
#The dataset is well suited for predicting whether an MBA student will be placed after graduation, since it includes academic performance measures and prior work experience, both of which are commonly associated with job placement. However, there are a few limitations that may raise concern. The dataset is relatively small and appears to represent students from a single institution, which limits how well the results would generalize to other programs or job markets. In addition, placement outcomes are affected by factors that are not observed here, such as networking opportunities, interview skills, and current labor market conditions. While the data is useful for modeling trends, these limitations suggest the results should be interpreted as directional more than something definite.

