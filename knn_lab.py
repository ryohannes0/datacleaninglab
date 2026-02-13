# %%
# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load dataset #1
df = pd.read_csv("/workspaces/datacleaninglab/cc_institution_details.csv")
# Question 1 restated:Can we predict which institutions will have high student retention rates based on financial aid levels? 

#%%
# classification of target variable
df["high_retention"] = (df["retain_value"] >= 70).astype(int)

# now I have to confirm the binary distribution of the target variable
print(df["high_retention"].value_counts())
# The target variable should be treated as a category, not a continuous number, so the classification is correct. 
# The variable is not treated as a continuous/numerical value.
# 1 is high retention, 0 is low retention.


# %%
#[markdown]
# Step 1
# now we have to separate the predictors and target variable (define the X(predictors) and y(target/high_retention))

essential_columns = [
    "pell_value", 
    "aid_value",
    "student_count"
    
]

#%%

# fill in missing values using median imputation
for col in essential_columns:
    df[col] = df[col].fillna(df[col].median())

features = df[essential_columns]
target = df["high_retention"] # this contains the classification variable

#%%
# check for missing values within features
features[essential_columns].isna().sum()

# %%

# split the data into training and testing sets 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, 
    target, 
    test_size=0.2, 
    random_state=42
)
features.dtypes

#%%
#scale the features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[essential_columns]) #X_train[essential_columns] is used to select only the essential columns from the training data for scaling, to ensure there are no unneccessary columns left
X_test_scaled = scaler.transform(X_test[essential_columns]) # .fit_transform is used to compute the mean and standard deviation for scaling, while .transform is used on the test data to apply the same scaling parameters without recomputing

#%%
#[markdown]
# Step 2
# build the kNN model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) # we are instructed to predict the target variable using the 3 nearest neighbors
knn.fit(X_train_scaled, y_train) # knn.fit() is used to train the model on the training data to make predictions

#%%
#[markdown]
# Step 3
# test target values
y_test_values = y_test.values #the actual target values for the test data are given
y_pred = knn.predict(X_test_scaled) # this gives the predicted class labels for the test data
y_prob = knn.predict_proba(X_test_scaled)[:, 1] # this gives the predicted probabilities for the positive class (high retention (1))

#create the required dataframe 
results_df = pd.DataFrame({
    "test_target": y_test_values,
    "predicted_class": y_pred,
    "probability_positive_class": y_prob
})
print(results_df.head())

#%%
#[markdown]
# Step 4
print("If the k hyperparameter was adjusted, the threshold function would behave different because the predicted probabilities are based on the nearest neighbors. While k=3, the probabilities increase in increments of a third (1/3). Increasing the k would make the probability increase in smaller and msoother increments. This would change what observations cross the 0.5 classification threshold. Similarly, the confusion matrix would probably not look the same at the same threshold because different values of k would create different predicted class labels.")


#%%
#[markdown]
# Step 5
#now we have to compute the evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc_score = accuracy_score(y_test_values, y_pred)
print("Accuracy Score:", acc_score) #shows percentage of correct predictions
conf_matrix = confusion_matrix(y_test_values, y_pred)
print("Confusion Matrix:" , conf_matrix) #shows the true positives/negatives and false positives/negatives
class_rep = classification_report(y_test_values, y_pred)
print("Classification Report:\n", class_rep) #shows precision, recall, f1-score for each class

# The confusion matrix is showing that there are 360 true negatives (correctly predicted low retention)
# and 177 true positives (correctly predicted high retention).
# There are 105 false positives (incorrectly predicted high retention) 
# and 118 false negatives (incorrectly predicted low retention).

#%%
#Step 5 interpretation of the results
print("The kNN model was about 71% (70.7) accurate meaning that approximately 71% of institutions were classified correctly. The confusion matrix showed that the model correctly identified 360 low-retention institutions and 177 high-retention institutions. Although, it incorrectly classified 118 high-retention institutions as low retention. This indicates a weaker recall for the positive class. The classification report showed the model performs best at predicting low retention than high retention (precision was 0.63 for high-retention vs. 0.75 for low). This suggests that although financial aid and institution size has predictive power, they don't fully factor variables impacting student retention. Additionally, having k=3 may not be the optimal number of neighbors for this dataset.")

#%%
#Step 6
#function for cleaning data and split into training|test
def prepare_data(df, target_col, essential_columns):
    
    df = df.copy()
    
    #binary target
    df["binary_target"] = (df[target_col] >= df[target_col].median()).astype(int)
    
    #filling missing values
    df[essential_columns] = df[essential_columns].fillna(df[essential_columns].median())
    
    #split
    X = df[essential_columns]
    y = df["binary_target"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

#function for training and testing mode with different k and threshold combinations
def train_and_evaluate(X_train, X_test, y_train, y_test, k, threshold):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    
    return acc, cm

#test combinations OUTSIDE of function
#simply replace retention_rate with real target variable and essential_columns with the actual columns you want to use as predictors
# %%
X_train, X_test, y_train, y_test = prepare_data(df, "retain_value", essential_columns)

k_values = [3, 5, 7]
thresholds = [0.5, 0.6, 0.7]
results = []

for k in k_values:
    for threshold in thresholds:
        acc, cm = train_and_evaluate(X_train, X_test, y_train, y_test, k, threshold)
        results.append((k, threshold, acc, cm))

#%%
#Step 7
print("The model performed moderately well with the accuracy at around 70%. Testing values of k and thresholds changed the model to stable when fixing the thresholds which impacted both precision and recall. A lower threshold makes the predicted high-retention institutions have a better recall while also increasing false positives. Overall, when adjusting k and threshold, the model improves its classification balance. However, the number of predictors being limited likely restricts the model's performance.")
# %%
# Step 8
# choose another variable as target and repeat the process.
#prepare data function already creates binary target

#define features
essential_columns = ["retain_value", "student_count"]

#call prepare_data with new target
X_train, X_test, y_train, y_test = prepare_data(
    df,
    "aid_value", #changed target
    essential_columns
)

#train model
acc, cm = train_and_evaluate(X_train, X_test, y_train, y_test, k=5, threshold=0.5)

print("Accuracy:", round(acc, 3))
print("confusion Matrix:")
print(cm)

#This model is less accurate, and indicates that retention and student count may not be a strong predictor of aid levels.

