#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
[markdown]
#Question 4: College Retention Prediction Pipeline

def load_college_data():
    # load college retention dataset
    return pd.read_csv("/workspaces/datacleaninglab/cc_institution_details.csv")

def clean_college_data(df):
    # drop columns not used in modeling
    df = df.drop(columns=[
        'hbcu', 'flagship', 'chronname', 'city', 'site', 'unitid',
        'long_x', 'lat_y', 'med_sat_value', 'med_sat_percentile',
        'vsa_year', 'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first',
        'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first',
        'vsa_grad_after6_first', 'vsa_grad_elsewhere_after6_first',
        'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first',
        'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer',
        'vsa_enroll_after4_transfer', 'vsa_enroll_elsewhere_after4_transfer',
        'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer',
        'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer'
    ])
    
    # create binary target variable
    df["high_retention"] = (df["retain_value"] >= 70).astype(int)
    
    return df


def prepare_college_features(df):
    # select predictor variables
    features = [
        "aid_value",
        "student_count",
        "awards_per_value",
        "control",
        "level"
    ]
    
    X = df[features]
    y = df["high_retention"]
    
    # one-hot encode categorical features
    X = pd.get_dummies(
        X,
        columns=["control", "level"],
        drop_first=True
    )
    
    # fill missing financial aid values
    X["aid_value"] = X["aid_value"].fillna(
        X["aid_value"].median()
    )
    
    return X, y

def split_and_scale_data(X, y):
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # standardize numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

#dag
# run college retention pipeline
college_df = load_college_data()
college_df = clean_college_data(college_df)
X_college, y_college = prepare_college_features(college_df)
Xc_train, Xc_test, yc_train, yc_test = split_and_scale_data(
    X_college, y_college
)

# %%
[markdown]
#Question 4: Placement Prediction Pipeline

def load_placement_data():
    # load MBA placement dataset
    return pd.read_csv("/workspaces/datacleaninglab/Placement_Data_Full_Class.csv")

def clean_placement_data(df):
    # keep relevant columns only
    df = df[
        ["ssc_p", "hsc_p", "degree_p", "mba_p", "workex", "status"]
    ]
    
    # create binary target variable
    df["placed"] = df["status"].apply(
        lambda x: 1 if x == "Placed" else 0
    )
    
    return df

def prepare_placement_features(df):
    # select predictor variables
    features = [
        "ssc_p",
        "hsc_p",
        "degree_p",
        "mba_p",
        "workex"
    ]
    
    X = df[features]
    y = df["placed"]
    
    # one-hot encode work experience
    X = pd.get_dummies(
        X,
        columns=["workex"],
        drop_first=True
    )
    
    # fill missing numeric values
    for col in ["ssc_p", "hsc_p", "degree_p", "mba_p"]:
        X[col] = X[col].fillna(X[col].median())
    
    return X, y

# run placement pipeline
placement_df = load_placement_data()
placement_df = clean_placement_data(placement_df)
X_place, y_place = prepare_placement_features(placement_df)
Xp_train, Xp_test, yp_train, yp_test = split_and_scale_data(
    X_place, y_place
)

