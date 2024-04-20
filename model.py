import numpy as np # Linear Algebra
import pandas as pd # Handling DataFrame
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

import pickle

import warnings # To mitigate any unwanted warnings
warnings.filterwarnings('ignore')

## load dataset
heart_data = pd.read_csv('G:/Heart_Disease_Prediction/modified_dataset.csv')
print(heart_data.columns)



# Dropping the duplicate values
heart_data.drop_duplicates(inplace = True)
print(f'Number of Duplicate values: {heart_data.duplicated().sum()}')

# Getting the numerical columns
num_cols = heart_data.select_dtypes(include = ['number']).columns.tolist()
cat_cols = heart_data.select_dtypes(include = ['object', 'category']).columns.tolist()

print(f'Numerical Columns: \n{num_cols}\n\nCategorical Columns: \n{cat_cols}')

data_new = heart_data.copy() ### taking caopy of the dataset

# Remove outliers from the  column
data_new.drop(data_new[data_new['bmi'] < 17].index, axis = 0, inplace = True)
data_new.drop(data_new[data_new['bmi'] > 41].index, axis = 0, inplace = True)

# Remove outliers from the  column
data_new.drop(data_new[data_new['physical_health'] < 3 ].index, axis = 0, inplace = True)
data_new.drop(data_new[data_new['physical_health'] > 30.0].index, axis = 0, inplace = True)

# Remove outliers from the unemployment column
data_new.drop(data_new[data_new['sleep_time'] < 3].index, axis = 0, inplace = True)
data_new.drop(data_new[data_new['sleep_time'] > 11].index, axis = 0, inplace = True)

### Encoding 10 Feature
heart_df = data_new.copy()

binary_columns = ['heart_disease','diff_walking']

for column in binary_columns:
   heart_df[column] = heart_df[column].replace({'No':0 , 'Yes':1})

heart_df['sex'] = heart_df['sex'].replace({'Female':0 ,  'Male':1}) 

heart_df['gen_health'] = heart_df['gen_health'].map({'Poor':0, 'Fair':1, 'Good':2, 'Very good':3, 'Excellent':4}) 

heart_df['diabetic'] = heart_df['diabetic'].replace({'No':0 , 'No, borderline diabetes':1 ,  'Yes (during pregnancy)':2 , 'Yes':3}) 

col = 'sleep_time'
heart_df.loc[ (heart_df[col] <= 6) , col ] = 0 
heart_df.loc[ (heart_df[col] > 6) & (heart_df[col] <= 8 ) , col ] = 1
heart_df.loc[ (heart_df[col] > 8 ) & (heart_df[col] <=24), col ] = 2  

cols =['physical_health']

for col in cols:
        heart_df.loc[ heart_df[col] <=10 , col ] = 0
        heart_df.loc[ (heart_df[col] > 10) & (heart_df[col] <=20 ) , col ] = 1
        heart_df.loc[ (heart_df[col] > 20) & (heart_df[col] <=25 ) , col ] = 2
        heart_df.loc[ (heart_df[col] > 25) & (heart_df[col] <=30 ) , col ] = 3

def transform_bmi(value):
    if value<16:
        return 0
    elif value>=16 and value<17:
        return 1
    elif value>=17 and value<18.5:
        return 2
    elif value>=18.5 and value<25:
        return 3
    elif value>=25 and value<30:
        return 4
    elif value>=30 and value<35:
        return 5
    elif value>=35 and value<40:
        return 6
    elif value>=40 :
        return 7
heart_df["bmi"] = heart_df["bmi"].apply(transform_bmi)
heart_df["bmi"].value_counts(ascending= False)

mapping = {'18-24':0, '25-29': 0, '30-34':0, '35-39':1 , 
           '40-44':1,  '45-49':1, '50-54':2, '55-59':'2', 
           '60-64':'2','65-69':'3', '70-74':'3', '75-79':'3', '80 or older':'4'}


heart_df['age_category'] = heart_df['age_category'].replace(mapping).astype(int)

race_mapping = {
    'American Indian/Alaskan Native': 0,
    'Asian': 1,
    'Black': 2,
    'Hispanic': 3,
    'Other': 4,
    'White': 5
}
heart_df['race'] = heart_df['race'].map(race_mapping).astype(int)


# handle imblanced
X = heart_df.drop(['heart_disease'], axis=1)
y = heart_df['heart_disease']

scaler = MinMaxScaler(feature_range = (0, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)


smote = SMOTE(random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

### Spilt dataset

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Print the Shapes of Training and Testing Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of testing label:', y_test.shape)


decomposer = PCA(svd_solver='auto')
X_pca = decomposer.fit_transform(X_scaled)

### Modeling
# Define the parameter distribution for RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300,400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
}

# Initialize Random Forest Classifier
rf = RandomForestClassifier()

# Initialize RandomizedSearchCV
randomized_search_rf = RandomizedSearchCV(
    rf,
    param_distributions=param_dist_rf,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Perform RandomizedSearchCV
randomized_search_rf.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rf = randomized_search_rf.best_params_
print(f"Best Hyperparameters for Random Forest: {best_params_rf}")

# Get the best model
best_rf_model = randomized_search_rf.best_estimator_

# Make predictions on the test set
rf_predicted = best_rf_model.predict(X_test)

# Calculate accuracy
rf_acc_score = accuracy_score(y_test, rf_predicted)

# Print classification report
print("Classification Report for Random Forest:")
print(classification_report(y_test, rf_predicted))

pickle.dump(best_rf_model, open("model.pkl", "wb"))