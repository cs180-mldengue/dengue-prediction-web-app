import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv('dengue-dataset.csv')

# Define label-value mapping
desired_order = ['Minimal to No risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Severe Risk']
label_mapping = {
    'Severe Risk': 4,
    'High Risk': 3,
    'Moderate Risk': 2,
    'Low Risk': 1,
    'Minimal to No risk': 0
}

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit LabelEncoder with the unique labels from mapping dictionary
label_encoder.fit(desired_order)

# Encode the labels using the mapping dictionary
df['labels'] = df['labels'].map(label_mapping)

# Select the columns to exclude from the train set
target_column = 'labels'
columns_to_exclude = ['serial', 'cases',target_column,'snow','snowdepth']

# Assuming your dataset is stored in a pandas DataFrame called 'data'
columns_to_drop = [col for col in df.columns if 'none' in df[col].values]

# Drop the identified columns from the dataset
data = df.drop(columns_to_drop, axis=1)

# Assign X to train variable and y to target variable
X = df.drop(columns=columns_to_exclude)
y = df[target_column]

xgb_params = {'colsample_bytree': 0.8, 'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
rf_params = {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}
svm_params = {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}

xgb_model = xgb.XGBClassifier(**xgb_params) # best performing model

xgb_model.fit(X,y)

pickle.dump(xgb_model, open('xgb_model.pkl','wb')) 

label_mapping = {
    4: 'Severe Risk',
    3: 'High Risk',
    2: 'Moderate Risk',
    1: 'Low Risk',
    0: 'Minimal to No risk'
}

# Get feature importance scores
importance_type = "weight"  # Choose either "weight" or "gain"
importance_scores = xgb_model.get_booster().get_score(importance_type=importance_type)

# Print feature importance scores
for feature, score in importance_scores.items():
    print(f"{feature}: {score}")
