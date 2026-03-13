import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 1. Feature Engineering: Map ratings to Satisfaction levels
def categorize_satisfaction(rating):
    if rating <= 2:
        return "Unhappy"
    elif rating == 3:
        return "Neutral"
    else:
        return "Very Good"

# Assuming 'win' is your DataFrame - apply satisfaction mapping
# win = pd.read_csv('your_data.csv') 
win['Satisfaction'] = win['Customer_rating'].apply(categorize_satisfaction)

# 2. Target Encoding
label_target = LabelEncoder()
y = label_target.fit_transform(win['Satisfaction'])

# 3. Define Feature Groups
numeric_features = ['Customer_care_calls', 'Cost_of_the_Product', 'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
categorical_features = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']

# 4. Preprocessing Pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Data Splitting (70% Train, 30% Test)
X = win[numeric_features + categorical_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Handling Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 7. Model Building with Logistic Regression Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, multi_class='ovr'))
])

# 8. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

# 9. Model Prediction and Evaluation
y_pred = grid_search.predict(X_test)

print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_target.classes_))
