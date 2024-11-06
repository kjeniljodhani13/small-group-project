import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score

df = pd.read_csv("covid_data.csv")

if df['total_cases_per_million'].isnull().any():
    print("NaN values found in target variable. Dropping these rows.")
    df = df.dropna(subset=['total_cases_per_million'])

numerical_features = ['total_cases', 'total_deaths', 'total_tests', 'population']
categorical_features = ['continent']

if df[numerical_features + categorical_features].isnull().any().any():
    print("NaN values found in feature variables. Handling these now.")

    for col in numerical_features:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_features:
        df[col] = df[col].fillna('missing')

if df[numerical_features + categorical_features].isnull().any().any():
    print("There are still NaN values in the features after imputation.")
    print(df[numerical_features + categorical_features].isnull().sum())
    exit()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X = df[numerical_features + categorical_features]
y = df['total_cases_per_million']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Training target shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing target shape: {y_test.shape}")

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

model = RandomForestRegressor(random_state=42)

try:
    model.fit(X_train_preprocessed, y_train)
    y_pred = model.predict(X_test_preprocessed)

    r2 = r2_score(y_test, y_pred)
    print("R^2 Score:", r2)
except Exception as e:
    print("An error occurred during model fitting or prediction:", str(e))

