import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("covid_data.csv")

print(df.info())

df['total_cases'] = df['total_cases'].fillna(0) 
df = df.dropna(subset=['total_deaths', 'new_cases', 'total_cases_per_million']) 

df['date'] = pd.to_datetime(df['date'])

features = [
    'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'reproduction_rate',
    'icu_patients', 'hosp_patients', 'total_tests', 'positive_rate', 
    'total_vaccinations', 'people_vaccinated', 'stringency_index', 
    'population_density', 'median_age', 'gdp_per_capita', 
    'diabetes_prevalence', 'continent', 'location'
]

categorical_features = ['continent', 'location']
numerical_features = [f for f in features if f not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X = df[features]
y = df['new_cases']

X_preprocessed = preprocessor.fit_transform(X)

