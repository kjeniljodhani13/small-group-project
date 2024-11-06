import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("covid_data.csv")

numerical_features = [
    'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
    'reproduction_rate', 'icu_patients', 'hosp_patients', 
    'total_tests', 'positive_rate', 'total_vaccinations', 
    'people_vaccinated', 'stringency_index', 'population_density', 
    'median_age', 'gdp_per_capita', 'diabetes_prevalence'
]

numerical_features = [col for col in numerical_features if col in df.columns]

plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Visualization 2: Histogram of New Cases
plt.figure(figsize=(10, 6))
sns.histplot(df['new_cases'].dropna(), bins=30, kde=True)
plt.title('Distribution of New Cases')
plt.xlabel('New Cases')
plt.ylabel('Frequency')
plt.show()

# Visualization 3: Scatter Plot of Total Cases vs. Total Deaths
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='total_cases', y='total_deaths')
plt.title('Scatter Plot of Total Cases vs. Total Deaths')
plt.xlabel('Total Cases')
plt.ylabel('Total Deaths')
plt.show()

