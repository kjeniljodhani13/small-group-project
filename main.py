# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('covid_data_total.csv')

# 1. Data Exploration
# Display the first few rows of the dataset to get an overview
print("First 5 rows of the dataset:\n", df.head())

# Display summary statistics for numerical columns
print("\nSummary statistics:\n", df.describe())

# Check for missing values in the dataset
print("\nMissing values:\n", df.isnull().sum())

# Convert 'date' into datetime type
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')




# 2. Visualize of heatmap: the correlation between numerical variables
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()



# 3. Visualization of box plot: Distribution of total cases by continent
# This helps to understand how new cases changed over dates
plt.figure(figsize=(8, 6))
sns.boxplot(x='continent', y='total_cases', data=df)
plt.title('total_cases by continent')
plt.show()


# 4. Visualization of bar plot: Distribution of total death by continent
# Aggregate data to get the total deaths per million for each continent
continent_data = df.groupby('continent')['total_deaths'].sum().reset_index()
plt.figure(figsize=(8, 6))
plt.bar(continent_data['continent'], continent_data['total_deaths'])
plt.title("Bar Plot of Total Deaths by Continent")
plt.xlabel("Continent")
plt.ylabel("Total Deaths")
plt.xticks(rotation=90)
plt.show()

# 5. Visualization of Line plot: Distribution of new cases by date
plt.figure(figsize=(8, 6))
plt.plot(df['date'], df['new_cases_smoothed']) # Pass x and y as series, not as keyword arguments
plt.title("Line Plot of new cases")
plt.xlabel("Date")
plt.ylabel("New Cases Smoothed")
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(df['total_cases'], df['total_deaths'])
plt.title("Line Plot of relationship between total cases and total deaths")
plt.xlabel("total_cases")
plt.ylabel("total_deaths")
plt.show()


# 6. Visualization of pair plot: Distribution of cases by continent
# Select relevant numerical columns and the categorical 'continent' column for hue
covid_data = df[['total_cases', 'new_cases_smoothed', 'total_deaths',  'total_vaccinations', 'continent']].dropna()

# Create the pair plot with 'continent' as the hue
sns.pairplot(covid_data, hue='continent', height=2.5)

plt.show()


# conclusion:
# There is a positive correlation between total deaths and total cases.
# Europe has the largest total deaths, followed by North America and Asia.
# December 2022 and January 2023 is the peak of the global coronavirus outbreak.






