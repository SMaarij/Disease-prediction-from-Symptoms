import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#EXPLORATORY DATA ANALYSIS(EDA)
# Load the dataset with the correct path and encoding
df = pd.read_csv(r'Dataset\cleaned_data.csv', encoding='latin1')  # or try 'ISO-8859-1' if 'latin1' does not work

# Display the first few rows of the dataset
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Univariate Analysis
# Histograms
df.hist(bins=30, figsize=(20, 15))
plt.show()

# Box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# Bivariate Analysis
# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Variable1', y='Variable2', data=df)
plt.show()

# Correlation matrix
corr = df.corr()
print(corr)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Multivariate Analysis
# Pair plot
sns.pairplot(df)
plt.show()

# Categorical Data Analysis
# Bar plot
plt.figure(figsize=(10, 6))
sns.countplot(x='CategoricalVariable', data=df)
plt.xticks(rotation=90)
plt.show()

# Pie chart
df['CategoricalVariable'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.show()

# Count the frequency of each category
category_counts = df['Variable2'].value_counts()
# Plot the histogram (bar chart)
plt.bar(category_counts.index, category_counts.values)
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Histogram of Categorical Variable')
plt.show()
