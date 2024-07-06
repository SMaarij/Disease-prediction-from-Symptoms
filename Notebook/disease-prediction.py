# Basic imports
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# DATA COLLECTION
# Reading the dataset
dataset = pd.read_excel(r'Dataset\\training_data.csv')
print("Original Dataset:")
print(dataset.head())

# DATA PREPROCESSING
data = dataset.ffill() # Filling all the Null values
print("Data after filling null values:")
print(data.head())

def process_data(data): # Processing disease and symptom names
    data_list = []
    data_name = data.replace('^','_').split('_')
    n = 1
    for names in data_name:
        if (n % 2 == 0):
            data_list.append(names)
        n += 1
    return data_list

# Data Cleanup
disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}
count = 0

for idx, row in data.iterrows():
    # Get the Disease Names
    if (row['Disease'] !="\xc2\xa0") and (row['Disease'] != ""): # Making sure that the disease field is neither a specific space-breaking character nor an empty string
        disease = row['Disease']
        disease_list = process_data(disease)
        count = row['Count of Disease Occurrence']

    # Get the Symptoms Corresponding to Diseases
    if (row['Symptom'] !="\xc2\xa0") and (row['Symptom'] != ""): # Making sure that the symptom field is neither a specific space-breaking character nor an empty string
        symptom = row['Symptom']
        symptom_list = process_data(symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count

# Saving cleaned data
output_path = r'C:\Users\LENOVO\Desktop\ML CEP 3\Disease-prediction-from-Symptoms\Notebook\Dataset\cleaned_data.csv'
with open(output_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['disease', 'symptom', 'occurrence_count'])  # Writing the header
    for key, val in disease_symptom_dict.items():
        for i in range(len(val)):
            writer.writerow([key, val[i], disease_symptom_count[key]])

# Reading the cleaned dataset with a different encoding
file_path = r'C:\\Users\\LENOVO\\Desktop\\ML CEP 3\\Disease-prediction-from-Symptoms\\Notebook\\Dataset\\cleaned_data.csv'
# Assuming the CSV file has columns in the order: 'disease', 'symptom', 'occurrence_count'
df = pd.read_csv(file_path, encoding='ISO-8859-1')
print("Cleaned Dataset:")
print(df.head())

# Check column names
print("Columns in cleaned dataset:", df.columns)

# Replace all float('nan') with np.nan (standard way to represent missing values in pandas)
df.replace(float('nan'), np.nan, inplace=True)
# Removing rows having missing value(np.nan)
df.dropna(inplace=True)

# Check column names after cleaning
print("Columns after cleaning:", df.columns)

# Total number of unique symptoms
n_unique = len(df['symptom'].unique())
print("Total number of unique symptoms:", n_unique)
# Data type of columns
print("Data types of columns:")
print(df.dtypes)

# Convert categorical labels in the 'symptom' column into columns and these columns will contain boolean labels by using LabelEncoder and OneHotEncoder technique. This process is useful when we have categorical data and need to convert it into a numerical format for certain machine learning algorithms
from sklearn import preprocessing
# LabelEncoder from sklearn.preprocessing to convert categorical labels in the 'symptom' column of a DataFrame into integer-encoded labels
from sklearn.preprocessing import LabelEncoder
# One-hot Encode the Integer-encoded Labels using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the 'symptom' column
integer_encoded = label_encoder.fit_transform(df['symptom'])
print("Integer encoded symptoms:")
print(integer_encoded)
# One-hot Encode the Integer-encoded Labels using OneHotEncoder
# Initialize the OneHotEncoder with sparse_output set to False
onehot_encoder = OneHotEncoder(sparse_output=False)
# Reshape integer_encoded to a 2D array as required by OneHotEncoder
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print("One-hot encoded symptoms:")
print(onehot_encoded)
print(onehot_encoded[0])
cols = np.asarray(df['symptom'].unique())
print("Unique symptoms:", cols)

# Create a new dataframe to save OHE labels
df_ohe = pd.DataFrame(columns=cols)
print("Empty OHE DataFrame:")
print(df_ohe.head())
for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]
print("OHE DataFrame:")
print(df_ohe.head())

print("Length of OHE DataFrame:", len(df_ohe))

# Disease Dataframe
df_disease = df['disease']
print("Disease DataFrame:")
print(df_disease.head())

# Concatenate OHE Labels with the Disease Column
df_concat = pd.concat([df_disease, df_ohe], axis=1)
print("Concatenated DataFrame:")
print(df_concat.head())
df_concat.drop_duplicates(keep='first', inplace=True)
print("Concatenated DataFrame after dropping duplicates:")
print(df_concat.head())
print("Length of concatenated DataFrame:", len(df_concat))
cols = df_concat.columns
print("Columns in concatenated DataFrame:", cols)
cols = cols[1:]

# Since, every disease has multiple symptoms, combine all symptoms per disease per row
df_concat = df_concat.groupby('disease').sum()
df_concat = df_concat.reset_index()
print("Grouped and summed DataFrame:")
print(df_concat.head())
print("Length of grouped DataFrame:", len(df_concat))
df_concat.to_csv(r'C:\Users\LENOVO\Desktop\ML CEP 3\Disease-prediction-from-Symptoms\Notebook\Dataset\training_dataset.csv', index=False)

# One Hot Encoded Features
X = df_concat[cols]

# Labels
y = df_concat['disease']