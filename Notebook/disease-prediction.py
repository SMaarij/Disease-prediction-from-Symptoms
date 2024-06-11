# Basic imports
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


# Reading the dataset 
dataset = pd.read_excel('Notebook\Dataset\dataset.xlsx')
# print(dataset.head)

data = dataset.ffill() # Filling all the Null values
print(data)


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

# print("Disease-Symptom Dictionary:", dict(disease_symptom_dict))
# print("Disease-Symptom Count:", disease_symptom_count)



f = open('Notebook\Dataset\cleaned_data.csv', 'w') # saving cleaned data
with f:
    writer = csv.writer(f)
    for key, val in disease_symptom_dict.items():
        for i in range(len(val)):
            writer.writerow([key, val[i], disease_symptom_count[key]])
            
            
file_path = 'Notebook\Dataset\cleaned_data.csv' # reading the cleaned dataset

df = pd.read_csv(file_path, header=None, encoding='latin1')
df.columns = ['disease', 'symptom', 'occurrence_count']
print(df.head())

#Replace all float('nan') with np.nan (standard way to represent missing values in pandas)
df.replace(float('nan'), np.nan, inplace=True)
#Removing rows having missing value(np.nan)
df.dropna(inplace=True)

#total number of unique symtoms 
n_unique = len(df['symptom'].unique())
print(n_unique)
#data type of columns
print(df.dtypes)


#convert categorical labels in the 'symptom' column into columns and these columns will contain boolean labels by using LabelEncoder and OneHotEncoder technique. This process is useful when we have categorical data and need to convert it into a numerical format for certain machine learning algorithms
from sklearn import preprocessing
#LabelEncoder from sklearn.preprocessing to convert categorical labels in the 'symptom' column of a DataFrame into integer-encoded labels
from sklearn.preprocessing import LabelEncoder
#  One-hot Encode the Integer-encoded Labels using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the 'symptom' column
integer_encoded = label_encoder.fit_transform(df['symptom'])
print(integer_encoded)
#  One-hot Encode the Integer-encoded Labels using OneHotEncoder
# Initialize the OneHotEncoder with sparse_output set to False
onehot_encoder = OneHotEncoder(sparse_output=False)
# Reshape integer_encoded to a 2D array as required by OneHotEncoder
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
print(onehot_encoded[0])
cols = np.asarray(df['symptom'].unique())
print(cols)

# Create a new dataframe to save OHE labels
df_ohe = pd.DataFrame(columns = cols)
print(df_ohe.head())
for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]
print(df_ohe.head())