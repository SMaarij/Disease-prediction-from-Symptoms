# Basic imports
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


# Reading the dataset 
dataset = pd.read_excel('./Dataset/dataset.xlsx')
print(dataset.head)

data = dataset.fillna(method='ffill') # Filling all the Null values
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



f = open('./Dataset/cleaned_data.csv', 'w') # saving cleaned data
with f:
    writer = csv.writer(f)
    for key, val in disease_symptom_dict.items():
        for i in range(len(val)):
            writer.writerow([key, val[i], disease_symptom_count[key]])
            
            

df = pd.read_csv('./Dataset/cleaned_data.csv') # Reading cleaned data
df.columns = ['disease', 'symptom', 'occurence_count']
df.head()