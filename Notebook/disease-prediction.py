# Basic imports
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
#

# Data Collection
# Reading the dataset 
dataset = pd.read_excel(r'C:\Users\LENOVO\Desktop\ML CEP 11\Disease-prediction-from-Symptoms\Notebook\Dataset\dataset.xlsx')

# Data Preprocessing
data = dataset.ffill()  # Filling all the Null values
print(data)

def process_data(data): 
    """Function to process disease and symptom names."""
    data_list = []
    data_name = data.replace('^', '_').split('_')
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
    if (row['Disease'] != "\xc2\xa0") and (row['Disease'] != ""):
        disease = row['Disease']
        disease_list = process_data(disease)
        count = row['Count of Disease Occurrence']

    # Get the Symptoms Corresponding to Diseases
    if (row['Symptom'] != "\xc2\xa0") and (row['Symptom'] != ""):
        symptom = row['Symptom']
        symptom_list = process_data(symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count

# Saving cleaned data to CSV
csv_file_path = r'Dataset\cleaned_data.csv'
try:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for key, val in disease_symptom_dict.items():
            for i in range(len(val)):
                writer.writerow([key, val[i], disease_symptom_count[key]])
    print(f"Cleaned data saved successfully to '{csv_file_path}'")
except FileNotFoundError:
    print(f"Error: Directory 'Dataset' does not exist or file path '{csv_file_path}' is incorrect.")

# Continue with the rest of your code...
