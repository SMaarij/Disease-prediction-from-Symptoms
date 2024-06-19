import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'Notebook\Dataset\training_data.csv', encoding='latin1')
# One Hot Encoded Features
X=df.drop(columns='prognosis')
# Labels
y = df['prognosis']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y,random_state=101)
print(len(X_train)), print(len(y_train))
print(len(X_test)), print(len(y_test))
# Check class distribution in the training set
train_counter = Counter(y_train)
test_counter = Counter(y_test)
# Ensure all classes are present in the training set
missing_classes = [print(cls) for cls in test_counter if cls not in train_counter]
#any class is not missing in the training set



#Model Training With python package
#The default entropy used which is Shannon entropy (measures the impurity or disorder in a set of data)
dt = DecisionTreeClassifier(criterion='entropy',random_state=0)
clf_dt=dt.fit(X_train, y_train)
disease_pred = clf_dt.predict(X_test)

#Evaluation Metrics
#training score
print("Accuracy on training set: {:.3f}".format(clf_dt.score(X_train, y_train)))
disease_real = y_test.values
for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0}\nActual: {1}\n'.format(disease_pred[i], disease_real[i]))
print("Accuracy on test set: {:.3f}".format(clf_dt.score(X_test, y_test)))
print("The accuracy is "+str(metrics.accuracy_score(y_test,disease_pred)*100)+"%")
conf_matrix=confusion_matrix(y_test,disease_pred)
print(confusion_matrix(y_test,disease_pred))
plt.figure(figsize=(10, 7))
unique_diseases = df['prognosis'].drop_duplicates()
target_names = unique_diseases
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(y_test,disease_pred,target_names=target_names))
#ROC_AUC metric
label_encoder = LabelEncoder()
# Fit and transform the string labels to numerical labels
y_true = label_encoder.fit_transform(y_test)
y_pred = label_encoder.transform(disease_pred)
print("Encoded y_true:", y_true)
print("Encoded y_pred:", y_pred)
print("Classes:", label_encoder.classes_)
# Get the classes from the label encoder
classes = list(label_encoder.classes_)
# Binarize the numerical labels
y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
y_pred_binarized = label_binarize(y_pred, classes=range(len(classes)))
print("Binarized y_true:\n", y_true_binarized)
print("Binarized y_pred:\n", y_pred_binarized)
from sklearn.metrics import roc_auc_score
# Compute ROC-AUC for macro and micro averages
roc_auc_macro = roc_auc_score(y_true_binarized, y_pred_binarized, average='macro')
roc_auc_micro = roc_auc_score(y_true_binarized, y_pred_binarized, average='micro')
print(f'Macro ROC-AUC: {roc_auc_macro}')
print(f'Micro ROC-AUC: {roc_auc_micro}')


#Visualizing a Decision Tree Model
# Create DOT data
#dot_data = tree.export_graphviz(dt,out_file=None,feature_names=X.columns,class_names=target_names)
# Draw graph
#graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
#Image(graph.create_png())
#graph.write_pdf("tree.pdf")

