import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time

"""
citation: Data is from http://archive.ics.uci.edu/ml/datasets/banknote+authentication
"""

#import data based on fake vs. real banknotes
banknote_data = pd.read_csv("data_banknote_authentication.txt",delimiter=",",
                            names = ["Variance","Skewness","Curtosis","Entropy","Class"])

#create named lists of all the features
variance = banknote_data["Variance"]
skewness = banknote_data["Skewness"]
curtosis = banknote_data["Curtosis"]
entropy = banknote_data["Entropy"]

#where 0 = genuine and 1 = fake, seperate the feature data from the label data
new_banknote_data = banknote_data.drop("Class", 1)
#label only data
class_note_data = banknote_data.drop(["Variance","Skewness","Curtosis","Entropy"],1)

#making the data usable
def prepData(class_dataset, feature_dataset):
    labels = class_dataset.to_numpy()
    labels = np.squeeze(labels)
    features = feature_dataset.to_numpy()
    return labels, features

#making the variables according to the data prep function
labels, features = prepData(class_note_data, new_banknote_data)

#start recording the runtime
start = time.time()

#split the data and use 2/3 for training and 1/3 for testing
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

#use the Naive Bayes classifier
gnb = GaussianNB()

#fit the classifier to the model
model = gnb.fit(train,train_labels)

#make predictions
preds = gnb.predict(test)

#stop the run time calculator and show the total time
end = time.time()
preds_time = end - start

#results of the Naive Bayes classifier
print("The Naive Bayes classifier takes", preds_time, "many seconds to process.")
print("The Naive Bayes classifier has a",accuracy_score(test_labels, preds)*100,"% accuracy score.")
print(classification_report(test_labels, preds))

#make a confusion matrix
gnb_matrix = confusion_matrix(test_labels, preds, labels = [0,1])

#make the confusion matrix look nice and show it
fig, ax = plt.subplots()
ax.set_title("Authenticity of Banknotes w/ Naive Bayes")

sns.heatmap(gnb_matrix/np.sum(gnb_matrix), annot=True, fmt = ".2%",
            xticklabels=["Authentic","Counterfeit"], yticklabels=["Authentic","Counterfeit"])

plt.xlabel("Predicted Value", fontsize = 10)
plt.ylabel("Actual Value")


#prepare data for the second classifier
def prepData_2(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    return X, y

#create variables using the second prep data function
X, y = prepData_2(banknote_data)

#looking at distribution of variables
y_df = banknote_data["Class"].values.reshape(-1,1)
X_df = banknote_data.copy()
X_df.drop(["Class"], axis=1, inplace=True)

#show boxplot of the nonstandardized distributions of the features
plt.figure(figsize=(8,(X_df.shape[-1] // 2) * 4))
idx = 1
for Feature in X_df:
    plt.subplot((X_df.shape[-1] // 2),2, idx)
    idx += 1
    plt.boxplot(X_df[Feature], meanline=False, notch=True, labels=[''])
    plt.title(Feature)
plt.tight_layout()

#start recording the runtime of the second classifier
start = time.time()

#split up the data so that 33% of it is used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#normalizing the data so it's less sensitive to feature transformation
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#use the KNN classifier on the data, w/ the K-value being 5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

#get predictions using the KNN classifier
y_pred = classifier.predict(X_test)

#stop the run time calculator
end = time.time()
preds_time_2 = end - start

#show results
print("The K-Nearest Neighbours classifier takes",preds_time_2,"many seconds to process.")
print("The K-Nearest Neighbours classifier has a",accuracy_score(y_test, y_pred)*100,"% accuracy score.")
print(classification_report(y_test, y_pred))

knn_matrix = confusion_matrix(y_test, y_pred, labels = [0,1])

#show the confusion matrix for the KNN Classifier
fig1, ax1 = plt.subplots()
ax1.set_title("Authenticity of Banknotes - KNN w/ k = 5")

sns.heatmap(knn_matrix/np.sum(knn_matrix), annot=True, fmt = ".2%",
            xticklabels=["Authentic","Counterfeit"], yticklabels=["Authentic","Counterfeit"])

plt.xlabel("Predicted Value", fontsize = 10)
plt.ylabel("Actual Value")

#make a list to hold errors
error = []

#calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#graph the error rate for the K-values
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title("Error Rate of the K Value")
plt.xlabel("K Value")
plt.ylabel("Mean Error")

#show all the visualizations
plt.show()

