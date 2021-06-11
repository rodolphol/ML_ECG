# Author: Rodolpho Lopes
# Date: November 2020
# Coursework for GRE COMP1801 - Machine Learning


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,learning_curve
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix, classification_report
import pickle
from datetime import date

# Import raw data
df_raw = pd.read_csv('rawdata/data.csv')

#################### PRE PROCESSING ###########################

# divide between features (X) and labels (y)
X_raw = df_raw.drop(['Unnamed: 0','y'], axis=1)
y_raw = df_raw['y']


# Check whether there are missing values in the dataset
if X_raw.isna().any().any() == True:
    X_raw = X_raw.fillna(value = 0) # add 0 for missing values 
else:
    X_raw

# Check label values
y_raw.value_counts()


# Standardise X
scaler = StandardScaler()
X_pre = scaler.fit_transform(X_raw)


# Label encoder for labels - transform classes 2, 3, 4 and 5 into 0 (no epilepsy)
le = LabelEncoder()
y_pre = le.fit_transform(y_raw)
y_pre = y_raw.replace([2,3,4,5], 0)


# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_pre, y_pre, test_size = 0.2, random_state = 1024)


#################### MODEL AND HYPERPARAMETER SELECTION ###########################


#Use GridSearchCV to identify the best hyperparameter to fhe the SVC model
param_grid = [{'C': [0.1, 1], 'kernel': ['linear'], 'class_weight': ['balanced',None]},
              {'C': [0.1, 1, 10, 100], 'kernel': ['rbf','sigmoid'], 'class_weight': ['balanced',None], 'gamma': ['scale', 'auto']}]

scoring = ['accuracy', 'balanced_accuracy','f1','precision','recall']

clf_svc = GridSearchCV(svm.SVC(), param_grid, scoring = scoring, cv=5, return_train_score = True, verbose = 50, n_jobs = -1,refit=False)


#Fit the training data into the model to obtain the best hyperparameters
clf_svc_fit = clf_svc.fit(X_train, y_train)


# Using the cv_results_ from GridSearchCV create a dataframe to find the best hyperparameter
df_grid = pd.DataFrame.from_dict(clf_svc_fit.cv_results_)


#################### VALIDATION OF SELECTED MODEL ###########################


#Create the model using the best ranked model assessed - Accuracy, F1, Precision, Recall and Fit_time
clf_candidate = svm.SVC(C=10, kernel = 'rbf', class_weight = 'balanced', gamma = 'auto')


# Obtain the learning curve of the selected model during fitting
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf_candidate, X_train, y_train, cv=5, n_jobs=4,
                                                                      return_times=True, random_state=0, scoring = 'accuracy')

# Calculate different metrics to plot learning curve                                                                  
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)


# Plot learning curve
fig1, ax1 = plt.subplots()

ax1.set_title("Learning Curves (SVM, RBF kernel, auto $\gamma$, Balanced class weight)")
ax1.grid()
ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation accuracy")
ax1.legend(loc="best")
plt.show()


# Plot fit_time vs score
fig3, ax3 = plt.subplots()
ax3.grid()
ax3.plot(fit_times_mean, test_scores_mean, 'o-')
ax3.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
ax3.set_xlabel("fit_times (s)")
ax3.set_ylabel("Accuracy")
ax3.set_title("Performance of the model")
plt.show()

#Fit training data to the selected model
clf_svc_fit = clf_candidate.fit(X_train, y_train)

#################### PREDICT VALUES USING SELECTED MODEL ###########################

#Predict values
y_pred = clf_svc_fit.predict(X_test)

#Check performance

class_report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame.from_dict(class_report).T

plot_confusion_matrix(clf_svc_fit, X_test, y_test, cmap='Greens')


#################### SAVE MODEL  ###########################

# Save trained models
path = 'trained_models/'
filename = 'SVC_' + str(date.today()).replace('-','') + '.sav'
pickle.dump(clf_svc_fit, open(path + filename, 'wb'))

