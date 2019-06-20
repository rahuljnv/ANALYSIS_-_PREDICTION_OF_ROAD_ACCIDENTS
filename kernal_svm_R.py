# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:46:33 2019

@author: rahul
"""

#importing the lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
#reading the dataset in csv
acc1 = pd.read_csv('accidents.csv')
#describing the mean,count,std,min,max
acc1.describe()
#desplaying first 5 tuples
acc1.head()
#plotting graph b/w severity(D.V) Vs count
import seaborn as sns
sns.countplot(y = "severity" , data = acc1 )
plt.tight_layout()

#creating Dataframe with important features
pd.DataFrame( {"count": acc1["severity"].value_counts().values } , index = acc1["severity"].value_counts().index )
acc1= acc1.loc[acc1["severity"] >  1].loc[acc1["severity"] < 4]
acc1["month"] = acc1["time"].apply(lambda x:int(x[:2]))
acc1["day"] = acc1["time"].apply(lambda x:int(x[3:5]))
acc1["year"] = acc1["time"].apply(lambda x:int(x[6:8]))
acc1["hour"] =  acc1["time"].apply(lambda x: int(x[9:11]) if str(x)[15] == 'A' else 12 + int(x[9:11])  )
acc1["lon"] = acc1["lon"].apply(lambda x:abs(x))
#so that multinomialNB works (only with positive features)
#creating the date at the datetime format (easier to deal with)
acc1[ "date" ]= acc1[["month" , "day" ,"year"]].apply(lambda x:pd.datetime(month = x['month'] , day = x['day']  , year = 2000+x["year"]), axis = 1)
acc1["weekday"] =  acc1["date"].apply(lambda x:x.weekday())

#severity by hours
severity_by_hour = pd.crosstab(index = acc1["hour"] , columns = acc1["severity"] )
severity_by_hour = pd.DataFrame(severity_by_hour.values)
severity_by_hour["ratio"] = severity_by_hour.apply(lambda x:x[0]/float(x[1]) , axis = 1)
severity_by_hour.sort_values(by = "ratio")

# correlation heatmap
acc1_corr = acc1[["lat" , "lon" , "month" , "year" , "hour" , "weekday" , "severity"]]
correlation = acc1_corr.corr()
sns.heatmap(correlation)
plt.tight_layout()

# shifting to 0-1 values instead of 2-3 for DV & IN IV making the arrangement of coloumn
X = acc1[["month" , "hour" , "year", "weekday" ,"lon" , "lat"]]
y = acc1["severity"].apply(lambda x:x-2) 

#splitting dataset into training and test data set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

#give the dimeensionality
X_train.shape
X_test.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Predicting the Train set results
y_pred1 = classifier.predict(X_train)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm1 = confusion_matrix(y_train, y_pred1)

#For accuracy On the basis of test set
"""(5043+3765)/11960(X_test)
Out[7]: 0.7364548494983277
"""


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
accuracies.mean()
accuracies.std()
"""
accuracies.mean()
Out[4]: 0.742998352553542

accuracies.std()
Out[5]: 0.0004942339373970595

"""