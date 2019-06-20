# -*- coding: utf-8 -*-
"""
Created on Fri May  3 01:55:48 2019

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

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


#implimentation of Random Forest &ravel is used to place in 1-D
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0)
random_forest.fit(X_train,y_train.values.ravel())

#Prediction BY X_test
y_pred = random_forest.predict(X_test)

#prediction By X_train
y_pred1 = random_forest.predict(X_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#checking for overfitting
cm1 = confusion_matrix(y_train, y_pred1)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, random_forest.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random_Forest (Training set)')
plt.xlabel('PC1')#PC1,PC2 are the cummulative principale component
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, random_forest.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random_Forest (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()