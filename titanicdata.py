import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
%matplotlib inline
import math

titanicdata = pd.read_csv("Titanic.csv")
titanicdata.head(20)

print(len(titanicdata.index))

#Analysing data
sbn.countplot(x= "Survived", data = titanic_data)
sbn.countplot(x= "Survived", hue = "Sex", data = titanic_data)
sbn.countplot(x= "Survived", hue = "Pclass", data = titanic_data)
titanicdata["Age"].plot.hist()
titanicdata["Fare"].plot.hist()


titanicdata.info()



#REmoving null values
titanicdata.isnull() # gives true in place of null values and false in place of not null values

titanicdata.isnull().sum()
sns.boxplot(x = "Pclass" , y = "Age", data = titanicdata)
#remove the coloumn with null values
titanicdata.drop("Cabin" axis = 1, inplace = True)


titanicdata.head(10)

titanicdata.dropna( inplace = True)
titanicdata.isnull()

titanicdata.isnull().sum()

sex = pd.get_dummies(titanicdata['Sex'], drop_first = True)
embark = pd.get_dummies(titanicdata['Embarked'], drop_first = True)
pclass = pd.get_dummies(titanicdata['Pclass'], drop_first = True)
titanicdata = pd.concat(titanicdata, sex, embark, pclass)
titanicdata.drop(['Sex', 'Embarked', 'PassengerId', 'Name', 'Ticket','Pclass'], axis = 1, inplace = True)
titanicdata.head()


#train dataset
x = titanicdata.drop("Survived", axis = 1)
y = titanicdata["Survived"]


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)


prediction = logmodel.predict(x_test)
from sklearn.metrics  import classification_report
classification_report(y_test, predictions)
from sklearn.metrics  import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
