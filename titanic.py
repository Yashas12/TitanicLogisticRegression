import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import seaborn as sb

data = pd.read_csv("C:/Users/admin/Downloads/train.csv")
print(data)
#sb.heatmap(data.isnull())
#pp.show()


def impute_age(cols):
    age=cols[0]
    pclass=cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)
#sb.heatmap(data.isnull())
#pp.show()

data.drop('Cabin',axis=1,inplace=True)
#sb.heatmap(data.isnull())
#pp.show()

data.dropna(inplace=True)
#sb.heatmap(data.isnull())
#pp.show()

sex = pd.get_dummies(data['Sex'],drop_first=True)
embark = pd.get_dummies(data['Embarked'],drop_first=True)

data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
data=pd.concat([data,sex,embark],axis=1)

from sklearn.model_selection import train_test_split
x=data.drop('Survived',axis=1)
y=data['Survived']

xtrain,ytrain,xtest,ytest = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(xtrain,ytrain)

ypred = logreg.predict(xtest)

import sklearn.metrics
cnf = metrics.confusion_matrix(ytest,ypred)
print(cnf)


