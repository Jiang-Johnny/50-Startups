#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

#Reading the data 

df = pd.read_csv('/content/drive/MyDrive/Datasets/50_Startups.csv')
df.head()

#Using heatmap to see the correlations between each attribute

sns.heatmap(df.corr(),annot=True)
plt.show()

#Running a linear regression machine learning algorithm for accuracy

df2=df[['R&D Spend','Administration','Marketing Spend', 'Profit']]

predict = 'Profit'

x= np.array(df2.drop([predict],1))
y=np.array(df[predict])

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

lr = LinearRegression()

lr.fit(x_train,y_train)
acc = lr.score(x_test,y_test)
print(acc)

#Print out the Linear Regression coefficients and the intercept

print("Coefficient:", lr.coef_)
print("Intercept", lr.intercept_)

#Using the machine learning algorithm to making predicitions.

predictions = lr.predict(x_test)

for i in range(len(predictions)):
  print(predictions[i],x_test[i],y_test[i])


