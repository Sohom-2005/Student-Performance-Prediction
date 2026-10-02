import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#Loading Dataset

df=pd.read_csv('students_performance_dataset.csv')

print(df.head())

print(df.shape)

print(df.info())

#Missing Values

print(df.isnull().sum())

df=df.dropna(subset=['Study_Hours_per_Week','Final_Score'])

#Feature and Target

X=df[['Study_Hours_per_Week']]
y=df['Final_Score']

#Train-Test Split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Training Model

model=LinearRegression()
model.fit(X_train,y_train)

#Prediction

pred_score=model.predict(X_test)

#Evaluation

mae=mean_absolute_error(y_test,pred_score)
mse=mean_squared_error(y_test,pred_score)
rmse=np.sqrt(mse)
r2=r2_score(y_test,pred_score)

print('MAE=',round(mae,2),'MSE=',round(mse,2),'RMSE=',round(rmse,2),'R2 Score=',round(r2,2))

#Plot 1 (Histogram)

plt.figure(figsize=(12,8))
plt.hist(df['Final_Score'],bins=30,color='orange',edgecolor='black')
plt.title('Distribution of Final Exam Scores',fontsize=12)
plt.xlabel('Final Exam Score',fontsize=8)
plt.ylabel('Number of Students',fontsize=8)
plt.grid(True)
plt.tight_layout()

plt.show()

#Plot 2 (Scatter Plot with Regression Line)

plt.figure(figsize=(12,8))
plt.scatter(X_test,y_test,color='blue',label='Actual Scores')
plt.plot(X_test,pred_score,color='red',label='Predicted Scores')
plt.title('Model Prediction vs Actual Score',fontsize=12)
plt.xlabel('Study Hours per Week',fontsize=8)
plt.ylabel('Final Marks',fontsize=8)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()



