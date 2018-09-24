import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

# import data
X, y, cf= make_regression(n_samples=1000, n_features=100, noise=0, coef=True, random_state=42)
np.savetxt("make_regressionX.csv", X, delimiter=",")
np.savetxt("make_regressionY.csv", y, delimiter=",")
df = pd.DataFrame(X,y) # first col as Y and remaining column as X
print (df.head())

# EDA
col=[0, 14, 17, 26, 42, 58, 60, 84, 95]
sns.pairplot(df.loc[:, col])
plt.tight_layout()
plt.show()

cm = np.corrcoef(df.loc[:, [0, 14, 17, 26, 42, 58, 60, 84, 95]])
print (cm)
#hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15})
                 #yticklabels=cols,
                 #xticklabels=cols)
#plt.show()




# split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='o', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-600, xmax=600, color='black', lw=1)
plt.ylim(-0.002, 0.002)
plt.title("Residual plot")
plt.show()
print('MSE train: %.3f, MSE test: %.3f' %
      (mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, R^2 test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

# linear regression coefficient
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print (w)

# Ridge Regression
alpha = np.arange(0, 10)
MSE_train = np.empty(len(alpha))
MSE_test = np.empty(len(alpha))
R2_train = np.empty(len(alpha))
R2_test = np.empty(len(alpha))

for i,k in enumerate(alpha):
    ridgereg = Ridge(alpha = k)
    ridgereg.fit(X_train, y_train)
    y_train_pred = ridgereg.predict(X_train)
    y_test_pred = ridgereg.predict(X_test)
    MSE_train[i] = mean_squared_error(y_train, y_train_pred)
    MSE_test[i] = mean_squared_error(y_test, y_test_pred)
    R2_train[i] = r2_score(y_train, y_train_pred)
    R2_test[i] = r2_score(y_test, y_test_pred)
plt.plot(alpha, MSE_train, label='Ridge Training MSE')
plt.plot(alpha, MSE_test, label='Ridge Testing MSE')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('Mean Squared Error')
plt.title("Mean Squared Error for Ridge regression")
plt.show()
plt.plot(alpha, R2_train, label='Ridge Training R2')
plt.plot(alpha, R2_test, label='Ridge Test R2')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('R-Squared')
plt.title("R-Squared for Ridge regression")
plt.show()



# Lasso Regression
alpha = np.arange(0, 10)
MSE_train = np.empty(len(alpha))
MSE_test = np.empty(len(alpha))
R2_train = np.empty(len(alpha))
R2_test = np.empty(len(alpha))

for i,k in enumerate(alpha):
    lassoreg = Lasso(alpha = k)
    lassoreg.fit(X_train, y_train)
    y_train_pred = lassoreg.predict(X_train)
    y_test_pred = lassoreg.predict(X_test)
    MSE_train[i] = mean_squared_error(y_train, y_train_pred)
    MSE_test[i] = mean_squared_error(y_test, y_test_pred)
    R2_train[i] = r2_score(y_train, y_train_pred)
    R2_test[i] = r2_score(y_test, y_test_pred)
plt.plot(alpha, MSE_train, label='Lasso Training MSE')
plt.plot(alpha, MSE_test, label='Lasso Testing MSE')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('Mean Squared Error')
plt.title("Mean Squared Error for Lasso regression")
plt.show()
plt.plot(alpha, R2_train, label='Lasso Training R2')
plt.plot(alpha, R2_test, label='Lasso Test R2')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('R-Squared')
plt.title("R-Squared for Lasso regression")
plt.show()


# ElasticNet Regression
ratio = np.arange(0, 1, 0.1)
MSE_train = np.empty(len(ratio))
MSE_test = np.empty(len(ratio))
R2_train = np.empty(len(ratio))
R2_test = np.empty(len(ratio))

for i,k in enumerate(ratio):
    elasticreg = ElasticNet(alpha = 1, l1_ratio=k)
    elasticreg.fit(X_train, y_train)
    y_train_pred = elasticreg.predict(X_train)
    y_test_pred = elasticreg.predict(X_test)
    MSE_train[i] = mean_squared_error(y_train, y_train_pred)
    MSE_test[i] = mean_squared_error(y_test, y_test_pred)
    R2_train[i] = r2_score(y_train, y_train_pred)
    R2_test[i] = r2_score(y_test, y_test_pred)
plt.plot(ratio, MSE_train, label='ElasticNet Training MSE')
plt.plot(ratio, MSE_test, label='ElasticNet Testing MSE')
plt.legend()
plt.xlabel('ratio')
plt.ylabel('Mean Squared Error')
plt.title("Mean Squared Error for ElasticNet regression")
plt.show()
plt.plot(ratio, R2_train, label='ElasticNet Training R2')
plt.plot(ratio, R2_test, label='ElasticNet Test R2')
plt.legend()
plt.xlabel('ratio')
plt.ylabel('R-Squared')
plt.title("R-Squared for ElasticNet regression")
plt.show()


print("My name is Yuxin Sun")
print("My NetID is: yuxins5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
