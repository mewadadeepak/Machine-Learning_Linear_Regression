import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
#diabetes_X = diabetes.data


diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes_X[:-30]
diabetes_y_test = diabetes_X[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train , diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean square error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)


plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_y_test, diabetes_y_predicted)
plt.show()





'''
#ouput
#including few datsets

Mean square error is:  1.8367444043274482e-33
Weights:  [[1.]]
Intercept:  [-9.48676901e-20]          




#ouput
#including all datsets:

Weights:  [[ 1.00000000e+00 -1.49533389e-16 -8.21071346e-17 -1.86251764e-16
  -2.32330920e-17 -2.13214501e-16 -1.09810752e-16  2.51569401e-18
   1.02037097e-16  5.18553538e-17]
 [ 2.43535539e-16  1.00000000e+00  2.77555756e-16 -4.44089210e-16
  -3.33066907e-16  7.77156117e-16  5.55111512e-16  5.55111512e-17
   2.22044605e-16 -5.55111512e-17]
 [ 9.12599016e-17 -2.22044605e-16  1.00000000e+00  4.16333634e-17
  -9.71445147e-17  1.24900090e-16 -2.77555756e-17  3.88578059e-16
  -3.46944695e-16  1.38777878e-16]
 [-4.93942686e-16  1.66533454e-16  2.49800181e-16  1.00000000e+00
  -6.10622664e-16  4.16333634e-16  1.14491749e-16 -3.33066907e-16
   2.22044605e-16  3.60822483e-16]
 [-2.07986048e-16  3.33066907e-16 -5.27355937e-16 -5.55111512e-17
   1.00000000e+00  5.55111512e-16 -3.46944695e-18 -3.19189120e-16
   1.66533454e-16  2.77555756e-16]
 [-7.54180647e-17  4.99600361e-16 -1.80411242e-16 -6.93889390e-17
   5.68989300e-16  1.00000000e+00 -1.97758476e-16  1.66533454e-16
  -1.66533454e-16  1.38777878e-17]
 [-4.86020984e-16  8.32667268e-17  1.38777878e-17  1.28369537e-16
   8.91647867e-16 -9.08995101e-16  1.00000000e+00 -2.77555756e-17
  -2.08166817e-16  1.04083409e-16]
 [ 1.12915272e-16  2.77555756e-16  5.55111512e-17 -2.49800181e-16
   6.10622664e-16 -6.52256027e-16 -5.55111512e-16  1.00000000e+00
   9.71445147e-17 -2.35922393e-16]
 [ 3.37009474e-16  1.66533454e-16  9.71445147e-17 -3.33066907e-16
  -3.74700271e-16  1.94289029e-16  1.07552856e-16 -1.24900090e-16
   1.00000000e+00  2.77555756e-16]
 [-5.19848472e-17  0.00000000e+00 -8.32667268e-17  5.55111512e-17
   2.77555756e-17  1.24900090e-16  3.67761377e-16  1.24900090e-16
   1.66533454e-16  1.00000000e+00]]
Intercept:  [-3.52365706e-19  1.02999206e-18  4.33680869e-19 -1.08420217e-19
  1.51788304e-18 -2.16840434e-19  0.00000000e+00 -7.04731412e-19
 -5.42101086e-20  8.67361738e-19]

Process finished with exit code 0                                                                                   '''














