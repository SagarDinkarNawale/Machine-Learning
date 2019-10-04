# first scikit learn progarm of ML using supervised regression.
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import  datasets,linear_model
from sklearn.metrics import  mean_squared_error

diabetes=datasets.load_diabetes()


# -------------------------------------------------------------------------------------------


# # print(diabetes.keys())#gives foll output
# # # --dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# # print(diabetes.data) #print the data keys all value
# # print(diabetes.DESCR)# gives compleete attribute details
# diabetes_x=diabetes.data[:,np.newaxis,2]#makes numpy array of array of 2nd element
# # print(diabetes_x)# simply print the that numpy list array

# diabetes_x_train=diabetes_x[:-30]#takes the last 30 element using slicing
# diabetes_x_test=diabetes_x[-30:]#takes the first 30 to tesing
# diabetes_y_train = diabetes.target[:-30]
# diabetes_y_test = diabetes.target[-30:]
# model = linear_model.LinearRegression()

# model.fit(diabetes_x_train, diabetes_y_train)
# diabetes_y_predicted = model.predict(diabetes_x_test)
# print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

# print("Weights: ", model.coef_)#tan theta 
# print("Intercept: ", model.intercept_)#y that is slope
# # ////// for colouring purpose
# n_points = 4
# xs = ys = zs = np.full(n_points, np.nan)
# colors = list(np.full(2, 'k'))
# # for cououring purpose using numpy
# plt.scatter(diabetes_x_test, diabetes_y_test,c=colors)#plot the all 30 ppint
# plt.plot(diabetes_x_test, diabetes_y_predicted)# show the pridicted line

# plt.show()
# -------------------------------------------------------------------------------------------
# Mean squared error is:  3035.0601152912695
# Weights:  [941.43097333]# here we takes only one wates using sliceing 2 
# Intercept:  153.39713623331698 
# in case we dont take slicing the we canot plot graph and line see as follow and in this
 #  case our all features means weights all morethan one seeee...


diabetes_x=diabetes.data#take numpy array of array 
diabetes_x_train=diabetes_x[:-30]#takes the last 30 element using slicing
diabetes_x_test=diabetes_x[-30:]#takes the first 30 to tesing
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]
model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_x_test)
print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights: ", model.coef_)#tan theta 
print("Intercept: ", model.intercept_)#y that is slope

# plt.scatter(diabetes_x_test, diabetes_y_test,c=colors)#plot the all 30 ppint
# plt.plot(diabetes_x_test, diabetes_y_predicted)# show the pridicted line
# plt.show()
# output
# Mean squared error is:  1826.5364191345427
# Weights:  [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#   458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
# Intercept:  153.05827988224112
