import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =====================================================
# Load Data
# =====================================================
my_data = pd.read_csv("data/satislar.csv")
print(my_data.describe())

# Get Sales and Months Columns
sales = my_data[['Satislar']]
months = my_data [['Aylar']]

print(sales)
print(months)

# Split values as test and training with %33-%66 Ratio
x_train, x_test, y_train, y_test = train_test_split(months,sales, test_size=0.33)

#Eğer standart scaler kullanırsak tüm değerleri aynı mertebeden görebiliriz. Zorunlu değil.

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# Scaled Values
print("X_train = ", X_train)
print("X test = ", X_test)
print("Y_train = ", Y_train)
print("Y_test = ", Y_test)

# Fit for Scaled. Fit is meaning to Apply Linear Regression ML Method
lg = LinearRegression()
lg.fit(X_train,Y_train)

# Fit for Non Scaled.
lg2 = LinearRegression()
lg2.fit(x_train,y_train)

# Non Scaled X_test'e karşılık gelen Y_test
my_predict = lg2.predict(x_test)
print(my_predict)

print(y_test)

print(type(y_test))

y_test_array = y_test.to_numpy()

print(y_test_array)

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(my_predict, 
       y_test_array,color="red")

ax.scatter(y_test_array,my_predict,color='blue')

plt.title("Non Scaled ML Comparision Linear Reg.")

plt.xlabel("Trained Values on sales (predict)")

plt.ylabel("Real values on sales")

plt.grid()

plt.legend(["Trained values (my_predict)","Real Values (y_test)"], loc ="lower right")

plt.show()

# Scaled X_test'e karşılık gelen Y_test
my_predict2 = lg.predict(X_test)
print(my_predict2)

print(Y_test)

#Y_test_array = Y_test.to_numpy()

#print(Y_test_array)

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(my_predict2, 
       Y_test,color="red")

ax.scatter(Y_test,my_predict2,color='blue')

plt.title("Scaleli ML Karşılaştırma Linear Reg.")

plt.xlabel("Eğitilmiş değerler satış (tahmin)")

plt.ylabel("Gerçek Değerler Satış")

plt.grid()

plt.legend(["Eğitilmiş Değerler (my_predict)","Gerçek değerler (y_test)"], loc ="lower right")

plt.show()

# Scaled X_test'e karşılık gelen Y_test
my_predict2 = lg.predict(X_test)
print(my_predict2)

print(Y_test)

#Y_test_array = Y_test.to_numpy()

#print(Y_test_array)

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(my_predict2, 
       Y_test,color="red")

ax.scatter(Y_test,my_predict2,color='blue')

plt.title("Scaleli ML Karşılaştırma Linear Reg.")

plt.xlabel("Eğitilmiş değerler satış (tahmin)")

plt.ylabel("Gerçek Değerler Satış")

plt.grid()

plt.legend(["Eğitilmiş Değerler (my_predict)","Gerçek değerler (y_test)"], loc ="lower right")

plt.show()

# Sort ile sıralama. Linear reg için uygun grafik. Scalesiz
x_train_sorted = x_train.sort_index()
y_train_sorted = y_train.sort_index()

plt.plot(x_train_sorted,y_train_sorted)
plt.plot(x_test,my_predict)

plt.title("Satış vs. Aylar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")

# Sort ile sıralama. Linear reg için uygun grafik. Scaleli
#X_train_sorted = X_train.sort_index()
#Y_train_sorted = Y_train.sort_index()

#plt.plot(X_train,Y_train)
#plt.plot(X_test,my_predict2)

#plt.savefig("mygraph.png")