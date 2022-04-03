from utils import *
from LinearRegression import LinearRegression

linear = LinearRegression(graphfile)
X,y = create_data(csv_file,X_name,y_name)
print(X.shape)

x_train,y_train,x_test,y_test = train_test_split(X,y)
print("Starting update weight ...")
linear.fit(X,y)

print("Weight after update:")
linear.get_weight()

# print("Graph for the linear:")
# # linear.plot_graph()

sample = readfile(inputfile)
print(sample)

print("Predicting phase:")
out = []
for i in range(len(sample)):
	print("Test {}:".format(i),linear.predict_one_value(sample[i]))
	out.append(linear.predict_one_value(sample[i]))
writefile(outfile,out)

y_predict = linear.predict(x_test)
print(y_predict)
print("Testing model:")
mae = linear.mae(y_predict,y_test)
mse = linear.mse(y_predict,y_test)
print("Mae: ",mae)
print("MSE: ",mse)
f = open(evalfile,"w")
f.write("Mae score: " + str(mae) +"\n"
		+"Mse score: " + str(mse)+"\n")
