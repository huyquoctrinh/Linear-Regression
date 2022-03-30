from utils import *
from LinearRegression import LinearRegression

linear = LinearRegression(graphfile)
X,y = create_data(csv_file,X_name,y_name)
print(X.shape)

print("Starting update weight ...")
linear.proof(X,y)

print("Weight after update:")
linear.get_weight()

print("Graph for the linear:")
linear.plot_graph()

sample = readfile(inputfile)
print(sample)

print("Predicting phase:")
out = []
for i in range(len(sample)):
	print("Test {}:".format(i),linear.predict(sample[i]))
	out.append(linear.predict(sample[i]))
writefile(outfile,out)