import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
X_name = ["MLY-TMIN-NORMAL","MLY-TMIN-NORMAL"]
y_name = ["MLY-PRCP-NORMAL"]
csv_file = "./asset/data.csv"
inputfile ="./asset/test.txt"
outfile = "./asset/out.txt"
graphfile = "./asset/graph_out.png"
def create_data(csv_file,X_name,y_name):
	X = []
	y = []
	df = pd.read_csv(csv_file)
	for name in X_name:
		tmp = []
		for i in range(len(df)):
			tmp.append(df[name][i])
		X.append(tmp)
	for name in y_name:
		tmp = []
		for i in range(len(df)):
			tmp.append(df[name][i])
		y.append(tmp)
	return np.array(X).T,np.array(y).T

def readfile(inputfile):
	list_of_lists = []
	with open(inputfile) as f:
	    for line in f:
	        inner_list = [int(elt.strip()) for elt in line.split()]
	        list_of_lists.append(inner_list)
	return list_of_lists

def writefile(outfile, output):
	out = open(outfile,"w") 
	for i in range(len(output)):
		out.write("Prediction of test {}: ".format(i+1)+ str(output[i]) +"\n")
	print("Finished")