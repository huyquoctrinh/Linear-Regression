from utils import *
class LinearRegression:
  def __init__(self,graphfile):
    self.w0 = 0
    self.w1 = 0
    self.y0 = 0
    self.X =[]
    self.y = []
    self.x0 = np.linspace(0,800,2)
    self.graph_file = graphfile
  def proof(self,X,y):
    self.X = X
    self.y = y
    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one,X),axis = 1)
    a = np.dot(Xbar.T,Xbar)
    b = np.dot(Xbar.T,y)
    w = np.dot(np.linalg.pinv(a),b)
    self.w0 = w[0][0]
    self.w1 = w[1][0]
    print("w0 update to", self.w0)
    print("w1 update to",self.w1)
    self.y0 = self.w0 + self.w1*self.x0
    # print(y0,"=",w0,"+",w1,"*",x0)
    print("Finish update weight")
  def get_weight(self):
    print("w1:", self.w1)
    print("w0:",self.w0)
    return (self.w1,self.w0)
  def plot_graph(self):
    if self.X == [] or self.y == []:
      print("fail to plot, please add data")
      return False
    else:
      plt.plot(self.X.T, self.y.T, 'ro') 
      plt.plot(self.x0, self.y0)             
      plt.xlabel('target temperature')
      plt.ylabel('prcp')
      # plt.show()
      plt.savefig(self.graph_file)
  def predict(self,x):
    return np.round(self.w1*x + self.w0,5)
  def loss(self,y_true):
    res = 0
    m = len(y_true)
    for i in range(len(y_true)):
      res+= (y_true[i] - self.predict(y))**2
    return float(res/m)