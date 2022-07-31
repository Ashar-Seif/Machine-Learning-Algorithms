import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
 
 



class LinearSVM:
    def __init__(self, C=1.0):

        self.support_vectors = None
        #ٌ C: Regularization Constant
        self.C = C
        #ٌ w: the weight vector 
        self.w= None
        #ٌ b: the bias  
        self.b = 0
        # n is the number of data points
        self.n = 0
        # d is the number of dimensions
        self.d = 0

        
    def hypothesis(self, X):
      """
        Calculate the hyperplane equation 

        Parameters
        ----------
         X: n dimensional array-like, shape (n_samples, n_features)

         Returns
         -------
         hypothesis: n dimensional array-like which represent the hyperplane.
      """
      hypothesis= X.dot(self.w) + self.b
      return hypothesis

    def margin(self, X, y):
      """
        Calculate the margin equation 

        Parameters
        ----------
         X: n dimensional array-like, shape (n_samples, n_features)
         y: 1-D array , Include labels values (-1 or 1)

         Returns
         -------
         margin : n dimensional array-like which represent the margin.
      """
      margin=y * self.hypothesis(X)
      return margin
 
    def cost_function(self, margin):
      """
        Calculate the cost function which need to be minimized by gradient descent 

        Parameters
        ----------
         margin : n dimensional array-like, includes the data points located in the margin. 

         Returns
         -------
         loss : a scalar represents the cost of using specific weights . 
      """
      loss=(1 / 2) * self.w.dot(self.w) + self.C * np.sum(np.maximum(0, 1 - margin))
      return loss
 


    def fit(self, X, y, alpha=1e-3, n_iteraton=1000):
        """
        Train model with training data to get the proper weights.

        Parameters
        ----------
         X: n dimensional array-like, shape (n_samples, n_features) represents the training data .
         y:  y: 1-D array , Include labels values (-1 or 1).
         alpha: learning rate.
         n_iteraton: number of iterations for gradient descent.
        """
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.w= np.random.randn(self.d)
       
        loss_array = []
        for i in range(n_iteraton):
          margin = self.margin(X, y)
          loss = self.cost_function(margin)
          loss_array.append(loss)
          misclassified_points = np.where(margin < 1)[0]
          gradient = self.w - self.C * y[misclassified_points].dot(X[misclassified_points])
          self.w = self.w - alpha *gradient
 
          Regularization = - self.C * np.sum(y[misclassified_points])
          self.b = self.b -alpha * Regularization
 
        self.support_vectors = np.where(self.margin(X, y) )
        print("The last loss values",loss_array[-10:-1])
    

    def predict(self, X):
      """
        Predict labels for test data 

        Parameters
        ----------
         X: n dimensional array-like, shape (n_samples, n_features) represents the testing data.
        Returns
        -------
         predicted : list contains predicted values for test data. 
      """
      predicted=np.sign(self.hypothesis(X))
      return predicted
 
    def accuracy_metric(self,y, predictions):
      """
        Calculate the accuracy of the system.

        Parameters
        ----------
        y: 1-D array contains the actual labels.
        predictions: 1-D array contains the predicted labels.
        Returns
        -------
        accuracy : scalar.  
      """
      true_labels=0
      for i in range(len(y)):
           if(y[i]==predictions[i]):
               true_labels+=1
      accuracy= true_labels/len(y)
      return accuracy*100 

    def plot_decision_boundary(self,x,y):
        plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap=plt.cm.Paired, alpha=.7)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
 
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.hypothesis(xy).reshape(XX.shape)
 
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])
 
        plt.show()

 
if __name__ == '__main__':

    iris = sns.load_dataset("iris")
    #print(iris["species"].unique())
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(iris["species"])
    X = iris.drop(["species"], axis=1)
    X=X.iloc[:,2:4]
   
   
   
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X=np.array(X)

    # One Vs One Classification
    ## setosa vs versicolor
    class2=list(np.where(Y==2)[0])
    x1=np.delete(X,class2,axis=0)
    y1=np.delete(Y,class2)
    y1[y1 == 0] = -1

   
    ## versicolor vs virginica
    class0=list(np.where(Y==0)[0])
    x2=np.delete(X,class0,axis=0)
    y2=np.delete(Y,class0)
    y2[y2 == 1] = -1
    y2[y2 == 2] = 1


    ## setosa vs virginica
    class1=list(np.where(Y==1)[0])
    x3=np.delete(X,class1,axis=0)
    y3=np.delete(Y,class1)
    y3[y3 == 0] = -1
    y3[y3 == 2] = 1
    
    X_list=[x1,x2,x3]
    Y_list=[y1,y2,y3]
    Tasks=["setosa vs versicolor","versicolor vs virginica","setosa vs virginica"]


    for exp in range(len(X_list)):
    
    

      X_train, X_test, y_train, y_test = train_test_split(X_list[exp], Y_list[exp], test_size=0.4,random_state=42)    
    
      # train the model
      clf = LinearSVM(C=15.0)
      clf.fit( X_train, y_train)
      y_pred=clf.predict(X_test)
      print(Tasks[exp]+" accuracy :", clf.accuracy_metric(y_test,y_pred))
      clf.plot_decision_boundary(X_train, y_train)