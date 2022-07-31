import os
import sklearn
import math
import logging
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split





class LinearRegression:

    def __init__(self):

        # log setting
        program = os.path.basename(__name__)
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')

        # Linear regression params
        self.n_Iterations = 100
        self.reg_param = 0.15


        

    def normalize(self,X):
        """
         Normalizes the features in the input X.This involves scaling the features for fast and efficient computation.

         Parameters
         ----------
         X : n dimensional array-like, shape (n_samples, n_features)
         Features(input varibale) to be normalized.

         Returns
         -------
         X_norm : n dimensional array-like, shape (n_samples, n_features)
         A normalized version of X.
         m : n dimensional array-like, shape (n_features,)
         The mean value.
         std : n dimensional array-like, shape (n_features,)
         The standard deviation.
         """
        #Gets the meaning of each column (feature) in the input 
        m = np.mean(X, axis = 0)  
        # ddof (Delta Degrees of Freedom)  
        std = np.std(X, axis= 0, ddof = 1)  # Standard deviation
        x_norm = (X - m)/std
        m, n = np.shape(x_norm)
        #stack a column of ones to be multiplyed in w0
        x_norm= np.hstack((np.ones((m,1)), x_norm))
        return x_norm,m,std



    def predict (self,x,w):

        """
         predict y values using hypothesis function 

         Parameters
         ----------
         x : n dimensional array-like, shape (n_samples, n_features)
         Features(input varibale).
         
         w: 1-D array , shape(1, n_features)
         initial weights for linear regression.

         Returns
         -------
         predicted values: 1-D array , shape (1,n_samples) 
         
         """ 
        self.logger.info("predict output values")
        self.weights=w
        predictions = np.dot(x,self.weights)
        return  predictions


    def compute_cost(self, x, y, w):
         """
          Compute the cost function of a particular omega for linear regression.

         Input Parameters
         ----------------
         predictions : 1D array-like ,shape (1,n_samples)
         n_samples= number of training examples
         n= number of features (including X_0 column of ones)
         y : 1D array of labels/target value for each traing example,shape (1,n_samples)

         Output Parameters
         -----------------
         J: Cost function (scalar)
         """
         self.logger.info("compute cost function")
         m=len(y)
         self.weights=w
         loss=(self.predict(x, self.weights)) - y 
         J= np.sum(loss ** 2) / (2 * m)
         return J


    def gradientDescent(self,x, y, w, alpha=None, n_Iterations=None):
        
        """
         Iterates on the input-output function to reach the most suitable weights .

        Input Parameters
        ----------------
         x : n dimensional array-like, shape (n_samples, n_features)
         Features(input varibale).
        
         w: 1-D array , shape(1, n_features)
         initial weights for linear regression.
         
        y : 1D array of labels/target value for each traing example,shape (1,n_samples)
        
        alpha: learning rate (float)
        
        
        n_Iterations : number of iteration for learning process 

        Output Parameters
        -----------------
        weights:1-D array , shape(1, n_features) 
        new weight for each feature to obtain a better labels prediction
        """
        self.logger.info("gradient descent function")

        if alpha is None:
             alpha = self.reg_param
        if n_Iterations is None:
            n_Iterations=self.n_Iterations

        m=len(y)
        xTrans = x.transpose()
        self.weights=w
        cost_history = np.zeros(n_Iterations)
        for i in range(n_Iterations):
            hypothesis =self.predict(x, self.weights)
            loss = hypothesis - y
            gradient = np.dot(xTrans, loss) / m
            self.weights =self.weights- alpha*gradient 
            cost_history[i] = self.compute_cost(x, y, self.weights)  
        print('First 5 values from cost_history =', cost_history[:5])
        print('Last 5 values from cost_history =', cost_history[-5 :])    
        return self.weights

    


    def fit(self,x,y):
        
        """
         fits the input data to extract the suitable function weights .

        Input Parameters
        ----------------
         x : n dimensional array-like, shape (n_samples, n_features)
         Features(input varibale).
        
         w: 1-D array , shape(1, n_features)
         initial weights for linear regression.
         
        y : 1D array of labels/target value for each traing example,shape (1,n_samples)
        
        Output Parameters
        -----------------
        weights:1-D array , shape(1, n_features) 
        new weight for each feature to obtain a better labels prediction
        """
        self.logger.info("fit matrices to get new weights")
        m, n = np.shape(x)
        w= np.zeros(n)
        self.weights=self.gradientDescent(x,y,w)
        return self.weights 


    def transform(self, x):
        """
         transforms the input data with the weights from the fit function .

        Input Parameters
        ----------------
         x : n dimensional array-like, shape (n_samples, n_features)
         Features(input varibale).
        Output Parameters
        -----------------
        y_transformed :  1D array of labels/target value for each traing example,shape (1,n_samples)
        new y predicted values after transformation the x values with new weights . 
        """
        
        self.logger.info("transform matrices with new weights")
        print(self.weights)
        y_transformed=np.dot(x,self.weights)
        return y_transformed 
    
    
    
    def Evaluate_performance(self,x,y):
        """
         Evaluates the performance of the linear regression model .

        Input Parameters
        ----------------
      
        y_transformed :  1D array of labels/target value for each traing example,shape (1,n_samples) after using new weights
        y : 1D array of labels true labels 
        """
        #Root Mean Squared Error (RMSE)
        MSE = np.square(np.subtract(x,y)).mean() 
        RMSE = math.sqrt(MSE)
        
        #squared residual error
        rss = np.sum(np.square(x-y))
        return RMSE,rss
        
        
        
        
        
    
    
    

def main():
    
    # set log level
    logging.root.setLevel(level=logging.INFO)
    lg=LinearRegression()
######################################################################## ~ Working on univariateData ~ ################################################################################
    #univariateData= pd.read_table('univariateData.dat' ,sep=",")
    #X=univariateData.iloc[:, 0] 
    #print(X.shape)
    #y=univariateData.iloc[:,1]  # output 
    #m = len(y)
    #plt.scatter(X,y, color='red',marker= '+')
    #plt.grid()
    #plt.rcParams["figure.figsize"] = (10,6)
    #plt.xlabel('X input values')
    #plt.ylabel('Y output values')
    #plt.title('Scatter plot of data')
    #plt.show()
    #x0= np.ones((m, 1))
    #X=np.array(X)
#
    #x1 = X.reshape(m, 1)
    #x=np.hstack(( x0 , x1))
    #
    ## split into train test sets
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    #w= np.zeros(2)
    #n_Iterations = 15000
    #alpha = 0.1
    ##Computeing the new weights using the gradient descent function
    #weights =  lg.gradientDescent( X_train, y_train, w, alpha, n_Iterations)
    ##print(weights)
    ##Computeing the new weights using fit function
    #fit_weights=lg.fit(X_train, y_train)
    ##print(fit_weights)
    #
    ##Testing the model 
    ##Transform test data with the new weights
    #y_transformed=lg.transform(X_test)
    #
    #
    #
    ## Check Evaluation performance
    #y_predicted=lg.predict(X_test,w)
    #RMSE,rss=lg.Evaluate_performance(y_predicted,y_test)
    #print('The values of RMSE error before gradient descent are',RMSE)
    #print('The values of RSS error before gradient descent are',rss)
    #RMSE2,rss2=lg.Evaluate_performance(y_transformed,y_test)
    #print('The values of RMSE error after gradient descent are',RMSE2)
    #print('The values of RSS error after gradient descent are',rss2)
    
    
    
    
    

######################################################################## ~ Working on multivariateData ~ ###################################################################################

    multivariateData= pd.read_table('multivariateData.dat' ,sep=",")
    X=multivariateData.iloc[:, 0:2] 
    y=multivariateData.iloc[:,2]  # output 
    m, n = np.shape(X)
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    x_train_norm,mean,std=lg.normalize(X_train)
    
    #Create an initial weights values 
    w= np.zeros(n+1)
    #number of iterations for gradient descent 
    n_Iterations = 100
    #learning rate
    alpha = 0.15
    #Computeing the new weights using the gradient descent function
    weights =  lg.gradientDescent(x_train_norm, y_train, w, alpha, n_Iterations)
    #Computeing the new weights using fit function
    #fit_weights=lg.fit(x_train_norm, y_train)
    
    
    #Testing the model 
    #normalize the testing data with the mean of the training data 
    x_test_norm = ((X_test - mean) / std)
    m, n = np.shape(x_test_norm)
    x_test_norm= np.hstack((np.ones((m,1)), x_test_norm))
    
    ##Transform test data with the new weights
    y_transformed=lg.transform(x_test_norm)
    
   # Check Evaluation performance
    y_predicted=lg.predict(x_test_norm,w)
    RMSE,rss=lg.Evaluate_performance(y_predicted,y_test)
    print(y_test-y_predicted)
    print('The values of RMSE error before gradient descent are',RMSE)
    print('The values of RSS error before gradient descent are',rss)
    RMSE2,rss2=lg.Evaluate_performance(y_transformed,y_test)
    print(y_test-y_transformed)
    print('The values of RMSE error after gradient descent are',RMSE2)
    print('The values of RSS error after gradient descent are',rss2)
    

if __name__=="__main__":
    main()