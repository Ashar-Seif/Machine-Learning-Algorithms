import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict


############################################################ ~ Reading and splitting data ~ ###################################################################################################
# Fetching mnist dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#print(mnist.keys())

X, Y = mnist["data"], mnist["target"]
#print(X.shape)
#print(Y.shape)

# Splitting data into train (60,000 images) and test data (10,000 images)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]

# Some learning algorithms are sensitive to the order of the training instances, and they perform poorly if they get many similar instances in a row.Shuffling the dataset ensures that this won’t happen
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]



################################ ~ Apply a binary model that can classify one digit, say 3 and determine if this digit is 3 or not ~ ###################################################################################################

# Training a Binary Classifier
y_train_3 = (y_train == "3") # True for all 3s, False for all other digits.
y_test_3 = (y_test == "3")

# The SGD classifier has the advantage of being capable of handling very large datasets efficiently
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_3)

# Get list index for a 3 in Y list 
three_index=np.where(Y=="3")[0][0]
#print(three_index)   #index=7

# Choosing a 3 number from the X list and plot it 
Random_digit=X[three_index]
Random_digit_image = Random_digit.reshape(28, 28)
plt.imshow(Random_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.savefig("DigitThree.png")
#plt.show()

# Making Sure it is the same at the Y list 
#print(Y[7])       # 3 

# Predict the 3 number using prediction function 
print(sgd_clf.predict([Random_digit]))      # True 


############################################################ ~ Get the confusion matrix ~ ###################################################################################################

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_3, cv=3)

#Implement the confusion matrix 
conf_matrix=confusion_matrix(y_train_3, y_train_pred)


############################################################ ~ Multiclass Classification~ ###################################################################################################

sgd_clf.fit(X_train, y_train)
sgd_clf.predict(X_test)

############################################################ ~ Report Requirements ~ ########################################################################################################

# 1.Show 5 examples from the used mnist dataset.
# The five examples : Y[0] , Y[150] ,print(Y[777]) , Y[1000] , Y[40000]

#print(Y[0])        #5
#print(Y[150])      #4
#print(Y[1000])     #0
#print(Y[777])      #8
#print(Y[40000])    #7

examples_indicies=[0,150,777,1000,40000]

for i in range(5):
    
  example_digit=X[examples_indicies[i]].reshape(28, 28)
  plt.imshow(example_digit)
  plt.savefig("example_digit"+str(i)+".png",facecolor ="white")
  #plt.show()

# 3.Interpret the output of the confusion matrix
# Getting Performance measures from the confusion matrix 

TN=conf_matrix[0][0]   # True negative
FP=conf_matrix[0][1]   # False positive
FN=conf_matrix[1][0]   # False negative 
TP=conf_matrix[1][1]   # True positive 

accuracy =  (TP+TN) /(TP+FP+TN+FN)
precision = TP/(TP+FP)
sensitivity = TP/(TP+FN) 
specifity = TN/(TN+FP)


#Plotting confusion matrix
fig, ax = plt.subplots(figsize=(5,5))
ax.matshow(conf_matrix)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j])
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig("conf_matrix.png",facecolor ="white")
plt.show()

