import csv
import pandas as pd
import numpy as np




def train_test_split(data,percentage):
    """
    Split data into train and test.

    Parameters
    ----------
         x: n dimensional array-like, shape (n_samples, n_features)
         y: 1D-array ,shape(n_samples,) presents the values of classes (labels)
         percentage:The ratio between the train data znd the total data 

         Returns
         -------
         x_train : n dimensional array-like, shape (n_samples, n_features) selected for training .
         x_test : n dimensional array-like, shape (n_samples, n_features)  selected for testing .
         y_train : 1D-array ,shape(n_samples,) presents the values of classes (labels) selected for training .
         y_test : 1D-array ,shape(n_samples,) presents the values of classes (labels) selected for testing .
    """
    length=data.shape[0]
    percent=int(length*percentage)
    train_data=data.iloc[:percent+1,:]
    test_data=data.iloc[percent+1:,:]
    return  train_data, test_data


def best_split(data,split_method):

    """
    Get the best root node to split data on it 

    Parameters
    ----------
         data: n dimensional array-like, shape (n_samples, n_features)

         Returns
         -------
         index : The index of the root 
         value : The value at which we split the node and put the spliiting condition on it .
         groups : The splitted feature values groups after comparing with the splitting condition.
    """
    data=np.array(data)
    class_values = list(set(row[-1] for row in data))     
    max_gini=0.5
    max_entropy=0.3
    splitted_groups=None
    features_length=len(data[0])-1
    for feature in range(features_length):
        uniqueValues = np.unique(data[:,feature])
        if len(uniqueValues)>9:
            descritized_data=data[:,feature]
            sorted_feature_values=np.sort(data[:,feature])
            number_of_categories=int(len(uniqueValues)/5)
            feature_value_groups=[ sorted_feature_values[i:i + number_of_categories] for i in range(0, len(sorted_feature_values), number_of_categories)]
            for category in range(len(feature_value_groups)):                 
                for i in range(len(descritized_data)):
                    if descritized_data[i] in feature_value_groups[category]:
                        descritized_data[i]=category    
            data[:,feature]= descritized_data 

        for row in data:
            splitted_groups=feature_split(feature,row[feature],data)
            if split_method == "gini_index":
               gini=gini_index(splitted_groups,class_values)
               if(gini<max_gini):
                   feature_index=feature
                   feature_value=row[feature]
                   max_gini=gini
                   feature_groups=splitted_groups
            elif split_method == 'entropy':
                Information_Gain=entropy(splitted_groups,class_values)
                if(Information_Gain>max_entropy):
                   feature_index=feature
                   print(1)
                   feature_value=row[feature]
                   max_IG=Information_Gain
                   feature_groups=splitted_groups

    return {'index':feature_index, 'value':feature_value, 'groups': feature_groups}
   




def Decision_tree(train,test, max_depth=2, min_size=2,split_method="gini_index"):
	    # The algorith main function to build the tree and return predictions of test data 
        # Parameters : train : n dimensional array-like, shape (n_samples, n_features) selected for training .
        # test : n dimensional array-like, shape (n_samples, n_features)  selected for testing .
        # max_depth : determine the number of child nodes user wants in the tree (default=2)
        # min_size : determine minimum number of splits in the child nodes 
        # Returns : return the predicted vaulues of test data 
        predicted_labels=fit_predict(train, test,max_depth, min_size,split_method)
        true_labels=[row[-1] for row in test]
        return predicted_labels,true_labels
            
		

def feature_split(featureIndex, featurevalue, data):
    # Split feature values into two groups (left and right) nodes according to the splitting condition.
	left, right = [],[]
	for row in data:
		if row[featureIndex] < featurevalue:
			left.append(row)
		else:
			right.append(row)
	return left, right




#The selection of best attributes is being achieved with the help of a technique known as the Attribute selection measure (ASM).
def gini_index(groups, classes):
    # Gets the gini impurity for each feature to determine the node by taking the smallest value 
    # Parameters : groups : Two-groups of feature values results from the condition of the feature node split 
    # classes : unique values of classes (labels)
    
    sample_count = sum([len(group) for group in groups])
    gini_groups = []
    for group in groups:
        gini_group = 0.0
        group_size = len(group)
        for class_value in classes:
            if group_size == 0:
                continue
            probability = [row[-1] for row in group].count(class_value) / float(group_size)
            gini_group += (probability  * (1.0 - probability ))
        
        weighted_gini = group_size / sample_count
        gini_groups.append(weighted_gini * gini_group)
        
    return sum(gini_groups)
   
def entropy(groups, classes):
    # Gets the entropy impurity for each feature to determine the node by taking the largest value of the information gain .
    # Parameters : groups : Two-groups of feature values results from the condition of the feature node split 
    # classes : unique values of classes (labels)
    
    sample_count = sum([len(group) for group in groups])
    entropy_groups = []
    for group in groups:
        entropy= 0.0
        group_size = len(group)
        for class_value in classes:
            if group_size == 0:
                continue
            probability = [row[-1] for row in group].count(class_value) / float(group_size)
            entropy += -probability *np.log2(probability +1e-9)

        weighted_entropy = group_size / sample_count
        entropy_groups.append(weighted_entropy *entropy )
       
    return (1-sum(entropy_groups))

def child_nodes(node, max_depth, min_size, depth,split_method):
    # Create child nodes after determing the root node or the decision node 
	left, right = node['groups']
	del(node['groups'])
	# check for first split 
	if not left or not right:
		node['left'] = node['right'] = leaf_node(left + right)
		return
	# check for number of desired splits 
	if depth >= max_depth:
		node['left'], node['right'] = leaf_node(left), leaf_node(right)
		return
	# make a left leaf node if the desired size is not achieved 
	if len(left) <= min_size:
		node['left'] = leaf_node(left)
	else:
		node['left'] = best_split(left,split_method)
		child_nodes(node['left'], max_depth, min_size, depth+1,split_method)
	# make a right leaf node if the desired size is not achieved 
	if len(right) <= min_size:
		node['right'] = leaf_node(right)
	else:
		node['right'] = best_split(right,split_method)
		child_nodes(node['right'], max_depth, min_size, depth+1,split_method)



def leaf_node(group):
    # Create a terminal node value (also in case f lower size tree) , It puts a node in terminal 
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)



def build_tree(train, max_depth, min_size,split_method):
    #Build the decision tree with the desired depth and size 
	root = best_split(train,split_method)
	child_nodes(root, max_depth, min_size, 1,split_method)
	return root


def compare(tree, row):
    # Compare test features groups with the created tree to make a prediction.
	if row[tree['index']] < tree['value']:
		if isinstance(tree['left'], dict):
			return compare(tree['left'], row)
		else:
			return tree['left']
	else:
		if isinstance(tree['right'], dict):
			return compare(tree['right'], row)
		else:
			return tree['right']

def fit_predict(train, test, max_depth, min_size,split_method):
    # fit train data to build the tree 
    #predict the test values after applying the tree on its features values.
	tree = build_tree(train, max_depth, min_size,split_method)
	predictions = []
	for row in test:
		prediction = compare(tree, row)
		predictions.append(prediction)
	return(predictions)


def accuracy_metric(actual, predicted):
    # Get the performance measure (accuracy) of the implemented algorithm 
    # Parameters : actual: 1D-array ,shape(n_samples,) presents the actual values of classes (labels) of the testing trials.
    # predicted  : 1D-array ,shape(n_samples,) presents the values of classes (labels) of the testing trials.
	true = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			true += 1
	return ((true/float(len(actual))*100))


        
	


def main():


  rows=[]

  with open(r"CardioVascular\cardio_train.csv", 'r') as file:
    reader = csv.reader(file, delimiter = ';')
    for df_row in reader:
        rows.append(df_row)
    
    # Create the dataframe
    df = pd.DataFrame(rows)
    df=df.iloc[1:500, 2:]
    # Splitting the data into train and test data 
    train_data,test_data=train_test_split(df,0.9)
    #Reshaping data to fit the algorithm 
    train_data=np.array(train_data)
    train_data = train_data.astype(np.float64)
    train_data= train_data.astype(int)
    test_data=np.array(test_data)
    test_data = test_data.astype(np.float64)
    test_data= test_data.astype(int)

   

    # Choose tree size patrameters
    max_depth = 2
    min_size = 1
    ASM_method='gini_index'
    #ASM_method='entropy'
    

    predicted,actual= Decision_tree(train_data,test_data, max_depth, min_size,ASM_method)  
    accuracy = accuracy_metric(actual, predicted)
    print(accuracy)



if __name__=="__main__":
    main()