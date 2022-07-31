![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.001.png) ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.002.png)

Cairo University 

Faculty of Engineering            Systems and Biomedical Department 

` `**Decision tree Implementation**  

**Submitted by:** 

`                  `Ashar Seif Al-Naser Saleh                   Sec: 1     BN: 9          

` `SBE452\_AI  **Dr.Inas A. Yassine** 11 December, 2021 

- **Problem (1:B)** 

Recommend another type of trees than ID3 that would build a less deep tree than ID3.  

The recommended type is C4.5 where it has the characteristic of Error Based pruning which minimize the trees size by cutting the less important leaves out unlike the ID3 where No pruning is done. 

- **Problem (2)** 

How does the Decision Tree algorithm work? 

The basic idea behind any decision tree algorithm is as follows: 

- Select the best attribute using Attribute Selection Measures (ASM) to split the records and we will use entropy or gini\_index in this step. 
- Make that attribute a decision node and breaks the dataset into smaller subsets. 
- Starts tree building by repeating this process recursively for each child until one of the condition will match: 
- All the tuples belong to the same attribute value. 
- There are no more remaining attributes. 
- There are no more instances. 

![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.003.png)


The sequence of code implementation functions  

- Reading data using dataframe 
1. rows=[] 
 
1. with open(r"CardioVascular\cardio\_train.csv", 'r') as file: 
1. reader = csv.reader(file, delimiter = ';') 
1. for df\_row in reader: 
1. rows.append(df\_row) 
 
1. # Create the dataframe 
1. df = pd.DataFrame(rows) 
1. df=df.iloc[1:, 2:] 
- Splitting the data into train and test data using the train\_test\_split function which split data corresponding to the percentage user enter to determine the size of train and test data. 

train\_data,test\_data=train\_test\_split(df,0.9) 

def train\_test\_split(data,percentage):         length=data.shape[0] 

`    `percent=int(length\*percentage) 

`    `train\_data=data.iloc[:percent+1,:]     test\_data=data.iloc[percent+1:,:]     return  train\_data, test\_data ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.004.png)

- Use fit\_predict function to first train the data and build the tree and this is done by multiple steps : 

\1. build\_tree function which builds the decision tree with the desired  depth  and  size  of  tree  using  two  other  functions (best\_split and child\_nodes) 

def build\_tree(train, max\_depth, min\_size,split\_method):     root = best\_split(train,split\_method) 

child\_nodes(root, max\_depth, min\_size, 1,split\_method) return root 

- best\_split function gets the best root node to split data on it using one of Attribute Selection Measures (ASM) determined by user .  
- It calculates the ASM (entropy (information gain ) or gini\_index ) for each feature to get the node impurity in case of using it as a root node and takes the minimum value of the gini to consider its feature as a root node or the maximum value of information gain . 
- In case of features with multiple values (Which can be considered as continuous, Ex: height ) , I put a condition which discretize the values  into  categories   to  avoid  entering  an  infinite  loop.  The condition state that if there are more than nine different values for a feature then we split these values into number of groups which equaled to : 

number\_of\_categories=int(len(uniqueValues)/5) 

def best\_split(data,split\_method): ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.005.png)

""" 

Get the best root node to split data on it  ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.006.png)

`    `Parameters 

`    `---------- 

`         `data: n dimensional array-like, shape (n\_samples, n\_features) ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.006.png)

`         `Returns 

`         `------- 

`         `index : The index of the root  

`         `value : The value at which we split the node and put the spliiting condition on it . 

`         `groups : The splitted feature values groups after comparing with the splitting condition. 

`    `""" 

`    `data=np.array(data) 

`    `class\_values = list(set(row[-1] for row in data))      

`    `max\_gini=0.5 

`    `max\_entropy=0.3 

`    `splitted\_groups=None 

`    `features\_length=len(data[0])-1 

`    `for feature in range(features\_length): 

`        `uniqueValues = np.unique(data[:,feature]) 

`        `if len(uniqueValues)>9: 

`            `descritized\_data=data[:,feature] 

`            `sorted\_feature\_values=np.sort(data[:,feature]) 

`            `number\_of\_categories=int(len(uniqueValues)/5) 

`            `feature\_value\_groups=[ sorted\_feature\_values[i:i + number\_of\_categories] for i in range(0, len(sorted\_feature\_values), number\_of\_categories)] 

`            `for category in range(len(feature\_value\_groups)):                                  for i in range(len(descritized\_data)): 

`                    `if descritized\_data[i] in feature\_value\_groups[category]:                         descritized\_data[i]=category     

`            `data[:,feature]= descritized\_data  ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.007.png)

`        `for row in data: 

`            `splitted\_groups=feature\_split(feature,row[feature],data)             if split\_method == "gini\_index": 

`               `gini=gini\_index(splitted\_groups,class\_values) 

`               `print(gini) 

`               `if(gini<max\_gini): 

`                   `feature\_index=feature 

`                   `feature\_value=row[feature] 

`                   `max\_gini=gini 

`                   `feature\_groups=splitted\_groups 

`            `elif split\_method == 'entropy': 

`                `Information\_Gain=entropy(splitted\_groups,class\_values)                 if(Information\_Gain>max\_entropy): 

`                   `feature\_index=feature 

`                   `print(1) 

`                   `feature\_value=row[feature] 

`                   `max\_IG=Information\_Gain 

`                   `feature\_groups=splitted\_groups ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.008.png)

`    `return {'index':feature\_index, 'value':feature\_value, 'groups': feature\_groups} 

- 2- child\_nodes : Create child nodes after determine the root node or the decision node, It do multiple functions , first :if it is the first split , it put the groups (left, right) as leaf nodes, second :it checks 

if the tree got to its desired size, if not it process a node as a leaf node (right or left according to the uncompleted size direction.  

def child\_nodes(node, max\_depth, min\_size, depth,split\_method): 

- Create child nodes after determine the root node or the decision node  

`    `left, right = node['groups'] 

`    `del(node['groups']) 

- check for first split  

`    `if not left or not right: 

`        `node['left'] = node['right'] = leaf\_node(left + right) 

`        `return 

- check for number of desired splits  

`    `if depth >= max\_depth: 

`        `node['left'], node['right'] = leaf\_node(left), leaf\_node(right) 

`        `return 

- make a left leaf node if the desired size is not achieved  

`    `if len(left) <= min\_size: 

`        `node['left'] = leaf\_node(left) 

`    `else: 

`        `node['left'] = best\_split(left,split\_method) 

`        `child\_nodes(node['left'], max\_depth, min\_size, depth+1,split\_method) 

- make a right leaf node if the desired size is not achieved  

`    `if len(right) <= min\_size: 

`        `node['right'] = leaf\_node(right) 

`    `else: 

`        `node['right'] = best\_split(right,split\_method) 

`        `child\_nodes(node['right'], max\_depth, min\_size, depth+1,split\_method) 

- After these steps , the decision tree is ready and then we predict values for the test data using two functions : 
1. Compare: which apply the tree on every row on test data and return the prediction values. 
1. def compare(tree, row): 
1. # Compare test features groups with the created tree to make a prediction. 
1. if row[tree['index']] < tree['value']: 
1. if isinstance(tree['left'], dict): 
1. return compare(tree['left'], row) 
1. else: 
8. return tree['left'] 
8. else: 
8. if isinstance(tree['right'], dict): 
8. return compare(tree['right'], row) 
8. else: 
8. return tree['right'] 

\14. 

\2. accuracy\_metric : which takes the predicted values and the actual values of test data and evaluates the accuracy of the prediction . 

![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.009.png)

def accuracy\_metric(actual, predicted): 

- Get the performance measure (accuracy) of the implemented algorithm  
- Parameters : actual: 1D-array ,shape(n\_samples,) presents the actual values 

of classes (labels) of the testing trials. 

- predicted  : 1D-array ,shape(n\_samples,) presents the values of classes 

(labels) of the testing trials. 

`    `true = 0 

`    `for i in range(len(actual)): 

`        `if actual[i] == predicted[i]: 

`            `true += 1 

`    `return ((true/float(len(actual))\*100)) ![](Aspose.Words.537b5ea9-3202-4d11-98f9-8048dea9661d.010.png)

**Observation  on  accuracy  :The  accuracy  of  the implementation increase with the decreasing of number of samples from the data ( It reaches 100% using only 15 sample in both gini and entropy) but in general it gives around 66%.** 
