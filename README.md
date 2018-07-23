# Design

## Architecture

In this program, computations are mainly based on spark Dataframe rather than RDD. The input csv is read via spark csv reader and it generate a Dataframe with label and 784 columns features.

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

 Then these dataframes will go through a VectorAssember to format themselves.

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png)

 Next step the dimension of features will be reduced with the help of PCA function in order to accelerate the running process. 

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)

Before the KNN, we extract the pca_feature and label of the training data and compute them into an array and broadcast to each node.

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png)

 Next is the KNN computation. The computation formula is simple:

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png)

Then we sort the matrix by row and return the first K label. The label that appear the most is the predicted label of the test point. 

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png)

The following step is to calculate the confusion matrix and the classification report.

Basically, all the functions are achieved without error.

## Optimization

However, during those time, Code has improved several times.

### Numpy Matrix 

When I first compute the KNN, I was calculating them row by row. In this case, it would cost over 24 hours to finish all the steps which is normally inefficient. Using matrix for the computation apparently saving lots time since it would only need to map 10000 times instead of 600 million. It helps me cut down the time from over 24hours to less than 3 minutes. 

### Argsort

Sorting is always the time-consuming step. I used to sort a two-dimension array with label in one column and distance in the others. With the help of Argsort, I only need to sort the distance list and it will return the index of list in ascending order. Then, we only need to use the index to map the label list and it will return the label in ascending distance order. In this way, it used to cost 0.02s for each output and now it cost 0.0075s for each, which is 3 time faster.

### Repartition

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png)

During computing, I find out that the collect() and saveAsTextFile() methods cost most of the computation time. What’s more, they are not able to work very parallelly with only 5 tasks running at the same time. With the help of repartition() function, tasks can be split into small subset and are able to run as parallel as possible. For this part, I refer to apache document and find out that the default size of each task is already defined. In this case, our size of input for these two methods can only be split into 5 tasks by default. With defining the repartition number, it can be split as many as we wish. However, I recommend setting it same as executors * cores since it will be able to run exactly the same number of tasks at the same time as the total cores number. I’ve try up to 32 repartitions with corresponding 32 total cores. However, the KNN computation time is 36s which is longer than 22s with only 16 repartitions. In conclusion, this factor should be defined based on situations. 

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image016.png)

Figure1: 5 partition of computations

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image018.png)

Figure2: 16 Repartition of computations

### Broadcast

It is a built-in function of spark context for broadcasting the values to different nodes. In this case, node will save time acquiring context each time from the master node. However, in different circumstance, result would be opposite. When I run the program with K=5 and PCA=50, the one with broadcast is 20s slower than the one without. For program with K=10 and PCA=100, with the help of broadcasting, it cut down over 70s overall. In result, it probably depends on the frequency of asking the context from master node. If the frequency is high enough and the broadcasting is able to help them cut down the asking frequency. On the contrast, if the frequency is low, the broadcasting time may be overwhelming the overall asking time. Therefore, with larger input, using broadcasting will become necessary.

## Sample Result

The following sample parameters: 8 executors, 2 cores, PCA=50, D=5

![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image020.png)                 ![img](file:///C:/Users/user/AppData/Local/Temp/msohtmlclip1/01/clip_image022.png)

​               Figure3: KNN only         	   Figure4: KNN & produce predict label