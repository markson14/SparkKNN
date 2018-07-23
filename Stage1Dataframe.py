# Import all necessary libraries and setup the environment for matplotlib
import time
from sys import argv
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, DoubleType
import matplotlib.pyplot as plt
from numpy import sum, sqrt
from pyspark import SparkContext
spark = SparkSession \
    .builder \
    .appName("Stage 1 Dataframe Complete") \
    .config("spark.executor.cores",argv[-3]) \
    .getOrCreate()

K = int(argv[-2])
PCA_D = int(argv[-1])
start = time.time()
sc = SparkContext.getOrCreate()


# Cluster Version
test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
output_path = "MNIST/predict-label"

# Local Version
# test_datafile= "MNIST/Test-label-28x28.csv"
# train_datafile = "MNIST/Train-label-28x28.csv"
# output_path = "MNIST/predict-label"

num_test_samples = 10000
num_train_samples = 60000

test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")
train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")


# Formatting the Dataframe
assembler = VectorAssembler(inputCols=test_df.columns[1:],
    outputCol="features")
test_vectors = assembler.transform(test_df).select(test_df[0].alias('label'),"features").repartition(16)
train_vectors = assembler.transform(train_df).select(train_df[0].alias('label'),"features").repartition(16)

# PCA implementing
pca = PCA(k=PCA_D, inputCol='features', outputCol='pca') 
model = pca.fit(test_vectors)
train_data = model.transform(train_vectors).select('label', 'pca')
test_data = model.transform(test_vectors).select('label', 'pca')

# KNN Data Preprocessing

# ONE time collect() function
train_matrix = []
train_label = []
train_rows = train_data.rdd.collect()
for i in train_rows:
    train_matrix.append(i.pca)
    train_label.append(i.label)

train_label = sc.broadcast(np.array(train_label)) 
train_matrix = sc.broadcast(np.array(train_matrix))
# train_label = np.array(train_label)
# train_matrix = np.array(train_matrix)


#KNN implementing
def knnPred(line):
    pca = line
    cal = sqrt(sum((train_matrix.value - np.tile(pca,(len(train_matrix.value),1)))**2, axis=1)) # Calculate the Euclidean Distance
    cal = np.argsort(cal)        # return the index of The List of Distance in ascending order. Save the sorting time and cut down half of the runing time
    pred_label_list = train_label.value[cal][:K] # return the K nearest neibourages' label
    pred_label = np.bincount(pred_label_list).argmax() # return the label with max appearance
    return  int(pred_label)

knn_udf = udf(knnPred, IntegerType()) #user define function

# confusion matrix
def confusion_matrix(test_true, test_pred,class_list):
    num_class_list = len(class_list)
    confusion_matrix = np.zeros((num_class_list, num_class_list))
    match_count = 0
    n_test = len(test_true)
    for k in range(n_test):
        cm_j = class_list.index(test_pred[k])
        cm_i = class_list.index(test_true[k])
        confusion_matrix[cm_i, cm_j] += 1
        if test_pred[k] == test_true[k]:
            match = True
            match_count += 1    
        else:
            match = False
    return confusion_matrix

#computing classification report
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()
def f1_score(label, confusion_matrix):
    row = confusion_matrix[label, :]
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label]*2 / (row.sum()+col.sum())
def support(label,confusion_matrix):
    row = confusion_matrix[label, :]
    return int(row.sum())
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

#start computation
pred = test_data.withColumn("pred_label",knn_udf(test_data.pca)) 
label_list = pred.select('pred_label','label').collect()
# print(label_list)
pred_label=[]
real_label=[]
for i in label_list:
    pred_label.append(i.pred_label)
    real_label.append(i.label)

predicted = pred.select('pred_label').rdd.saveAsTextFile(output_path)
full_class_list = [0,1,2,3,4,5,6,7,8,9]
cm = np.array(confusion_matrix(real_label,pred_label,full_class_list)).astype(int)
print('\n')
print("Confusion Matrix")
print(cm)
print('\n')
print("label precision recall f1-score Support")
pc=[]
rc=[]
f1=[]
sp=[]
for label in full_class_list:
    pc.append(precision(label, cm.astype(float)))
    rc.append(recall(label, cm.astype(float)))
    f1.append(f1_score(label,cm.astype(float)))
    sp.append(support(label,cm))
    print("{:5d} {:9.3f} {:6.3f} {:6.3f} {:5d}".format(label,precision(label, cm.astype(float)) \
    ,recall(label, cm.astype(float)),f1_score(label,cm.astype(float)),support(label,cm)))
print("average  {:6.3f} {:6.3f} {:6.3f} {:5d}".format(sum(pc)/len(full_class_list),sum(rc)/len(full_class_list),sum(f1)/len(full_class_list),sum(sp)))
print('\n')
acc = accuracy(cm.astype(float))
print('Accuracy: {}%'.format(round(acc*100,3)))
print('\n')
end = time.time()
print('(Execution_Time: {}s)'.format(round(end - start, 3)))
print('\n')



