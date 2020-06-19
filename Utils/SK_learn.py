import os
print(os.path.expanduser("~"))

import numpy as np
import csv

data_folder="D:\\iPython数据挖掘入门与实践"
data_filename=os.path.join(data_folder,"Ionosphere","Ionosphere.data")
print(data_filename)

X=np.zeros((351,34),dtype='float')
y=np.zeros((351,),dtype='bool')

with open(data_filename,'r') as input_file:
    reader=csv.reader(input_file)
    for i,row in enumerate(reader):
        # Get the data, converting each item to a float
        # 获取数据转换成值
        data=[float(datum) for datum in row[:-1]]
        # Set the appropriate row in our dataset
        # 设置适应行
        X[i]=data
        y[i]=row[-1]=='g'