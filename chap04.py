# 字符串
data = 'hello world'
print(data[0])
print(data[1:5])
print(len(data))
print(data)
if len(data) > 10:
    print(True)
else:
    print(False)
for i in range(len(data)):
    print(data[i])

# 元组 不能重新赋值
a = (1, 2, 3)
print(a)

# 列表 类似元组 可重新赋值
b = [1, 2, 3]
print(b)

# 字典 可变容器  存储任意类型对象
dict = {'a': 123, 'b': 1233, '132': a}
print(dict['132'])


# 函数
def function(x, y):
    return x + 2 * y


result = function(1, 2)
print(result)

# Numpy
import numpy as np

myarr = np.array([1, 2, 3])
print(myarr)
print(myarr.shape)

# numpy运算
arr1 = np.array([[1, 2, 3], [2, 3, 4]])
arr2 = np.array([[11, 22,33], [22, 33,11]])
print(arr2 * arr1)


#matplotlib 绘图
import matplotlib.pyplot as plt
plt.plot(arr2)

plt.xlabel('x')
plt.ylabel('y')
# plt.show()

plt.scatter(arr1,arr2)
plt.show()

# pandas速成w
import pandas as pd
mynewarr=np.array([1,2,3])
index=['a','b','c']
myseries=pd.Series(mynewarr,index=index)
print(myseries)

#dataframe
rowindex=['row1','row2','row3']
mydataframe=pd.DataFrame(mynewarr,index=rowindex)
print(mydataframe)