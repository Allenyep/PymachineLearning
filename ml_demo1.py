from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 第一个机器学习项目
# 莺尾花分类

filename = 'iris.data.csv'
names = ['separ-length', 'separ-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)

# 查看行列数
print('hang: %s,lie %s' % dataset.shape)
# 查看前十行
print(dataset.head(10))
# 统计描述数据信息 输出数量 均值 中位值 最小值 四分卫值 最大值 标准差(std)
print(dataset.describe())
# 分类分布情况
print(dataset.groupby('class').size())
print()

# 箱线图
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# pyplot.show()

# 直方图
dataset.hist()
# pyplot.show()

#散点矩阵图
scatter_matrix(dataset)
# pyplot.show()

#分离数据集
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.2
seed=7
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=validation_size,random_state=seed)

#算法审查
# 线性回归 线性判别分析 K邻近 分类与回归树  贝叶斯分类器  支持向量机
models={}
models['LR']=LogisticRegression()
models['LDA']=LinearDiscriminantAnalysis()
models['KNN']=KNeighborsClassifier()
models['CART']=DecisionTreeClassifier()
models['NB']=GaussianNB()
models['SVM']=SVC()

#评估算法
result=[]
for key in models:
    kfold=KFold(n_splits=10,random_state=seed)
    cv_result=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring='accuracy')
    result.append(cv_result)
    print('%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))

#箱线图比较算法
fiq=pyplot.figure()
fiq.suptitle('Algorithm Comparison')
ax=fiq.add_subplot(111)
pyplot.boxplot(result)
ax.set_xticklabels(models.keys())
# pyplot.show()

#使用评估数据集评估算法
svm=SVC()
svm.fit(X=X_train,y=Y_train)
predictions=svm.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))