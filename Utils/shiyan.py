import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

import os

#file_name='shiyan.csv'
#数据处理阶段  split切片点
def load_data(file_name,sequence_length=10,split=0.8):
    df=pd.read_csv(file_name,sep=',',usecols=[1])
    data_all=np.array(df).astype(float)
    scaler=MinMaxScaler()
    data_all=scaler.fit_transform(data_all)
    data=[]
    for i in range(len(data_all)-sequence_length-1):
        data.append(data_all[i:i+sequence_length+1])
    reshaped_data=np.array(data).astype('float64')
    #数据打乱
    #np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x=reshaped_data[:,:-1]
    y=reshaped_data[:,-1]
    split_boundary=int(reshaped_data.shape[0]*split)
    train_x=x[:split_boundary]
    test_x=x[split_boundary:]
    
    train_y=y[:split_boundary]
    test_y=y[split_boundary:]
    
    return train_x,train_y,test_x,test_y,scaler

#建立模型
def build_model():#layers[1,50,100,1]
    model=Sequential()
    
    # 5-隐层神经元数量
    model.add(LSTM(input_dim=1,output_dim=50,init='he_normal',activation='tanh',return_sequences=True))
    # 4-抽稀层节点去除比例(dropout)
    model.add(Dropout(1))
    #print(model.layers)
    
    model.add(LSTM(100,init='he_normal',activation='tanh',return_sequences=False))
    # 4-抽稀层节点去除比例(dropout)
    model.add(Dropout(1))
    
    # 3-权重初始化(init)
    model.add(Dense(output_dim=1,init='he_normal',activation='tanh'))
    # 2-激活函数(activation)
    #model.add(Activation('linear'))
    
    # 1-优化器(optimizer)
    model.compile(loss='mse',optimizer='RMSprop')
    return model

#训练模型
def train_model(train_x,train_y,test_x,test_y):
    model=build_model()
    
    try:
        model.fit(train_x,train_y,batch_size=700,nb_epoch=70,validation_split=0.1)
        predict=model.predict(test_x)
        predict=np.reshape(predict,(predict.size,))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)

    print(predict)
    print(test_y)

    f=open("output.txt","w")
    print(predict,'\n',test_y,file=f)

    try:
        fig=plt.figure(1)
        plt.plot(predict,'r:')
        plt.plot(test_y,'g-')
        plt.legend(['predict','true'])
    except Exception as e:
        print(e)
    return predict,test_y

#调用模块
# if __name__=='__main__':
def run_model(tmp):
    train_x,train_y,test_x,test_y,scaler=load_data(tmp)
    
    train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
    test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))
    
    predict_y,test_y=train_model(train_x,train_y,test_x,test_y)
    
    predict_y=scaler.inverse_transform([[i] for i in predict_y])
    
    test_y=scaler.inverse_transform(test_y)
    
    fig2=plt.figure(2)
    plt.plot(predict_y,'g:')
    plt.plot(test_y,'r-')
    plt.show()