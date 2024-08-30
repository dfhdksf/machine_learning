#单变量回归模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression  #导入线性回归类

data = pd.read_csv('../data/world-happiness-report-2017.csv')#导入数据

# 得到训练和测试数据
train_data = data.sample(frac = 0.8)#随机抽取data中80%的数据作为训练集
test_data = data.drop(train_data.index)#去除训练集的索引即将剩下的数据作为测试集

input_param_name = 'Economy..GDP.per.Capita.'#选择csv文件中数据的某一列作为特征值
output_param_name = 'Happiness.Score'#作为实际值

x_train = train_data[[input_param_name]].values #取得要训练的数据,values方法是将数据转为ndarray格式
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train,y_train,label='Train data')#散点图中蓝色为训练集，
plt.scatter(x_test,y_test,label='test data')# 橙色为测试集
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()#图例，即Train data和test data的样式提示
plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train,y_train)#对LinearRegression类的实例化
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)

print ('开始时的损失：',cost_history[0])
print ('训练后的损失：',cost_history[-1])

plt.plot(range(num_iterations),cost_history)#绘制二维数据的线性图形
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')#绘图的标题
plt.show()

#上面只进行了训练，下面为测试阶段
predictions_num = 100
#从训练集中最小值和最大值之间以相同步长生成100个数并转为列向量
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)#算出预测值

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_predictions,y_predictions,'r',label = 'Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()