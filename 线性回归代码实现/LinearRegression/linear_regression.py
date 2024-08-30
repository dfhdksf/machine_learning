import numpy as np
from utils.features import prepare_for_training #导入预处理模块

class LinearRegression:
        #__init__方法，在创建实例的时候，就把属性绑上去
        #data表示数据，lables为标签（因为是无监督）    polynomial_degree,sinusoid_degree分别是对特征做特征变换
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """ 
        #1.1 对数据进行预处理操作
        (data_processed,
         features_mean, 
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=True)
         #1.2 将接收预处理后的数据
        self.data = data_processed
        self.labels = labels  #预处理不会处理标签
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        #2.先得到所有的特征个数
        num_features = self.data.shape[1]#特征的数量=数据的列数，shape[1]表示列数
        #3.初始化参数矩阵
        self.theta = np.zeros((num_features,1)) #将参数theta设置为num_features行1列的0矩阵
        
    #训练函数，alpha：学习率；num_iterations：迭代次数，即每次迭代都会根据损失函数的梯度来更新模型参数
    def train(self,alpha,num_iterations = 500):
        """
                    训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history#训练完后返回最终的theta和每一次迭代的误差
        
    #gradient_descent即梯度下降
    def gradient_descent(self,alpha,num_iterations):
        """
                    实际迭代模块，会迭代num_iterations次
        """
        cost_history = []#指定为list，记录每次迭代的损失值
        for _ in range(num_iterations):#迭代500次
            self.gradient_step(alpha)#先梯度下降
            cost_history.append(self.cost_function(self.data,self.labels))#再记录保存损失值
        return cost_history

    def gradient_step(self,alpha):    
        """
                    梯度下降参数更新计算方法，注意是矩阵运算
                    因为每次迭代（梯度下降）都要重新计算参数theta
                    这里是小批量梯度下降法
                    这个梯度下降函数只执行了一次
        """
        num_examples = self.data.shape[0] #样本数量 shape[0]表示数据的行数即样本数量
        #prediction即预测值
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels  #预测值-真实值（标签）（delta是行向量还是列向量？？？？）
        theta = self.theta #拿到当前的theta          
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T #更新theta
        self.theta = theta
        
        
    def cost_function(self,data,labels):
        """
                    损失值计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels#预测值-实际值
        cost = (1/2)*np.dot(delta.T,delta)/num_examples#均方误差
        return cost[0][0]#返回损失值所在的位置，通过print可以找出


    @staticmethod #好处是不用实例化就可以直接调用
    #预测函数计算预测值
    def hypothesis(data,theta):   
        predictions = np.dot(data,theta)#预测值为θ与样本x的矩阵乘积（刚好行*列=一个数）
        return predictions


     #再获得一次损失值用于其他方面
    def get_cost(self,data,labels):  
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
        
        return self.cost_function(data_processed,labels)
    def predict(self,data):
        """
                    用训练的参数模型，去预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
         
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        
        return predictions
        
        
        
        