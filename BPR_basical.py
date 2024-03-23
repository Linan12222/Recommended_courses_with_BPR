import copy
import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import scores
import pandas as pd
import numpy as np
import os
class BPR:
    #用户数，测试集942，训练集943，总共943人
    user_count = 2322
    #项目数
    item_count = 685
    #k个主题,k数
    latent_factors = 30
    #步长α
    lr = 0.01
    #参数λ
    reg = 0.02
    #训练次数
    train_count = 20
    #训练集
    train_data_path = 'train2.txt'
    #测试集
    test_data_path = 'test2.txt'
    #U-I的大小
    size_u_i = user_count * item_count
    # 随机设定的U，V矩阵(即公式中的Wuk和Hik)矩阵
    U = np.random.rand(user_count, latent_factors) * 0.01 #大小无所谓
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01
    #生成一个用户数*项目数大小的全0矩阵
    test_data = np.zeros((user_count, item_count))
    #生成一个一维的全0矩阵
    test = np.zeros(size_u_i)
    #再生成一个一维的全0矩阵
    predict_ = np.zeros(size_u_i)

    #获取U-I数据对应
    '''
    函数说明：通过文件路径，获取U-I数据
    Paramaters:
        输入要读入的文件路径path
    Returns:
        输出一个字典user_ratings，包含用户-项目的键值对
    '''
    def load_data(self, path):
        user_ratings = defaultdict(set)
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
        return user_ratings
    '''
    函数说明：通过文件路径，获取测试集数据
    Paramaters：
        测试集文件路径path
    Returns:
        输出一个numpy.ndarray文件（n维数组）test_data,其中把含有反馈信息的数据置为1
    '''
    #获取测试集的评分矩阵
    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1
    '''
    函数说明：对训练集数据字典处理，通过随机选取，（用户，交互，未交互）三元组，更新分解后的两个矩阵
    Parameters：
        输入要处理的训练集用户项目字典
    Returns：
        对分解后的两个矩阵以及偏置矩阵分别更新
    '''
    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # 随机获取一个用户
            u = random.randint(1, self.user_count) #找到一个user
            # 训练集和测试集的用户不是全都一样的,比如train有948,而test最大为943
            if u not in user_ratings_train.keys():
                continue
            # 从用户的U-I中随机选取1个Item
            i = random.sample(user_ratings_train[u], 1)[0] #找到一个item，被评分
            # 随机选取一个用户u没有评分的项目
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count) #找到一个item，没有被评分
            #构成一个三元组（uesr,item_have_score,item_no_score)
            # python中的取值从0开始
            u = u - 1
            i = i - 1
            j = j - 1
            #BPR
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # 更新2个矩阵
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # 更新偏置项
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])
    '''
    函数说明：通过输入分解后的用户项目矩阵得到预测矩阵predict
    Parameters:
        输入分别为用户项目矩阵，也就是user*k和item*k
    Returns：
        输出相乘后的预测矩阵，即我们所要的评分矩阵，这里是矩阵的乘法
    '''
    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    #主函数
    def main(self):
        #获取U-I的{1:{2,5,1,2}....}数据
        user_ratings_train = self.load_data(self.train_data_path)
        #获取测试集的评分矩阵
        self.load_test_data(self.test_data_path)
        #将test_data矩阵拍平
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        #训练
        for i in range(self.train_count):
            self.train(user_ratings_train)  #训练10000次完成
        predict_matrix = self.predict(self.U, self.V) #将训练完成的矩阵內积
        # 预测
        self.predict_ = predict_matrix.getA().reshape(-1)  #.getA()将自身矩阵变量转化为ndarray类型的变量，然后再拍平
        print("predict_new",self.predict_)
        #第一个是训练数据的字典，第二个是预测出来的矩阵乘积，第三个是项目的个数，将已经有所交集的数据归为0
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        #test是个01的长向量，predict_也是一个长向量
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:     {}'.format(auc_score))
        # Top-K evaluation
        topks = [1,5,10,20]
        MRRS,HRS,Precs,Recas,NDCGs = [auc_score],[],[],[],[]
        for topk in topks:
            MRR,HR,Prec,Reca,NDCG = scores.topK_scores(self.test, self.predict_, topk, self.user_count, self.item_count)
            MRRS.append(MRR),HRS.append(HR),Precs.append(Prec),Recas.append(Reca),NDCGs.append(NDCG)
        index = pd.DataFrame(MRRS+HRS+Precs+Recas+NDCGs).T
        return index
    '''
    函数说明：对结果进行修正，即用户已经产生交互的用户项目进行剔除，只保留没有产生用户项目的交互的数据
    Paramaters:
        输入用户项目字典集，以及一维的预测矩阵，项目个数
    Returns:
        输出修正后的预测评分一维的预测矩阵
    '''
def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict

if __name__ == '__main__':
    #调用类的主函数
    indexs = pd.DataFrame(np.zeros([10,21]))
    for z in range(10):
        bpr = BPR()
        index = bpr.main()
        indexs.loc[z,:] = index.loc[0,:]
    indexs.columns = ['AUC', 'MRR@1', 'MRR@5', 'MRR@10', 'MRR@20', 'HR@1', 'HR@5', 'HR@10', 'HR@20', 'Prec@1', 'Prec@5',
                     'Prec@10', 'Prec@20', 'Reca@1', 'Reca@5', 'Reca@10', 'Reca@20', 'NDCG@1', 'NDCG@5', 'NDCG@10',
                     'NDCG@20']
    indexs.loc[10] = indexs.apply(lambda x: x.mean())
    indexs.index = [1,2,3,4,5,6,7,8,9,10,'mean']
    indexs.to_excel('index.xlsx',sheet_name='yes')
    print(indexs)
