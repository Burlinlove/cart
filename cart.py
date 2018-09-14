import numpy as np
import pandas as pd
from itertools import *   
import pdb   
from copy import deepcopy 
from math import ceil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from math import log

##计算的gini指数
def _gini(labely):
    if len(labely) == 0:   ###父节点为这一列属性这个值为空  
        return 0
    p = np.sum(labely) / len(labely)    #基尼系数

    gini_value = 2 * p * (1-p)
    return  gini_value  #因为是二分类，所以基尼系数的公式推出此公式是   2*p*(1-p) 


##生成一个特征的取值组合
def _featuresplit(features):   
    features = list(set(features))
    combiList = [] 
    
    if len(features) <= 1:                ##如果特征值只有一种的话，就不必再求组合
        return features
    
    count = len(features) // 2 + 1        ##计算特征值取值的长度，用于做特征值得组合，以便找最小的gain_gini值组合  
    
    combiList = []   
    for i in range(1,count):   
        com = list(combinations(features, len(features[0:i])))   
        combiList.extend(com)   
    
    return combiList


##连续的特征求解gain_gini
def _min_confeature_gain_gini(featurecol,labely,maxfeaturebin):
    
    feature = list(set(featurecol))     ##进行一个排序
    feature.sort()

    if len(feature) <= 1:               ##如果feature取值只有一种的话，直接返回这个取值，并取值为1
        return feature[0],1
    
    split_points = []
    
    if len(feature) >= maxfeaturebin:             ##如果取值的个数大于了300，开始分箱
        value_gap = (max(feature)-min(feature)) / maxfeaturebin    
        min_value = min(feature)
        for i in range(maxfeaturebin):
            split_point = min_value + i*value_gap
            split_points.append(split_point)
    else:                               ##如果取值的个数小于300,就不分箱
        for i in range(len(feature)-1):
            split_point = (feature[i] + feature[i+1]) / 2     #中位数作为一个分裂点
            split_points.append(split_point)
            
    
    feature_point_gain_list = []
    
    for split_point in split_points:
        left_index = []
        right_index = []
        
        for i in range(len(featurecol)):
            if featurecol[i] <= split_point:           #连续性特征 
                left_index.append(i)
            else:
                right_index.append(i)
        
        left_y = labely[left_index]
        right_y = labely[right_index]
        
        left_gain = _gini(left_y)            ##计算左右gini
        right_gain = _gini(right_y)   
        
        gain_gini_values = len(left_y) / len(labely) * left_gain + len(right_y) / len(labely) * right_gain
        feature_point_gain_list.append(gain_gini_values)    ##计算gain_gini_values

    min_index = feature_point_gain_list.index(min(feature_point_gain_list))
    return split_points[min_index],min(feature_point_gain_list)  ##返回最小值得point,和最小的gain_gini    


##寻找一个特征的最小gain_gini
def _min_feature_gain_gini(featurecol,labely):
    
#     feature_value_groups = _featuresplit(featurecol)    #####不用两两组合或者其他的组合的原因是因
                                                          #####为这样会计算组合的时候会非常的慢
    feature_value_points = list(set(featurecol))        ##直接计算按照每个离散值的gini
    
    if len(feature_value_points) <= 1:         ##如果这个特征列只有一种值，就返回这个值以及将1作为gini值
        return feature_value_points[0],1
    
    feature_point_gain_list = []
    
    
    for point in feature_value_points:
        left_index = []
        right_index = []
        
        for i in range(len(featurecol)):      ##利用得到的group划分数据集，求最小的gain_gini_value
            if featurecol[i] == point:
                left_index.append(i)
            else:
                right_index.append(i)
        
        left_y = labely[left_index]
        right_y = labely[right_index]
        
        left_gain = _gini(left_y)            ##计算左右gain_gini
        right_gain = _gini(right_y)
        
        gain_gini_values = len(left_y) / len(labely) * left_gain + len(right_y) / len(labely) * right_gain
        feature_point_gain_list.append(gain_gini_values)    ##计算gain_gini_values
 
    min_index = feature_point_gain_list.index(min(feature_point_gain_list))

    return feature_value_points[min_index],min(feature_point_gain_list)  ##返回最小值得point,和最小的gain_gini


##寻找所有特征的基尼系数最小的值,
##返回该特征下的group
#featurescols:n x k, labely : n x 1s
def _min_gain_gini(featurecols,labely,features_isDiscrete_arr,maxfeaturebin):
    
    gain_gini_list = []
    k = featurecols.shape[1]
    
    feature_dict = {}                            #创建一个字典用于存储特征point和min_gain
    feature_gain_list = []                       #每一个特征的最小gain_gini
    
    for i in tqdm(range(k),desc = 'train:'):
        feature = featurecols[:,i]
        isDiscrete = features_isDiscrete_arr[i]
        
        if isDiscrete == 1 :            ####离散值的判定，求和取余为0
            feature_dict[i] = _min_feature_gain_gini(feature,labely)
        else:                                    ##连续值的判定，求和取余不为0
            feature_dict[i] = _min_confeature_gain_gini(feature,labely,maxfeaturebin)
        
        gain_gini_value = feature_dict[i][1]     #得到最小gain_gini
        gain_gini_list.append(gain_gini_value) 
        
    min_gain_index = gain_gini_list.index(min(gain_gini_list))
    min_feature_col = min_gain_index
    min_gain_feature_point = feature_dict[min_gain_index][0]      ##最小gain的feature_point

    return min_feature_col,min_gain_feature_point


#input:data   格式：数据帧或者数组，推荐数据帧格式
#output:  dtypesarr  格式：1维数组，值：1代表数据是离散，0：连续
def _is_dis_con_feature(data):    
    assert type(data).__name__ == 'DataFrame' \
        or type(data).__name__ == 'ndarray'                     ##判断是否是数据帧或者数组格式

    dtypesarr = np.zeros(data.shape[1])                         ##开辟一个数组，用于存储是否离散，是否连续数据

    if type(data).__name__ == 'DataFrame':  ##数据是pandas 数据帧格式
        for i in range(data.shape[1]):
            if data.iloc[:,i].dtypes == 'object':
                dtypesarr[i] = 1
            else:
                dtypesarr[i] = 0
    else:                                   ##数据是数组格式
        for i in range(data.shape[1]):
            try:
                sum(data[:,i])
            except:
                dtypesarr[i] = 1             ##不能求和，肯定是离散数据（str)

            if len(list(set(data[:,i]))) <= 10: ##这个只有10种取值，可以认为是离散数据
                dtypesarr[i] = 1             
    return dtypesarr

##利用训练数据创建一颗cart树
def _create_tree(data,is_dis_con_feature,depth,featurebin):  
    
    x = data[:,:-1]
    y = data[:,-1]
    
    if len(y) == 0 or np.sum(y) == len(y) or np.sum(y) == 0 or depth <= 0:     #数据为空或者标签全部为一种值,或者深度已经达到要求值
        return (np.sum(y) / len(y))                                      #标签，(概率)

    tree = {}               ##创建一个树

    best_split_feature_index,best_split_feature_point = _min_gain_gini(x,y,is_dis_con_feature,featurebin)
    split_node = x[:,best_split_feature_index]    #得到最小gain_gini,node作为父节点
    best_split_feature_isDiscrete = is_dis_con_feature[best_split_feature_index]  ##最好的分裂特征是否是离散或者连续
   
    tree['best_split_feature'] = best_split_feature_index   #将最小gini指数的特征索引存到树里面
    tree['best_split_point'] = best_split_feature_point     #记录最好的point，也就是划分点
    tree['best_split_feature_isDiscrete'] = best_split_feature_isDiscrete #记录最好的分列特征是否离散
    tree['postive_points_sum'] = np.sum(y)
    tree['negtive_points_sum'] = len(y) - np.sum(y)
#     tree['father_node'] = father_node
    
    
    left_index = []
    right_index = []                               #得到分裂后的左右子节点的索引
    for i in range(len(split_node)):
        if best_split_feature_isDiscrete == 1:              #离散
            if split_node[i] == best_split_feature_point:
                left_index.append(i)
            else:
                right_index.append(i)
        else:                                                 #连续
            if split_node[i] <= best_split_feature_point:
                left_index.append(i)
            else:
                right_index.append(i)
    
    tree['left_points_sum'] = len(left_index)      
    tree['right_points_sum'] = len(right_index)
    
    y = y[:,np.newaxis]        #将y(n) ---> y(n,1)  ###记录被分到左右节点样本的个数，用于后面剪枝求信息熵所需要的 

    if len(left_index) != 0:      
        tree['left'] = _create_tree(np.hstack((x[left_index,:],y[left_index])),is_dis_con_feature,depth-1,featurebin) #f len(right_index) != 0:
        tree['right'] = _create_tree(np.hstack((x[right_index,:],y[right_index])),is_dis_con_feature,depth-1,featurebin)  #创建右子树

    return tree

##单个样本数据的预测
def _tree_predict(tree,x):
    if type(tree).__name__ != 'dict':                  #非字典类型，就是叶子结点，直接返回叶子结点的值
        return tree
    if tree['best_split_feature_isDiscrete'] == 1:   #离散
        if x[tree['best_split_feature']] == tree['best_split_point']:
            return _tree_predict(tree['left'],x)           #左子树预测
        else:
            return _tree_predict(tree['right'],x)          #右子树预测
    else:                                          #连续
        if x[tree['best_split_feature']] <= tree['best_split_point']:
            return _tree_predict(tree['left'],x)           #左子树预测
        else:
            return _tree_predict(tree['right'],x)          #右子树预测

#####后剪枝算法

def _cost_error(tree):                       ###递归求CT
    if type(tree['left']).__name__ != 'dict':  
        pl = min(tree['left'],1-tree['left'])
        l_error = tree['left_points_sum'] / 1 * pl     #####N可以不用   N = 1
    else:
        l_error = _cost_error(tree['left'])
    
    if type(tree['right']).__name__ != 'dict':
        pr = min(tree['right'],1-tree['right'])
        r_error = tree['right_points_sum'] / 1 * pr   #####N可以不用   N= 1
    else:
        r_error = _cost_error(tree['right'])
        
    error = l_error + r_error
    
    return error 


def _tree_cost_error(tree):
     
    t_pos_sum = tree['postive_points_sum']
    t_neg_sum = tree['negtive_points_sum']
    
    t_points_sum = t_pos_sum + t_neg_sum
    
    tree['gt_list'] = []
    
    if type(tree['left']).__name__ != 'dict':        ###叶子节点
        tl_leaves_node_number = 1             
    else:                                            ###非叶子节点
        tl_leaves_node_number = tree['left']['leaves_node_number']
        tree['gt_list'] += tree['left']['gt_list']
        
    
    if type(tree['right']).__name__ != 'dict':      ###叶子节点
        tr_leaves_node_number = 1
    else:                                           ###右节点的信息熵    
        tr_leaves_node_number = tree['right']['leaves_node_number']
        tree['gt_list'] += tree['right']['gt_list']
    
    
    tree['leaves_node_number'] = tl_leaves_node_number + tr_leaves_node_number
    
    CT = _cost_error(tree)   ###计算CT
    ###以这个叶子节点为根节点的信息熵
    
    p = min(t_pos_sum,t_neg_sum) / t_points_sum     #####低的样本作为误差
    
    Ct = t_points_sum / 1 * p  ####因为要求整个训练样本的数目N，但其实对这个gt的计算以及比较不会出现问题，所以置为1
    
    T = tree['leaves_node_number']      ###叶子节点的数目
    
    tree['gt'] = (Ct - CT) / (T - 1)
    tree['gt_list'].append(tree['gt'])
    
#     print('best_split_feature:',tree['best_split_feature'],'Ct:',Ct,'CT:',CT)
    return tree


def _get_gtlist(tree):
#     从下往上求C(t) 和 C(T),其中C(t) 和 C(T) 用信息熵代替
#     以及求每个内部节点的g(t)
    if type(tree['left']).__name__ != 'dict': ###左叶子节点
        if type(tree['right']).__name__ != 'dict':      ###右叶子节点     
            tree = _tree_cost_error(tree)      ###计算gt
            tree['a'] = tree['gt']
        else:   
            tree['right'] = _get_gtlist(tree['right'])###右节点非叶子节点
            tree = _tree_cost_error(tree)
            tree['a'] = min(tree['gt'],tree['right']['a'])
    else:                                     ###左节点非叶子节点
        if type(tree['right']).__name__ != 'dict':      ###右叶子节点     
            tree['left'] = _get_gtlist(tree['left'])
            tree = _tree_cost_error(tree)
            tree['a'] = min(tree['gt'],tree['left']['a'])
        else:                                           ##右节点非叶子节点
            tree['left'] = _get_gtlist(tree['left'])         ##左节点非叶子节点
            tree['right'] = _get_gtlist(tree['right'])       ##右节点非叶子节点
            tree = _tree_cost_error(tree)
            tree['a'] = min(tree['gt'],min(tree['right']['a'],tree['left']['a']))
            
    return tree   


####寻找整颗树中每个节点的
def _find_pruning_point(tree,gt):
    if type(tree['left']).__name__ == 'dict' and tree['gt'] > gt:   ##非叶子节点：
        tree['left'] = _find_pruning_point(tree['left'],gt) 
    if type(tree['right']).__name__ == 'dict' and tree['gt'] > gt:
        tree['right'] = _find_pruning_point(tree['right'],gt)
    
    if tree['gt'] <= gt:
        t_pos_sum = tree['postive_points_sum']
        t_neg_sum = tree['negtive_points_sum']
        return t_pos_sum / (t_neg_sum + t_pos_sum)
    
    return tree

####单个gt剪枝
def _single_pruning(tree,gt):          
    newtree = deepcopy(tree)
    newtree = _find_pruning_point(newtree,gt)
    return newtree


####开始剪枝
def _start_pruning(tree):
    gt_list = tree['gt_list']
    gt_list.sort()
    
    Trees_list = []
    Trees_list.append(tree)               ####存入T0
    
    tree_temp = tree
    for item in gt_list:                  #####得到剪枝的树list
        T = _single_pruning(tree_temp,item)
        if type(T).__name__ != 'dict':    ####如果返回的是一个单独的节点 ，这个节点是根节点
            break
        
        Trees_list.append(T)
        
        tree_temp = T     ####用剪枝后生成的树用来新一轮剪枝
        
    return Trees_list

def _post_predict(tree,valx):
    valx = np.array(valx)

    py = np.zeros((len(valx)))

    for i in range(len(valx)):
        x = valx[i]
        py[i] = _tree_predict(tree,x)  
    return py

def _post_pruning(tree,valx,valy):
    
    tree = _get_gtlist(tree)
    trees = _start_pruning(tree)   ###得到T0,T1,T2.....Tn
       
    valscore = []
    
    for t in trees:
        py = _post_predict(t,valx)
        py[py>= 0.5] = 1
        py[py<0.5] = 0
        score = accuracy_score(py,valy)
        valscore.append(score)
        
    maxscore_index = 0
    maxscore = valscore[0]                            ###寻找最大值索引
    for i in range(len(valscore)):
        if maxscore <= valscore[i]:
            maxscore = valscore[i]
            maxscore_index = i
    
    return trees[maxscore_index]

class CartClassifier:
    
    ##参数初始化
    def __init__(self,maxdepth = 6,maxfeaturebin = 300):
        self.tree = {}
        self.features_len = 0
        self.maxdepth = maxdepth
        self.maxfeaturebin = maxfeaturebin
    
    ##模型训练
    def fit(self,X,y):
        assert len(X) == len(y)   #检查是否是同一个维度
        assert len(list(set(y))) == 2  #检查是否是二分类
        
        is_dis_con_feature = _is_dis_con_feature(X)       ##是否是连续或者离散变量
        
        self.features_len = X.shape[1]
        
        X = np.array(X)
        y = np.array(y)
        
        y = y[:,np.newaxis]
        
        train = np.hstack((X,y))
        tree = _create_tree(train,is_dis_con_feature,self.maxdepth,self.maxfeaturebin) 
        
        valX,valy =X,y
        self.tree = _post_pruning(tree,valX,valy)     #####后剪枝
        
    #全部样本数据的预测
    def predict(self,X):
        X = np.array(X)
        
        assert X.shape[1] == self.features_len         #检查是否是数据维度是否一样

        py = np.zeros((len(X)))
        
        for i in tqdm(range(len(X)),desc = 'predict:'):
            x = X[i]
            py[i] = _tree_predict(self.tree,x)  
        return py