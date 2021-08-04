import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import itertools
from scipy import interp
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#绘制图形
def make_facies_log_plot(logs,
                         facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D'],
                         labels = [' SS ', 'CSiS', 'FSiS','SiSh', ' MS ', ' WS ', ' D  ', ' PS ', ' BS ']):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    f, ax = plt.subplots(nrows=1, ncols=8, figsize=(10, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='yellow')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    ax[5].plot(logs.NM_M, logs.Depth, '-', color='orange')
    ax[6].plot(logs.RELPOS, logs.Depth, '-', color='0.5')
    im=ax[7].imshow(cluster, interpolation='none', aspect='auto',cmap=cmap_facies,vmin=1,vmax=9)
    divider = make_axes_locatable(ax[7])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((11*' ').join(labels))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')   
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel("NM_M")
    ax[5].set_xlim(logs.NM_M.min(),logs.NM_M.max())
    ax[6].set_xlabel("RELPOS")
    ax[6].set_xlim(logs.RELPOS.min(),logs.RELPOS.max())
    ax[7].set_xlabel('Facies')
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([]); ax[6].set_yticklabels([])
    ax[7].set_yticklabels([]); ax[7].set_xticklabels([])
    f.suptitle('%s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.9)
    plt.show()
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes)-0.5,-0.5)
    plt.xlim(-0.5,len(classes)-0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')   
    
class DataFitting:
    def __init__(self,data,dispersed=False,n_components=5,dispersed_label = {},distribute_title = None,feature_name = None):
        self.data = data
        self.feature_name = feature_name
        self.dispersed = dispersed
        self.distribute_title = distribute_title
        #离散处理
        if dispersed:
            self.data_value_num_dict = dict(Counter(data).items())
            self.prob_dict = dict([(d_k,self.data_value_num_dict[d_k]/len(data)) for d_k in self.data_value_num_dict]) 
            self.dispersed_label = dispersed_label
        #连续处理    
        else:
            self.n_components = n_components
            gmm = GaussianMixture(n_components=n_components,random_state = 0).fit(data.reshape(-1,1))
            self.weights = gmm.weights_
            self.covariances = gmm.covariances_.flatten()
            self.means = gmm.means_.flatten()
            
    def show_distribute(self,bins=20,show=True):
        if self.dispersed:
            k_v = list(zip(*sorted(self.prob_dict.items())))
            k_list = [self.dispersed_label[item] if item in self.dispersed_label else str(item) for item in k_v[0]]
            v_list = k_v[1]
            plt.bar(range(1,len(v_list) + 1),v_list,width=0.4,color = 'cornflowerblue')
            plt.xticks(range(1,len(v_list) + 1), k_list)
            plt.xlim(0,len(v_list) + 1)
        else:    
            f_data = self.data.flatten()
            sns.distplot(f_data,kde=False,bins=bins,rug=True,norm_hist = True,color = '0.8')
            line_spc = np.linspace(np.min(f_data),np.max(f_data),100)
            prob_com = np.zeros((100,))
            for index in range(len(self.weights)):
                prob = (1 / np.sqrt(2 * np.pi * self.covariances[index])) * np.exp(- np.power((line_spc - self.means[index]),2) / (2 * self.covariances[index])) * self.weights[index]
                prob_com += prob
                plt.plot(line_spc,prob,'--', linewidth=1.5,label='component ' + str(index + 1))
            plt.plot(line_spc,prob_com,'-',linewidth=2,label='curve fitting',color = 'cornflowerblue')  
            if len(self.weights) > 1:
                plt.legend(loc = 'best', prop = {'size':8})
            if self.feature_name is not None:
                plt.xlabel(self.feature_name)    
        if self.distribute_title is not None:
            plt.title(self.distribute_title)
        if show:    
            plt.show()
        
    def calcu_prob(self,value):
        if self.dispersed:
            if value in self.prob_dict:
                return self.prob_dict[value]
            else:
                return 0
        else:    
            prob = 0
            for index in range(len(self.weights)):
                prob += (1 / np.sqrt(2 * np.pi * self.covariances[index])) * np.exp(- np.power((value - self.means[index]),2) / (2 * self.covariances[index])) * self.weights[index]
            return prob

def show_disribute_for_all_feature_in_on_facie(df,facie,n_components = 5,features = ['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS','Formation Index'],dispersed_features = ['NM_M','Formation Index'],
                                              dispersed_features_label_dic = {'Formation Index':{1: 'A1 LM',2: 'A1 SH',3: 'B1 LM',4: 'B1 SH',5: 'B2 LM',6: 'B2 SH',7: 'B3 LM',8: 'B3 SH',9: 'B4 LM',10: 'B4 SH',11: 'B5 LM',12: 'B5 SH',13: 'C LM',14: 'C SH'},'NM_M':{1:'NM_M=1',2:'NM_M=2'}}): 
    df_one_facie = df[df["Facies"] == facie]
    num = len(features)
    row_num = 1
    column_num = num
    floor = int(np.sqrt(num))
    for i in range(floor,0,-1):
        if num%i == 0:
            row_num = i
            column_num = int(num/i)
            break
    plt.figure(figsize = (4 * column_num,3.5 * row_num))        
    for ind,feature in enumerate(features):
        data = df_one_facie[feature].values 
        data_fitting = DataFitting(data,n_components=n_components,feature_name = feature) if feature not in dispersed_features else DataFitting(data,dispersed=True,dispersed_label = dispersed_features_label_dic[feature] if feature in dispersed_features_label_dic else {},feature_name = feature)
        plt.subplot(row_num,column_num,ind + 1)
        data_fitting.show_distribute(show=False)
    plt.show()
        
def get_data_fitting_dict(train_features_data,train_labels,features=['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','Formation Index','RELPOS'],dispersed_features=['NM_M','Formation Index'],n_components=5):
    data_fitting_map = {}
    for feature in features:
        for facie_index in np.unique(train_labels):
            data = train_features_data[train_labels == facie_index][feature].values 
            data_fitting_map[feature + '-facie' + str(facie_index)] = DataFitting(data,n_components=n_components) if feature not in dispersed_features else DataFitting(data,dispersed=True)
    return data_fitting_map    

def predict_proba(feature_df,data_fitting_map,train_labels):
    features = [feature for feature in feature_df]
    res = []
    label_ind_lis = np.unique(train_labels)
    for index in range(len(feature_df)):
        one_data = feature_df.loc[feature_df.index[index],:]
        prob_lis = np.ones((len(label_ind_lis),))
        for facie_ind in label_ind_lis:
            for feature in features:
                key = feature + '-facie' + str(facie_ind)
                if key in data_fitting_map:
                    prob_lis[facie_ind - 1] *= data_fitting_map[key].calcu_prob(one_data[feature])
        res.append(prob_lis)   
    counter = Counter(train_labels)
    p = np.array(res) * [counter[i]/len(train_labels) for i in label_ind_lis] 
    return  p / (np.sum(p,axis = 1).reshape(-1,1))       
    
class FeatureGroupDataFitting:
    def __init__(self,df,n_components=5,x_axis_dict = None,y_axis_dict = None):
        feature_name_lis = [feature_name for feature_name in df]
        self.x_feature_name = feature_name_lis[0]
        self.y_feature_name = feature_name_lis[1]
        self.x_data = df[self.x_feature_name].values
        self.y_data = df[self.y_feature_name].values
        self.x_axis_dict = x_axis_dict
        self.y_axis_dict = y_axis_dict
        self.n_components = n_components
        #两个特征都是连续特征
        if self.x_axis_dict is None and self.y_axis_dict is None:
            if n_components > len(df):
                n_components = len(df)
            gmm = GaussianMixture(n_components=n_components,random_state = 0).fit(df)
            self.weights = gmm.weights_
            self.covariances = gmm.covariances_
            self.means = gmm.means_
        #两个特征都是离散特征
        elif self.x_axis_dict is not None and self.y_axis_dict is not None:
            prob_dict = {}
            for i in [k_x for k_x,_ in self.x_axis_dict.items()]:
                for j in [k_y for k_y,_ in self.y_axis_dict.items()]:
                    prob_dict[(int(i),int(j))] = 0
            for ind,i in  enumerate(self.x_data):
                prob_dict[(int(i),int(self.y_data[ind]))] += 1
            for k,v in prob_dict.items():
                prob_dict[k] = v/len(self.x_data)
            self.prob_dic = prob_dict
        #x轴是连续特征，y轴是离散特征
        elif self.x_axis_dict is None and self.y_axis_dict is not None:
            dispersed_unique = np.unique(self.y_data)
            self.dispersed_unique = dispersed_unique
            gmm_dic = {}
            for dispersed_value in dispersed_unique:
                data_to_fit = self.x_data[np.argwhere(self.y_data == dispersed_value).flatten()]
                if len(data_to_fit) == 1:
                    data_to_fit = np.array([data_to_fit[0] for i in range(2)])
                if n_components > len(data_to_fit):
                    n_components = len(data_to_fit)
                gmm = GaussianMixture(n_components=n_components,random_state = 0).fit(data_to_fit.reshape(-1,1))
                gmm_dic[dispersed_value] = (data_to_fit,gmm)
            self.gmm_dic = gmm_dic    
        else:    
            print('warning')  
    def show_distribute(self,ax = None):
        if ax is None:
            plt.figure(figsize = (10,8))
            ax = plt.axes(projection='3d')
        #两个特征都是连续特征
        if self.x_axis_dict is None and self.y_axis_dict is None:
            x_line_spc = np.linspace(np.min(self.x_data),np.max(self.x_data),100)
            y_line_spc = np.linspace(np.min(self.y_data),np.max(self.y_data),100)
            X, Y = np.meshgrid(x_line_spc,y_line_spc)
            Z = np.zeros(X.shape)
            for index in range(len(self.weights)):
                covariances = self.covariances[index]
                normal = np.linalg.norm(covariances)
                inv = np.linalg.inv(covariances)
                means = self.means[index]
                weight = self.weights[index]
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        data = np.array([X[i][j],Y[i][j]]) - means
                        Z[i][j] += weight * (1 / (2 * np.pi * np.sqrt(normal))) * np.exp(-(np.dot(np.dot(data.reshape(1,-1),inv),data.reshape(-1,1)).flatten()[0])/2)
            ax.plot_surface(X,Y,Z,alpha=0.5,cmap='coolwarm',rstride = 1, cstride = 1)     #生成表面， alpha 用于控制透明度
            ax.contour(X,Y,Z,zdir='z', offset=0,cmap="coolwarm")  #生成z方向投影，投到x-y平面
            ax.plot_wireframe(X, Y, Z, color='black',alpha=0.02,rstride = 1, cstride = 1)
            ax.set_xlabel(self.x_feature_name)
            ax.set_ylabel(self.y_feature_name)
        #两个特征都是离散特征
        elif self.x_axis_dict is not None and self.y_axis_dict is not None:
            # setup the figure and axes
            x_value_lis = sorted([k_x for k_x,_ in self.x_axis_dict.items()])
            y_value_lis = sorted([k_y for k_y,_ in self.y_axis_dict.items()])
            _xx, _yy = np.meshgrid(x_value_lis, y_value_lis)
            x, y = _xx.ravel(), _yy.ravel()#ravel扁平化
            # 函数
            top = []
            for i in range(len(x)):
                top.append(self.prob_dic[(x[i],y[i])])       
            bottom = np.zeros_like(top)#每个柱的起始位置
            width = depth = 0.4 #x,y方向的宽厚
            ax.bar3d(x - 0.2, y - 0.6, bottom, width, depth, top,alpha = 0.5)
            x_values = [k_x for k_x,_ in sorted(self.x_axis_dict.items())]
            y_values = [k_y for k_y,_ in sorted(self.y_axis_dict.items())]
            x_labels = [v_x for _,v_x in sorted(self.x_axis_dict.items())]
            y_labels = [v_y for _,v_y in sorted(self.y_axis_dict.items())]
            plt.xticks(x_values,x_labels,rotation='vertical')
            plt.yticks(y_values,y_labels,rotation='vertical')
        #x轴是连续特征，y轴是离散特征
        elif self.x_axis_dict is None and self.y_axis_dict is not None:
            for item in self.dispersed_unique:
                data_to_fit,gmm = self.gmm_dic[item]
                line_spc = np.linspace(np.min(data_to_fit),np.max(data_to_fit),100)
                prob_com = np.zeros((100,))
                weights = gmm.weights_
                covariances = gmm.covariances_.flatten()
                means = gmm.means_.flatten()
                for index in range(len(weights)):
                    prob = (1 / np.sqrt(2 * np.pi * covariances[index])) * np.exp(- np.power((line_spc - means[index]),2) / (2 * covariances[index])) * weights[index]
                    prob_com += prob  
                prob_com *= len(data_to_fit)/len(self.x_data)
                ax.plot(line_spc, [item for i in range(100)], prob_com,linewidth=2)
            y_values = [k_y for k_y,_ in sorted(self.y_axis_dict.items())]
            y_labels = [v_y for _,v_y in sorted(self.y_axis_dict.items())]
            plt.yticks(y_values,y_labels,rotation='vertical')
            ax.set_xlabel(self.x_feature_name)
        else:    
            print('warning')
    def calcu_prob(self,value_tuple):
        #两个特征都是连续特征
        if self.x_axis_dict is None and self.y_axis_dict is None:
            prob = 0    
            for index in range(len(self.weights)):
                covariances = self.covariances[index]
                normal = np.linalg.norm(covariances)
                inv = np.linalg.inv(covariances)
                means = self.means[index]
                weight = self.weights[index]
                data = np.array(value_tuple) - means
                prob += weight * (1 / (2 * np.pi * np.sqrt(normal))) * np.exp(-(np.dot(np.dot(data.reshape(1,-1),inv),data.reshape(-1,1)).flatten()[0])/2)
            return prob
        #两个特征都是离散特征
        elif self.x_axis_dict is not None and self.y_axis_dict is not None:
            if value_tuple in self.prob_dic:
                return self.prob_dic[value_tuple]
            else:
                return 0
        elif self.x_axis_dict is None and self.y_axis_dict is not None:
            if value_tuple[1] not in self.gmm_dic:
                return 0
            data_to_fit,gmm = self.gmm_dic[value_tuple[1]]
            prob = 0
            weights = gmm.weights_
            covariances = gmm.covariances_.flatten()
            means = gmm.means_.flatten()
            for index in range(len(weights)):
                prob += (1 / np.sqrt(2 * np.pi * covariances[index])) * np.exp(- np.power(( value_tuple[0] - means[index]),2) / (2 * covariances[index])) * weights[index]
            prob *= len(data_to_fit)/len(self.x_data)
            return prob 
        else:    
            print('warning')    
            
def show_disribute_for_all_feature_group_in_on_facie(df,facie,n_components = 5,save_name = "fig.png",
                        continuous_features_use_for_group = ['GR','ILD_log10','DeltaPHI','PHIND','PE','RELPOS'],
                        dispersed_features_use_for_group = ['NM_M','Formation Index'],
                        dispersed_features_label_dic = {'Formation Index':{1: 'A1 LM',2: 'A1 SH',3: 'B1 LM',4: 'B1 SH',5: 'B2 LM',6: 'B2 SH',7: 'B3 LM',8: 'B3 SH',9: 'B4 LM',10: 'B4 SH',11: 'B5 LM',12: 'B5 SH',13: 'C LM',14: 'C SH'},'NM_M':{1:'NM_M=1',2:'NM_M=2'}}): 
    feature_groups = []
    for index,fearture in enumerate(continuous_features_use_for_group):
        for index_inner in range(index + 1,len(continuous_features_use_for_group)):
            feature_groups.append((fearture,continuous_features_use_for_group[index_inner]))
    for index,fearture in enumerate(dispersed_features_use_for_group):
        for index_inner in range(index + 1,len(dispersed_features_use_for_group)):
            feature_groups.append((fearture,dispersed_features_use_for_group[index_inner]))
    for item1 in continuous_features_use_for_group:
        for item2 in dispersed_features_use_for_group:
            feature_groups.append((item1,item2))
    df_one_facie = df[df["Facies"] == facie]
    data_fitting_lis = []
    for ind,feature_group in enumerate(feature_groups):
        group_df = df_one_facie[list(feature_group)] 
        data_fitting = None
        if feature_group[0] in continuous_features_use_for_group and feature_group[1] in continuous_features_use_for_group:
            data_fitting = FeatureGroupDataFitting(group_df,n_components=n_components)
        elif feature_group[0] in dispersed_features_use_for_group and feature_group[1] in dispersed_features_use_for_group:    
            data_fitting = FeatureGroupDataFitting(group_df,n_components=n_components,x_axis_dict = dispersed_features_label_dic[feature_group[0]],y_axis_dict = dispersed_features_label_dic[feature_group[1]])
        elif feature_group[0] in continuous_features_use_for_group and feature_group[1] in dispersed_features_use_for_group:  
            data_fitting = FeatureGroupDataFitting(group_df,n_components=n_components,y_axis_dict = dispersed_features_label_dic[feature_group[1]])
        else:
            print("warning")
        data_fitting_lis.append(data_fitting)
    
    num = len(data_fitting_lis)
    row_num = 1
    column_num = num
    floor = int(np.sqrt(num))
    for i in range(floor,0,-1):
        if num%i == 0:
            row_num = i
            column_num = int(num/i)
            break
    fig = plt.figure(figsize = (15 * column_num,15 * row_num))        
    for ind,data_fitting in enumerate(data_fitting_lis):
        ax = fig.add_subplot(row_num,column_num,ind + 1, projection='3d')
        data_fitting.show_distribute(ax)
    plt.savefig(save_name)    
    plt.show()  
    
def get_feature_group_data_fitting_dict(train_features_data,train_labels,n_components=5,continuous_features_use_for_group = ['GR','ILD_log10','DeltaPHI','PHIND','PE','RELPOS'],
                        dispersed_features_use_for_group = ['NM_M','Formation Index'],
                        dispersed_features_label_dic = {'Formation Index':{1: 'A1 LM',2: 'A1 SH',3: 'B1 LM',4: 'B1 SH',5: 'B2 LM',6: 'B2 SH',7: 'B3 LM',8: 'B3 SH',9: 'B4 LM',10: 'B4 SH',11: 'B5 LM',12: 'B5 SH',13: 'C LM',14: 'C SH'},'NM_M':{1:'NM_M=1',2:'NM_M=2'}}): 
    data_fitting_map = {}
    for facie_index in np.unique(train_labels):
        for index,fearture in enumerate(continuous_features_use_for_group):
            for index_inner in range(index + 1,len(continuous_features_use_for_group)):
                feature_group = (fearture,continuous_features_use_for_group[index_inner])
                data = train_features_data[train_labels == facie_index][list(feature_group)]
                data_fitting_map[str(feature_group) + '-facie' + str(facie_index)] = FeatureGroupDataFitting(data,n_components=n_components)
        for index,fearture in enumerate(dispersed_features_use_for_group):
            for index_inner in range(index + 1,len(dispersed_features_use_for_group)):
                feature_group = (fearture,dispersed_features_use_for_group[index_inner])
                data = train_features_data[train_labels == facie_index][list(feature_group)]
                data_fitting_map[str(feature_group) + '-facie' + str(facie_index)] = FeatureGroupDataFitting(data,n_components=n_components,x_axis_dict = dispersed_features_label_dic[feature_group[0]],y_axis_dict=dispersed_features_label_dic[feature_group[1]])
        for fearture in continuous_features_use_for_group:
            for inner_fearture in dispersed_features_use_for_group:
                feature_group = (fearture,inner_fearture)
                data = train_features_data[train_labels == facie_index][list(feature_group)]
                data_fitting_map[str(feature_group) + '-facie' + str(facie_index)] = FeatureGroupDataFitting(data,n_components=n_components,y_axis_dict=dispersed_features_label_dic[feature_group[1]])
    return data_fitting_map    

def predict_proba_feature_group(feature_df,data_fitting_map,train_labels,continuous_features_use_for_group = ['GR','ILD_log10','DeltaPHI','PHIND','PE','RELPOS'],
                        dispersed_features_use_for_group = ['NM_M','Formation Index']):
    feature_groups = []
    for index,fearture in enumerate(continuous_features_use_for_group):
        for index_inner in range(index + 1,len(continuous_features_use_for_group)):
            feature_group = (fearture,continuous_features_use_for_group[index_inner])
            feature_groups.append(feature_group)
    for index,fearture in enumerate(dispersed_features_use_for_group):
        for index_inner in range(index + 1,len(dispersed_features_use_for_group)):
            feature_group = (fearture,dispersed_features_use_for_group[index_inner])
            feature_groups.append(feature_group)
    for fearture in continuous_features_use_for_group:
        for inner_fearture in dispersed_features_use_for_group:
            feature_group = (fearture,inner_fearture)
            feature_groups.append(feature_group)    
    res = []
    label_ind_lis = np.unique(train_labels)
    for index in range(len(feature_df)):
        one_data = feature_df.loc[feature_df.index[index],:]
        prob_lis = np.ones((len(label_ind_lis),))
        for facie_ind in label_ind_lis:
            for feature_group in feature_groups:
                key = str(feature_group) + '-facie' + str(facie_ind)
                if key in data_fitting_map:
                    prob_lis[facie_ind - 1] *= data_fitting_map[key].calcu_prob((one_data[feature_group[0]],one_data[feature_group[1]]))
        res.append(prob_lis)   
    counter = Counter(train_labels)
    p = np.array(res) * [counter[i]/len(train_labels) for i in label_ind_lis] 
    return  p / (np.array([1 if i==0 else i for i in np.sum(p,axis = 1)]).reshape(-1,1))                
            
def display_cm(cm, labels, hide_zeros=False,display_metrics=False):
    precision = np.diagonal(cm) / cm.sum(axis=0).astype('float')
    recall = np.diagonal(cm) / cm.sum(axis=1).astype('float')
    F1 = 2 * (precision * recall) / (precision + recall)
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1[np.isnan(F1)] = 0
    total_precision = np.sum(precision * cm.sum(axis=1)) / cm.sum(axis=(0, 1))
    total_recall = np.sum(recall * cm.sum(axis=1)) / cm.sum(axis=(0, 1))
    total_F1 = np.sum(F1 * cm.sum(axis=1)) / cm.sum(axis=(0, 1))
    # print total_precision
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + " Pred", end=' ')
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=' ')
    print("%{0}s".format(columnwidth) % 'Total')
    print("    " + " True")
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeros:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            print(cell, end=' ')
        print("%{0}d".format(columnwidth) % sum(cm[i, :]))
    if display_metrics:
        print()
        print("Precision", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % precision[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_precision)
        print("   Recall", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % recall[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_recall)
        print("       F1", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % F1[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_F1)     

def show_roc_detail(true_labels_ohv,predict_proba,facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D'],labels = [' SS ', 'CSiS', 'FSiS','SiSh', ' MS ', ' WS ', ' D  ', ' PS ', ' BS ']):
    plt.figure(figsize=(8,6))
    n_classes = true_labels_ohv.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(true_labels_ohv[:, i], predict_proba[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(true_labels_ohv.ravel(), predict_proba.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw=1.5
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,alpha = 0.65, color = facies_colors[i],
                 label='{0} (AUC = {1:0.2f})'
                 ''.format(labels[i], roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2.5)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.tick_params(labelsize=12)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=10)
    plt.show()      

def load_data(file_name,facies_name = [' SS ', 'CSiS', 'FSiS','SiSh', ' MS ', ' WS ', ' D  ', ' PS ', ' BS '],facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']):
    training_data_fram = pd.read_csv(file_name)
    formation = np.unique(training_data_fram["Formation"].values)
    formation_dict = dict(zip(formation,range(1,1+len(formation))))
    facies_dict = dict(zip(np.unique(training_data_fram["Facies"].values),facies_name))
    formation_index_column = [formation_dict[formation_str] for formation_str in training_data_fram["Formation"].values]
    facies_name_column = [facies_dict[facie] for facie in training_data_fram["Facies"].values]
    training_data_fram['Formation Index'] = formation_index_column
    training_data_fram['Facies Name'] = facies_name_column
    return training_data_fram[['Facies','Facies Name','Formation','Formation Index','Well Name','Depth','GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']]  

def show_wells(training_data_fram): 
    for well_name in np.unique(training_data_fram['Well Name']):
        df_single_well = training_data_fram[training_data_fram['Well Name'] == well_name]
        make_facies_log_plot(df_single_well)    

def show_train_test_distribute(train_labels,test_labels,facies_name = [' SS ', 'CSiS', 'FSiS','SiSh', ' MS ', ' WS ', ' D  ', ' PS ', ' BS ']):
    train_count = list(zip(*sorted(Counter(train_labels).items())))[1]
    test_count = list(zip(*sorted(Counter(test_labels).items())))[1]
    plt.bar(facies_name, test_count, 0.4,label="test dataset")
    plt.bar(facies_name, train_count, 0.4, bottom=test_count,label="training dataset")
    plt.legend()
    plt.show()      
        
def train_test_split_and_show(training_data_fram,test_size = 0.3):
    all_features = training_data_fram[['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS','Formation Index']]
    all_labels = training_data_fram['Facies']
    train_features,test_features,train_labels,test_labels = train_test_split(all_features,all_labels,test_size = test_size,stratify = all_labels,random_state=0)
    show_train_test_distribute(train_labels,test_labels)
    return train_features,test_features,train_labels,test_labels

def show_result(test_labels,predict_proba,facies_name = [' SS ', 'CSiS', 'FSiS','SiSh', ' MS ', ' WS ', ' D  ', ' PS ', ' BS ']):   
    cv_conf = confusion_matrix(test_labels, np.argmax(predict_proba,axis = 1) + 1)
    display_cm(cv_conf, facies_name,display_metrics=True, hide_zeros=True)
    plot_confusion_matrix(cv_conf,facies_name,title = 'Confusion Matrix')
    show_roc_detail(to_categorical(test_labels - 1),predict_proba)  

def simple_nb_train_and_show_result(train_features,test_features,train_labels,test_labels):
    model = GaussianNB()
    model.fit(train_features, train_labels)
    predict_proba = model.predict_proba(test_features)
    show_result(test_labels,predict_proba)
    
def gmm_nb_and_show_result(train_features,test_features,train_labels,test_labels,n_components=5):    
    fit_map = get_data_fitting_dict(train_features,train_labels,n_components=n_components)  
    predict_proba_ = predict_proba(test_features,fit_map,train_labels)
    show_result(test_labels,predict_proba_)    

def feature_group_bn_train_and_show_result(train_features,test_features,train_labels,test_labels,n_components=5):    
    fit_map = get_feature_group_data_fitting_dict(train_features,train_labels,n_components=n_components)  
    predict_proba_ = predict_proba_feature_group(test_features,fit_map,train_labels)
    show_result(test_labels,predict_proba_)