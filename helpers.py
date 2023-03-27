# -*- coding: utf-8 -*-
"""

data 
@author: 86158


"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

## Model evaluaiton_measures
def evaluation_measures(y_test,prob):
    "AUC_ROC:"
    #auc_score = roc_auc_score(y_truth,y_pred)
    tpr_list = []
    tpr_list = []
    mean_fpr = np.linspace(0, 1, 100)
    fpr,tpr,threshold = roc_curve(y_test, prob)
    tpr_list.append(interp(mean_fpr, fpr, tpr))
    tpr_list[-1][0] = 0.0
    auc_roc= auc(fpr,tpr)
    
    "AUC_PR:"
    auc_pr = average_precision_score(y_test,prob)
    
    "BS+ and BS-:"
    min_class = []
    maj_class = []
    for i in range(len(y_test)):
        if y_test[i]==1:
            min_class.append((prob[i]-1)*(prob[i]-1))
        else:
            maj_class.append((prob[i]*prob[i]))
    bs_min=sum(min_class)/len(min_class)
    bs_maj=sum(maj_class)/len(maj_class)
   
    
    return [auc_roc,auc_pr,bs_min,bs_maj]

def other_evalution_measures(y_test,prob):
    conf_mat1=confusion_matrix(y_test,prob)
    TN=conf_mat1[0,0]
    #TP=conf_mat1[1,1]
    FP=conf_mat1[0,1]
    #FN=conf_mat1[1,0]
    R=recall_score(y_test,prob)
    P=precision_score(y_test, prob) 
    sp=TN / float(TN+FP)
    G=np.sqrt(R*sp)
    F=f1_score(y_test,prob)
    
    return [R,P,G,F]

# ## Preprocessing
    
def load_data(filepath):
    df=pd.read_csv(filepath)
    thr=len(df)*0.6
    "delete featrues with a miss rate over 60%"
    df.dropna(axis=1,  thresh=thr, subset=None, inplace=False)
    
    return df

def imputer(dataframe,cat_list,num_list):
    df=dataframe
    col_name= df.columns
    cat_name=cat_list
    num_name=num_list
    
    #Categorical variable:
    df_cat=df[cat_name].head()
    imp_cat=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    df_cat=imp_cat.fit_transform(df_cat)
        
    #Numeric variable:
    df_num=df[num_name].head()
    imp_num=SimpleImputer(missing_values=np.nan,strategy='mean')
    df_num=imp_num.fit_transform(df_num)
        
    df_new=pd.concat([df_cat,df_num],axis=1)
    n_features=df_new.shape[1]
    

    return [df_cat,df_num,df_new,n_features,col_name]

def transform(df_num,df_cat,df_new):
    
    encoder1= preprocessing.MinMaxScaler()
    encoder2= preprocessing.OneHotEncoder()
    df1=np.array(df_num)
    df1=encoder1.fit_transform(df1)
    df2=np.array(df_cat)
    df2=encoder2.fit_transform(df2)
    y=df_new['label']
    X=np.concatenate((df1,df2),axis=0)
    
    return [X,y]

def datasets_imbalance_ratio(file_path,IR):
    """
    obtain datasets with different imbalance ratios
    IR:imbalance_ratio

    """
    df=pd.read_csv(file_path)
    min_df=df.loc[df['label']==1]
    maj_df=df.loc[df['label']==0]
    maj_num=len(maj_df)
    sample_num=round(maj_num/IR)
    sample_min=min_df.sample(n=sample_num,random_state=43,axis=0)
    dataset_new=np.concatenate((maj_df,sample_min),axis=0)
    
    
    return  dataset_new
    
    
    
    


