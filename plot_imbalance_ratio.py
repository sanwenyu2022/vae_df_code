# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:50:19 2022

@author: 86158
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
fig = plt.figure()
plt.rcParams['figure.figsize'] = (7.8, 7.8) 
plt.rcParams['image.interpolation'] = 'nearest' 
plt.rcParams['image.cmap'] = 'gray' # 设
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['savefig.dpi'] =600 
plt.rcParams['figure.dpi'] =100

names = ['4', '6', '8', '10', '20','40','60','80','100']
x = range(len(names))
x=[2, 3, 4, 5, 6, 7, 8, 9,10]
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 7,}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 7,}
#1.
plt.figure
plt.subplot(431)
inputfile1="D:/results v3/IR/ROC/lending.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([95,100])
plt.xlim([1.9,10.3])
plt.ylabel(u'AUC-ROC (%)',font1,labelpad=-0.2)
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
plt.xticks(x,names)
ax = plt.gca()
ax.set_title('Lending',font1,pad=3)     
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 
 
#2.

plt.subplot(432)
inputfile1="D:/results v3/IR/ROC/prosper.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([50,80])
plt.xlim([1.9,10.3])
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
plt.xticks(x,names)
ax = plt.gca()
ax.set_title('Prosper',font1,pad=3)    
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#3.
plt.subplot(433)
inputfile1="D:/results v3/IR/ROC/home.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([50,80])
plt.xlim([1.9,10.3])

plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')

plt.xticks(x,names)
ax = plt.gca()
ax.set_title('Home',font1,pad=3)

plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#4
plt.subplot(434)
inputfile1="D:/results v3/IR/PR/lending.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([80,100])
plt.xlim([1.9,10.3])
plt.ylabel(u'AUC-PR (%)',font1,labelpad=-0.2)
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')

plt.xticks(x,names)
ax = plt.gca()
ax.set_title('Lending',font1,pad=3)
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#5
plt.subplot(435)
inputfile1="D:/results v3/IR/PR/prosper.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([0,45])
plt.xlim([1.9,10.3])

plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
#ax1.legend(prop=font2) # 让图例生效
plt.xticks(x,names)
ax = plt.gca()
ax.set_title('Prosper',font1,pad=3)

plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#6
plt.subplot(436)
inputfile1="D:/results v3/IR/PR/home.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([0,45])
plt.xlim([1.9,10.3])

plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
#ax1.legend(prop=font2) # 让图例生效
plt.xticks(x,names)
ax = plt.gca()
ax.set_title('Home',font1,pad=3) 
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#7
plt.subplot(4,3,7)
inputfile1="D:/results v3/IR/BS2/lending.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([0,30])
plt.xlim([1.9,10.3])
plt.ylabel(u'BS$^+$ (%)',font1,labelpad=0)

plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
plt.xticks(x,names)
ax = plt.gca() 
ax.set_title('Lending',font1,pad=3)     
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#8
plt.subplot(4,3,8)
inputfile1="D:/results v3/IR/BS2/prosper.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([15,100])
plt.xlim([1.9,10.3])
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
plt.xticks(x,names)
ax = plt.gca() 
ax.set_title('Prosper',font1,pad=3)    
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#9
plt.subplot(4,3,9)
inputfile1="D:/results v3/IR/BS2/home.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([15,100])
plt.xlim([1.9,10.3])
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
plt.xticks(x,names)
ax = plt.gca()
ax.set_title('Home',font1,pad=3)      
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 


#10
plt.subplot(4,3,10)
inputfile1="D:/results v3/IR/BS1/lending.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([0,5])
plt.xlim([1.9,10.3])
plt.ylabel(u'BS$^-$ (%)',font1)
plt.xlabel(u'IR',font1)
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')

plt.xticks(x,names)
ax = plt.gca() 
ax.set_title('Lending',font1,pad=3)     

plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#11
plt.subplot(4,3,11)
inputfile1="D:/results v3/IR/BS1/prosper.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([0,25])
plt.xlim([1.9,10.3])

plt.xlabel(u'IR',font1)
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.25,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.25,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.25,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.25,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.25,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.25,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.25,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.25,ms=3,label='VAE-DF')
plt.xticks(x,names)
ax = plt.gca()  
ax.set_title('Prosper',font1,pad=3)    
plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 

#12
plt.subplot(4,3,12)
inputfile1="D:/results v3/IR/BS1/home.csv"
data=pd.read_csv(inputfile1)
data=np.array(data,dtype=np.float64)
y1 =data[:,0]#Pure-DF
y2 =data[:,1]#ros-DF
y3 =data[:,2]#smoteDF
y4 =data[:,3]#blsmote-DF
y5=data[:,4]#adaDF
y6=data[:,5]#gan-DF
y7=data[:,6]#wcgan-DF
y8=data[:,7]#vae-df

plt.ylim([0,25])
plt.xlim([1.9,10.3])
plt.xlabel(u'IR',font1)
plt.plot(x, y1*100, marker='H',c='k',mfc='y',mew='0.2',linewidth=0.3,ms=3,label='Pure-DF')
plt.plot(x,y2*100,marker='v',c='k',mfc='g',mew='0.2',linewidth=0.3,ms=3,label='ROS-DF')
plt.plot(x,y3*100,marker='<',c='k',mfc='b',mew='0.2',linewidth=0.3,ms=3,label='SMOTE-DF')
plt.plot(x,y4*100,marker='>',c='k',mfc='c',mew='0.2',linewidth=0.3,ms=3,label='BLSMOTE-DF')
plt.plot(x,y5*100,marker='s',c='k',mfc='magenta',mew='0.2',linewidth=0.3,ms=3,label='ADASYN-DF')
plt.plot(x, y6*100, marker='d',c='k',mfc='darkblue',mew='0.2',linestyle='--' ,linewidth=0.3,ms=2,label='GAN-DF')
plt.plot(x,y7*100,marker='.',c='k',mfc='darkorange',mew='0.2',linestyle=':',linewidth=0.3,ms=3,label='cWGAN-DF')
plt.plot(x, y8*100, marker='*',c='k',mfc='r',mew='0.2',linewidth=0.3,ms=3,label='VAE-DF')
plt.xticks(x,names)
ax = plt.gca() 
ax.set_title('Home',font1,pad=3)     

plt.tick_params(labelsize=7)
plt.tick_params(labelsize=7) 



plt.subplots_adjust(wspace=0.2,hspace=0.3)
plt.legend(bbox_to_anchor=(-2.5,5.3, 3.5, 0), loc='upper center',
           ncol=4, mode="expand", prop=font2,borderaxespad=0.)



plt.savefig("D:/work1/International Journal of Forecasting/IRv2.jpeg")



