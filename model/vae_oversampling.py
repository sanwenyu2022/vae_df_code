#!/usr/bin/env python
# coding: utf-8

'''
Implementation of the VAE-based ovesampling method:
    
It uses the typical scikit-learn syntax  with a .fit() function for training vae networks
and  uses resample() function to generate new minority class samples.
Currently it only supports binary classification.

'''


import tensorflow as tf
import pandas as pd 
import collections
import numpy as np 
import warnings
from sklearn.model_selection import train_test_split 
warnings.filterwarnings('ignore')




class vae_model(tf.keras.Model):
    def __init__(self,z_dim,input_dim,hidden_num=2):
        super(vae_model, self).__init__()
        self.input_dim=input_dim
        self.z_dim=z_dim
        self.hidden_num=hidden_num
        """
        hidden_num: select the hidden layer size in the encoder and decoder. 
        hidden_num=1: the hidden layer size is [64]
        hidden_num=2: the hidden layer size is [128,64]
        hidden_num=3: the hidden layer size is [128]
        
        """
        self.encoder_f1=tf.keras.layers.Dense(128,activation='relu')
        self.encoder_f2=tf.keras.layers.Dense(64,activation='relu')
        self.encoder_f3_mean=tf.keras.layers.Dense(self.z_dim)
        self.encoder_f3_var=tf.keras.layers.Dense(self.z_dim)
        
        self.decoder_f1=tf.keras.layers.Dense(128,activation='relu')
        self.decoder_f2=tf.keras.layers.Dense(64,activation='relu')
        self.decoder_f3=tf.keras.layers.Dense(self.input_dim,activation='relu')

    # encoder
    def encoder(self, x):
        if self.hidden_num==3:
            h=self.encoder_f1(x)
            mu=self.encoder_f3_mean(h)
            log_var=self.encoder_f3_var(h)
        elif self.hidden_num==2:
            h=self.encoder_f1(x)
            h=self.encoder_f2(h)
            mu=self.encoder_f3_mean(h)
            log_var=self.encoder_f3_var(h)
        else:
            h=self.encoder_f2(x)
            mu=self.encoder_f3_mean(h)
            log_var=self.encoder_f3_var(h)

        return  mu, log_var

    # decoder
    def decoder(self, z):
        if self.hidden_num==3:
            h=self.decoder_f1(z)
            output=self.decoder_f3(h)
        elif self.hidden_num==2:
            h=self.decoder_f2(z)
            h=self.decoder_f1(h)
            output=self.decoder_f3(h)
        else:
            h=self.decoder_f2(z)
            output=self.decoder_f3(h)
        return output

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var)         
        std = std**0.5                
        z = mu + std * eps
        return z

    def call(self, inputs,training=None):
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

class VAE():
    def __init__(self,data_path,z_dim=2,max_epochs=1000,batch_size=128,patient_num=50,hidden_num=1,lr=1e-4,delta=1e-4):
#         super(VAE,self).__init__()
        self.data_path=data_path
        self.max_epochs=max_epochs
        self.batch_size=batch_size
        self.patient_num=patient_num
        self.z_dim=z_dim
        self.hidden_num=hidden_num
        self.lr=lr
        self.delta=delta
        
        self.load_data()        
        self.input_dim=self.data_label_1_x.shape[1]
        
        
        self.vae_model=vae_model(self.z_dim,self.input_dim,self.hidden_num)
#         self.vae_model.build(input_shape=(self.batch_size, self.input_dim))
        
    def load_data(self):
        """
        load processed datasets and get minority class sampels
        """
        self.data=pd.read_csv(self.data_path)
        self.label_num_dict=dict(collections.Counter(self.data['label']))
        self.label_0_num=self.label_num_dict[0]
        self.label_1_num=self.label_num_dict[1]
        self.need_add_label_num=self.label_0_num-self.label_1_num#需要增强的数据量
        self.data_label_0=self.data[self.data['label']==0]
        self.data_label_1=self.data[self.data['label']==1]
        
        self.data_label_1_x=self.data_label_1[self.data_label_1.columns[:-1]]
        self.data_label_1_y=self.data_label_1[self.data_label_1.columns[-1:]]
        
        
        
        self.train_x,self.train_y,self.train_dataset=self.zip_data(self.train_x,self.train_y)
        self.valid_x,self.valid_y,self.valid_dataset=self.zip_data(self.valid_x,self.valid_y)
        self.test_x,self.test_y,self.test_dataset=self.zip_data(self.test_x,self.test_y)

    def scale_process(self):
        
       
        
        self.train_x,self.test_x,self.train_y,self.test_y=train_test_split(self.data_label_1_x,self.data_label_1_y,test_size=0.2)
        self.valid_x,self.test_x,self.valid_y,self.test_y=train_test_split(self.test_x,self.test_y,test_size=0.25)

    def zip_data(self,data_x,data_y):
        """
        pack the training datases with the size of batch_size
        """
        train_data = tf.data.Dataset.from_tensor_slices(data_x).batch(self.batch_size)
        train_labels = tf.data.Dataset.from_tensor_slices(data_y.values[:,0]).batch(self.batch_size)
        train_dataset = tf.data.Dataset.zip((train_data,train_labels)).shuffle(True)
        return train_data,train_data,train_dataset
    
    def fit(self):
        """
        train VAE networks
        """
        global x_hat,x,mu, log_var,loss
        #build model parameters
        self.vae_model.build(input_shape=(self.batch_size, self.input_dim))
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.valid_loss_list=[]
        self.smallest_loss=np.inf
        self.train_mu_list=[]
        self.train_logvar_list=[]
        
        for epoch in range(self.max_epochs):
            epoch_loss=0
            data_num=0
            for x,y in self.train_dataset:
                with tf.GradientTape() as tape:
                    x_hat, mu, log_var=self.vae_model(x)
                    rec_loss = tf.reduce_mean(tf.losses.MSE(x, x_hat))
                    kl_div = -0.5 * (log_var + 1 -mu**2 - tf.exp(log_var))
                    kl_div = tf.reduce_mean(kl_div) / x.shape[0]
                    
                   
                    loss = rec_loss + 1. * kl_div
                
                grads = tape.gradient(loss, self.vae_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.vae_model.trainable_variables))
                epoch_loss+=loss.numpy()*x.shape[0]
                data_num+=x.shape[0]
                
                self.train_mu_list+=mu.numpy().tolist()
                self.train_logvar_list+=log_var.numpy().tolist()
            
            
            #early stopping 
            valid_loss=self.evaluate_loss()
            self.valid_loss_list+=[valid_loss]
            print(f'epoch={epoch},train_loss={epoch_loss/data_num},valid_loss={valid_loss}')
            
            if valid_loss<self.smallest_loss:
                self.best_vae_model=self.vae_model
                
            if self.patient_num!=None:
                
                min_loss_index=np.argmin(self.valid_loss_list)
                if len(self.valid_loss_list)-min_loss_index>self.patient_num or valid_loss<self.delta:
                    break                
            
    def evaluate_loss(self):
        """
        calculate validation loss
        """
        epoch_loss=0
        data_num=0
        for x,y in self.valid_dataset:
            x_hat, mu, log_var=self.vae_model(x)
            rec_loss = tf.reduce_mean(tf.losses.MSE(x, x_hat))
            kl_div = -0.5 * (log_var + 1 -mu**2 - tf.exp(log_var))
            kl_div = tf.reduce_mean(kl_div) / x.shape[0]
            loss = rec_loss + 1. * kl_div
            
            epoch_loss+=loss.numpy()*x.shape[0]
            data_num+=x.shape[0]
        return epoch_loss/data_num
    
    def resample(self,creat_num=None):
        """
        generate new minority class samples
        
        """
        if creat_num==None or creat_num<=0:
            new_creat_z=np.random.normal(size=(self.need_add_label_num,self.z_dim))
        else:
            new_creat_z=np.random.normal(size=(creat_num,self.z_dim))
        
        new_creat_x_normal=self.best_vae_model.decoder(new_creat_z).numpy()
       
        new_creat_x=self.maxabs.inverse_transform(new_creat_x_normal)
        new_creat_x=pd.DataFrame(new_creat_x,columns=self.data_label_1.columns[:-1])
        new_creat_y=pd.DataFrame([1]*len(new_creat_x),columns=['label'])
        return new_creat_x,new_creat_y


