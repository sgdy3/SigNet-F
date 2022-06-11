# -*- coding: utf-8 -*-
# ---
# @File: texture_mat.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/22
# Describe: 实现了signet-f
# ---

from keras.layers import Conv2D,MaxPool2D,Dense,Input,BatchNormalization,Lambda,Activation,Flatten
from keras.models import Sequential,Model
from keras.losses import categorical_crossentropy,binary_crossentropy
import keras.backend as K
from keras.optimizer_v2.gradient_descent import SGD
import numpy as np
from keras.utils.vis_utils import plot_model
import os


class SigNet_F():
    def __init__(self,num_class,mod='thin'):
        self.rows=150
        self.cols=220
        self.channles=1
        self.imgshape = (self.rows, self.cols, self.channles)
        self.user_dim=num_class

        self.batchsize=32
        self.epochs=6
        self.optimizer=SGD(lr=1e-3,momentum=0.9,nesterov=True,decay=5e-4)

        assert mod=='thin' or 'std',"model has only two variant: thin and std"
        if mod=='thin':
            self.backbone=self.backbone_thin()
        else:
            self.backbone=self.backbone_std()
        sig=Input(shape=self.imgshape)
        m_label=Input(shape=(self.user_dim,))
        f_label=Input(shape=(1,))

        feature=self.backbone(sig)
        pred_m=Dense(self.user_dim)(feature)
        pred_f=Dense(1)(feature)
        mixed_loss=Lambda(self.combine_loss,name='loss')([m_label,pred_m,f_label,pred_f])
        self.signet_f=Model([sig,m_label,f_label],[pred_m,pred_f,mixed_loss])

        loss_layer=self.signet_f.get_layer('loss').output
        self.signet_f.add_loss(loss_layer)
        self.signet_f.compile(optimizer=self.optimizer)
        plot_model(self.signet_f, to_file='signet_f.png', show_shapes=True)
        self.signet_f.summary()


    def combine_loss(self,args,alpha=0.99):
        m_label,pred_m,f_label,pred_f=args
        cat_los=categorical_crossentropy(m_label,pred_m)
        b_los=binary_crossentropy(f_label,pred_f)
        return K.mean((1-alpha)*cat_los+alpha*b_los)

    def backbone_thin(self):
        seq=Sequential()

        # 155*220->37*53
        seq.add(Conv2D(32,kernel_size=11,strides=4,input_shape=self.imgshape,use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 37*53->18*26
        seq.add(MaxPool2D(pool_size=3,strides=2))

        # 18*26->8*12
        seq.add(Conv2D(64,kernel_size=5,strides=1,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->8*12
        seq.add(MaxPool2D(pool_size=3,strides=2))

        # 8*12->8*12
        seq.add(Conv2D(64,kernel_size=3,strides=1,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->8*12
        seq.add(Conv2D(96,kernel_size=3,strides=1,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->8*12
        seq.add(Conv2D(96,kernel_size=5,strides=1,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->3*5
        seq.add(MaxPool2D(pool_size=3,strides=2))

        # 3*5->2048*1
        seq.add(Flatten())
        seq.add(Dense(128,use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.summary()

        # user_dim->binary
        img=Input(shape=self.imgshape)
        feature=seq(img)

        return Model(img,feature)

    def backbone_std(self):
        seq=Sequential()

        seq.add(Conv2D(96,kernel_size=11,strides=4,input_shape=self.imgshape,use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.add(MaxPool2D(pool_size=3,strides=2))

        seq.add(Conv2D(256,kernel_size=5,strides=2,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.add(MaxPool2D(pool_size=3,strides=2))

        seq.add(Conv2D(384,kernel_size=3,strides=1,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.add(Conv2D(384,kernel_size=3,strides=1,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.add(Conv2D(256,kernel_size=3,strides=1,padding='same',use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.add(MaxPool2D(pool_size=3,strides=2))

        seq.add(Flatten())
        seq.add(Dense(2048,use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.add(Dense(2048,use_bias=False))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.summary()
        input=Input(shape=self.imgshape)
        output=seq(input)

        return Model(input,output)

    def train(self,data,weights='',save=False):
        save_dir = './NetWeights/Signet_f_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.signet_f.load_weights(filepath)
            doc=None
        else:
            print('train')
            filepath = os.path.join(save_dir, 'signet_f.h5')
            train_img=data.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            time=0
            doc=[]
            for i in range(1,self.epochs+1):
                for batch in train_img:
                    loss=self.signet_f.train_on_batch(batch)
                    doc.append(loss)
                    print("%d round: loss %f"%(time,loss))
                    time+=1
                # 总共进行三次学习率下降，每次下降10%
                if i%(self.epochs//3)==0:
                    self.optimizer.lr-=0.1*self.optimizer.lr
            if save:
                self.signet_f.save_weights(filepath)
        return doc

def early_stop(stop_round,loss,pre_loss,threshold=0.005):
    '''
    early stop setting
    :param stop_round: rounds under caculated
    :param pre_loss: loss list
    :param threshold: minimum one-order value of loss list
    :return: whether or not to jump out
    '''
    if(len(pre_loss)<stop_round):
        pre_loss.append(loss)
        return False
    else:
        loss_diff=np.diff(pre_loss,1)
        pre_loss.pop(0)
        pre_loss.append(loss)
        if(abs(loss_diff).mean()<threshold): # to low variance means flatten field
            return True
        else:
            return False