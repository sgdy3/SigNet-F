# -*- coding: utf-8 -*-
# ---
# @File: main.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/6/5
# Describe: 
# ---


import numpy as np
import tensorflow as tf
from preprocessing import hafemann_preprocess
import keras.utils.np_utils
import re
import pickle
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from model import SigNet_F


def img_preprocess(file_name1,m_lab,f_lab,mod='train',ext_h=820,ext_w=890):
    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=hafemann_preprocess(img1,ext_h,ext_w)
    if mod=='train':
        m_lab= keras.utils.np_utils.to_categorical(m_lab,50)
    elif mod=='test':
        m_lab=m_lab
    else:
        raise Exception("mod doesn't exist")
    return img1,m_lab,f_lab

def path_extra(user_ord):
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    org_num = 24
    label_dict=dict(zip(user_ord,range(user_ord.shape[0])))  # 原用户标签映射到用户数量range之内
    file_path=[]
    for user in user_ord:
        for i in range(1,org_num+1):
            positive = [org_path % (user, i),label_dict[user],1]
            negative = [forg_path % (user, i),label_dict[user],0]
            file_path.append(positive)
            file_path.append(negative)
    return np.array(file_path)


def curve_eval(label,result):
    fpr, tpr, thresholds = roc_curve(label,result, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    pred_label=1-pred_label
    acc=(pred_label==label).sum()/label.size
    area = auc(fpr, tpr)
    print("EER:%f"%EER)
    print('AUC:%f'%area)
    print('ACC(EER_threshold):%f'%acc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()

if __name__=="__main__":
    net=SigNet_F(num_class=50)
    with open('./pair_ind/cedar_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)

    '''
    为了保证对比合理，SigNet和其他方式使用相同的用户训练
    但由于其他方案是pair输入，为了使得正负样本对数目相当，没有使用所有的负样本
    所以先提取出用户再使用其所有的样本
    '''
    user_order=[]
    for i in range(train_ind.shape[0]):
        user_order.append(int(re.search('(?<=_)\d+(?=_)',train_ind[i][0]).group())) # 提取测试图片的用户编号
    user_order=np.unique(user_order)
    train_ind=path_extra(user_order)
    train_ind=train_ind[np.random.permutation(train_ind.shape[0]),:]

    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1].astype(np.int8),train_ind[:, 2].astype(np.int8)))
    image=dataset.map(lambda x, y, z: tf.py_function(func=img_preprocess, inp=[x, y, z], Tout=[tf.uint8, tf.int8, tf.int8]))
    doc=net.train(image,save=True)



    '''
    用户相关判决阶段，训练集用户的真实签名特征向量作为负样本
    '''
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    train_ind=train_ind[train_ind[:,2].astype(int)==1]
    neg_vecs=[]
    neg_database=tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1].astype(np.int8),train_ind[:, 2].astype(np.int8)))
    neg_database=neg_database.map(lambda x, y, z: tf.py_function(func=img_preprocess, inp=[x, y, z], Tout=[tf.uint8, tf.int8, tf.int8]))
    for batch in neg_database.batch(32):
        neg_vecs.append(net.backbone.predict_on_batch(batch[0])) # 获得训练集中用户真实签名的特征向量
    neg_vecs=np.vstack(neg_vecs)

    '''
    获取测试集所有用户的所有签名特征向量
    '''
    test_ind=[]
    user_order=np.arange(1,56)[~np.isin(np.arange(1,56),user_order)] # 得到测试集用户
    for user in user_order:
        for i in range(1,25):
            pos=[org_path%(user,i),user,1]
            neg=[forg_path%(user,i),user,0]
            test_ind.append(pos)
            test_ind.append(neg)
    test_ind=np.array(test_ind)
    test_vec=[]
    test_label=[]
    test_sig=tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1].astype(np.int8),test_ind[:, 2].astype(np.int8)))
    test_sig=test_sig.map(lambda x, y, z: tf.py_function(func=img_preprocess, inp=[x, y, z,'test'], Tout=[tf.uint8, tf.int8, tf.int8]))
    for batch in test_sig.batch(32):
        test_vec.append(net.backbone.predict_on_batch(batch[0]))
        test_label.append(np.vstack([batch[1],batch[2]]))
    test_vec=np.vstack(test_vec)
    test_label=np.hstack(test_label).T

    '''
    对于每个训练集用户，随机采样24个真实签名中的12个做正样本，加上前述负样本一起训练用户相关SVM
    '''
    result=[]
    for user in user_order:
        user_ind=np.where(test_label[:,0]==user)[0] # test库中用户记录
        user_pos_ind=np.where((test_label[:,0]==user) & (test_label[:,1]==1))[0] # test库中用户真实样本记录
        user_train_ind=np.random.choice(user_pos_ind,12,replace=False) # 随机采样24个真实签名中的12个做正样本
        user_test_ind=user_ind[~np.isin(user_ind,user_train_ind)]

        skew = neg_vecs.shape[0] / user_train_ind.shape[0]
        svm_input=np.vstack([neg_vecs,test_vec[user_train_ind,:]])
        svm_label=np.concatenate([np.zeros(neg_vecs.shape[0]),np.ones(user_train_ind.shape[0])])
        svm=sklearn.svm.SVC(class_weight={1:skew},gamma=0.0048,probability=True)
        svm_with_scaler = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                             ('classifier', svm)])
        svm_with_scaler.fit(svm_input,svm_label)
        hyper_dist=svm_with_scaler.decision_function(test_vec[user_test_ind,:])
        result.append(np.vstack([hyper_dist,test_label[user_test_ind,1]]))
    result=np.hstack(result).T
    curve_eval(result[:,1],result[:,0])