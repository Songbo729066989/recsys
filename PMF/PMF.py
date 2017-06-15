#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Bo Song on 2017/6/14
import numpy as np



class PMF():

    def train(self,num_user,num_item,train,test,learning_rate,K,regu_u,regu_i,maxiter):
        U=np.random.normal(0,0.1,(num_user,K))
        V = np.random.normal(0, 0.1, (num_item, K))
        pre_rmse=100.0
        endure_count=3
        patience=0
        for iter in range(maxiter):
            loss=0.0
            for data in train:
                user=data[0]
                item=data[1]
                rating=data[2]

                predict_rating=np.dot(U[user],V[item].T)
                error=rating-predict_rating
                loss+=error**2
                U[user]+=learning_rate*(error*V[item]-regu_u*U[user])
                V[item]+=learning_rate*(error*U[user]-regu_i*V[item])

                loss+=regu_u*np.square(U[user]).sum()+regu_i*np.square(V[item]).sum()
            loss=0.5*loss
            rmse=self.eval_rmse(U,V,test)
            print('iter:%d loss:%.3f rmse:%.5f'%(iter,loss,rmse))
            if rmse<pre_rmse:   # early stop
                pre_rmse=rmse
                patience=0
            else:
                patience+=1
            if patience>=endure_count:
                break


    def eval_rmse(self,U,V,test):
        test_count=len(test)
        tmp_rmse=0.0
        for te in test:
            user=te[0]
            item=te[1]
            real_rating=te[2]
            predict_rating=np.dot(U[user],V[item].T)
            tmp_rmse+=np.square(real_rating-predict_rating)
        rmse=np.sqrt(tmp_rmse/test_count)
        return rmse

def read_data(path,train_ratio):
    user_set={}
    item_set={}
    u_idx=0
    i_idx=0
    data=[]
    with open(path) as f:
        for line in f.readlines():
            u,i,r,_=line.split('::')
            if u not in user_set:
                user_set[u]=u_idx
                u_idx+=1
            if i not in item_set:
                item_set[i]=i_idx
                i_idx+=1
            data.append([user_set[u],item_set[i],float(r)])

    np.random.shuffle(data)
    train=data[0:int(len(data)*train_ratio)]
    test=data[int(len(data)*train_ratio):]
    return u_idx,i_idx,train,test

if __name__=='__main__':
    num_user,num_item,train,test=read_data('data/ratings.dat',0.8)
    pmf=PMF()
    pmf.train(num_user,num_item,train,test,0.01,10,0.01,0.01,100)



























if __name__ == '__main__':
    pass