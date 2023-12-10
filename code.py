# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 00:34:19 2018

@author: Sushmit
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import matplotlib.pyplot as plt

data = pd.read_excel("SWLC_h.xlsx")
data2 = pd.read_excel("data2.xlsx")
data3 = pd.read_excel("data3.xlsx")

data = pd.concat([data,data2,data3],axis = 0)

#removing outliers (bad data)
from scipy import stats
data = data[(np.abs(stats.zscore(data)) < 1.0).all(axis=1)]

X = data.drop("Burst time", axis = 1).values
#X = data.iloc[:,2:5]
y = data.iloc[:,1].values
y = y.reshape(256,1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_ph = tf.placeholder(tf.float32,shape = [None,4])
y_ph = tf.placeholder(tf.float32, shape = [None,1])

input_layer = tf.layers.dense(inputs = X_ph, units = 4, activation=tf.nn.relu)
hidden1 = tf.layers.dense(inputs = input_layer, units = 3, activation=tf.nn.relu)
output = tf.layers.dense(inputs = hidden1, units = 1, activation=tf.nn.relu)

loss = tf.losses.mean_squared_error(y_ph,output)
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

steps = 5000

saver = tf.train.Saver()

# TRAINING THE MODEL
with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        sess.run(train, feed_dict={X_ph:X_train,y_ph:y_train})
        
        
        if(i % 100 == 0):
            print("On step {}".format(i))
            logits = output.eval(feed_dict={X_ph:X_test})
            
            ctr = 0
            for i in range(len(logits)):
                error = logits[i] - y_test[i]
                ctr += error**2
               
                
            print("RMSE: {}".format((ctr/len(logits))**(1/2)))
        
        if(i == steps - 1):
            print("On step {}".format(i))
            logits = output.eval(feed_dict={X_ph:X_test})
            
            ctr = 0
            for i in range(len(logits)):
                error = logits[i] - y_test[i]
                ctr += error**2
                
            for i in range(len(logits)):
                if(abs(logits[i]-y_test[i]) < 60):
                    print(logits[i],y_test[i])

            
            #print("RMSE: {}".format((ctr/len(logits))**(1/2)))
            
    saver.save(sess, './saved_model/model')
    
def f (n,bt):
    l=[]
    wt=[]
    bt1=[]
    avgwt=0
    tat=[]
    avgtat=0
    for i in bt:
        bt1.append(int(i))
    wt.insert(0,0)
    tat.insert(0,bt[0])
    
    for i in range(1,len(bt)):
        wt.insert(i,wt[i-1]+bt[i-1])
        tat.insert(i,wt[i]+bt[i])
        avgwt+=wt[i]
        avgtat+=tat[i]
        
    avgwt=float(avgwt)/n
    avgtat=float(avgtat)/n
    print("\n")
    print("Average Waiting time FCFS: "+str(avgwt))
    print("Average Turn Arount Time FCFS: "+str(avgtat))
    l.append(avgwt)
    
    processes=[]
    for i in range(0,n):
     processes.insert(i,i+1)
    for i in range(0,len(bt)-1):  
     for j in range(0,len(bt)-i-1):
      if(bt[j]>bt[j+1]):
       temp=bt[j]
       bt[j]=bt[j+1]
       bt[j+1]=temp
       temp=processes[j]
       processes[j]=processes[j+1]
       processes[j+1]=temp
    avgwt=0 
    avgtat=0   
    for i in range(1,len(bt)):  
     wt.insert(i,wt[i-1]+bt[i-1])
     tat.insert(i,wt[i]+bt[i])
     avgwt+=wt[i]
     avgtat+=tat[i]
    avgwt=float(avgwt)/n
    avgtat=float(avgtat)/n
    print("\n")
    print("Average Waiting time SJF: "+str(avgwt))
    print("Average Turn Arount Time SJF: "+str(avgtat))
    l.append(avgwt)
    
    processes=[]
    for i in range(0,n):
     processes.insert(i,i+1)
    
    print("\nEnter the priority of the processes: \n")
    priority=list(map(int, input().split()))
    tat=[]
    wt=[]
     
    for i in range(0,len(priority)-1):
     for j in range(0,len(priority)-i-1):
      if(priority[j]>priority[j+1]):
       swap=priority[j]
       priority[j]=priority[j+1]
       priority[j+1]=swap
     
       swap=bt1[j]
       bt1[j]=bt1[j+1]
       bt1[j+1]=swap
     
       swap=processes[j]
       processes[j]=processes[j+1]
       processes[j+1]=swap
     
    wt.insert(0,0)
    tat.insert(0,bt1[0])
     
    for i in range(1,len(processes)):
     wt.insert(i,wt[i-1]+bt1[i-1])
     tat.insert(i,wt[i]+bt1[i])
     
    avgtat=0
    avgwt=0
    for i in range(0,len(processes)):
     avgwt=avgwt+wt[i]
     avgtat=avgtat+tat[i]
    avgwt=float(avgwt)/n
    avgwt=float(avgtat)/n
    print("\n")
    print("Average Waiting time Priority: "+str(avgwt))
    print("Average Turn Around Time Priority: "+str(avgtat))
    l.append(avgwt)
    
    z=l[0]
    x=0
    for i in range(len(l)):
        if z>l[i]:
            z=l[i]
            x=i
    if x==0:
        print("FCFS")
    elif x==1:
        print("SJF")
    else:
        print("Priority")


with tf.Session() as sess:
    
    bt=[]
    print("Enter the number of process: ")
    n=int(input())
    arrivals = []
    n_res = []
    job_ids = []
    prempt = []
    
    for i in range(n):
        arrivals.append(float(input("Enter the arrival time: ")))
        job_ids.append(int(input("Enter the Job ID: ")))
        prempt.append(int(input("Is it preemptive or non-preemptive: (1 or 0): ")))
        n_res.append(int(input("Enter the number of resources: ")))
    
    saver.restore(sess, "./saved_model/model")
    
    inp = []
    for i in range(n):
        inp.append([job_ids[i],arrivals[i],prempt[i],n_res[i]])
        
    inps = np.array(inp).reshape(n,4)
    inps = sc.transform(inps)
    preds = output.eval(feed_dict={X_ph:inps})
    preds.reshape(-1,n)
    print(preds)
    f(n,preds)








