# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:27:38 2021

@author: Navid
"""
import numpy as np
import pandas as pd
from pulp import*
'''we define each node as an object'''
class NLDS():    
    def __init__(self,t,k):
        self.t = t
        self.k = k
        self.T = [(1.25,1.14),(1.06,1.12)]
        self.p = 0
        self.cuts = []
        self.e = np.zeros((200,1))
        self.E = np.zeros((200,2))
        self.l = 0        
        if self.t != 4:
            self.x = LpVariable.dicts('x',[(i)for i in range(1,3)] , 0 , cat='Continuous')
            self.theta =  LpVariable('theta',cat='Continuous')
        if self.t == 4 :
            self.w = LpVariable('w', 0 , cat = "Continuous")
            self.y = LpVariable('y', 0 , cat = "Continuous")
            self.theta = LpVariable('theta',cat = "Continuous")   
    def a(self,k):        
        if k % 2 == 0 :
            return 2
        else: 
            return 1   
    def add_cut(self,i):
        self.cuts.append(i)
        self.creat_model()
    def creat_model(self):               
        if self.t ==1 :            
            self.prob = LpProblem(sense = LpMinimize)
            self.prob += self.theta
            self.prob += lpSum(self.x[(i)] for i in range(1,3)) == 55            
            if len(self.cuts) != 0:
                for i in self.cuts:
                    self.prob += i
            else: 
                self.prob += self.theta == 0                    
            return self.prob            
        if self.t == 2 :            
            self.prob = LpProblem(sense = LpMinimize)
            self.prob += self.theta
            self.prob += lpSum(self.x[(i)] for i in range(1,3)) == lpSum(self.T[self.a(self.k)-1][i]*self.p[i] for i in range(len(self.p)))            
            if len(self.cuts) != 0:
                for i in self.cuts:
                    self.prob += i
            else: 
                self.prob += self.theta == 0                    
            return self.prob        
        if self.t == 3 :            
            self.prob = LpProblem(sense = LpMinimize)
            self.prob += self.theta
            self.prob += lpSum(self.x[(i)] for i in range(1,3)) == lpSum(self.T[self.a(self.k)-1][i]*self.p[i] for i in range(len(self.p)))            
            if len(self.cuts) != 0:
                for i in self.cuts:
                    self.prob += i
            else: 
                self.prob += self.theta == 0                    
            return self.prob        
        if self.t == 4:            
            self.prob = LpProblem(sense = LpMinimize)
            self.prob += self.theta + 4*self.w - self.y
            self.prob += lpSum(self.T[self.a(self.k)-1][i]*self.p[i] for i in range(len(self.p))) + self.w - self.y == 80
            self.prob += self.theta == 0            
            return self.prob        
    def solve(self):
        self.sigma=[]
        prob = self.creat_model()
        prob.solve()
        self.pi = list(self.prob.constraints.items())[0][1].pi
        if len(list(self.prob.constraints.items())) > 1:
            if str(list(self.prob.constraints.items())[1][1]) != 'theta = 0' :
                for i in range(len(list(self.prob.constraints.items()))-1):
                   self.sigma.append(list(self.prob.constraints.items())[i+1][1].pi)               
'''constructing NLDS problems'''
#first stage
t1 = [NLDS(1,1)]
#second stage
t2=[]
for k in range(1,3):
    t2.append(NLDS(2,k))
#third stage
t3=[]
for k in range(1,5):
    t3.append(NLDS(3,k))
#fourth stage
t4=[]
for k in range(1,9):
    t4.append(NLDS(4,k))
'''--------------------------------main body---------------------------------''' 
stage = [t1,t2,t3,t4]
j = True
T = np.array([[1.25,1.14],[1.06,1.12]])
l=0
itter = 1
while j:    
    flag1 = len(stage[0][0].cuts)
    #forward
    print("---------------------------------iteration{}---------------------------------".format(itter))
    print("DIR : FORWARD")
    for t in range(4):
        for k in range(2**(t)):
            if t==0 :
                stage[t][k].solve()
            else :
                stage[t][k].p = [stage[t-1][int(np.floor(k/2))].x[(1)].varValue,stage[t-1][int(np.floor(k/2))].x[(2)].varValue]
                stage[t][k].solve()
            if t != 3:
                print("t:{}/k:{}/x1:{}/x2:{}/theta:{}".format(t+1,k+1,np.round(stage[t][k].x[(1)].varValue,1),np.round(stage[t][k].x[(2)].varValue,1),np.round(stage[t][k].theta.varValue,2)))
            elif t ==3:
                print("t:{}/k:{}/w:{}/y:{}/theta:{}".format(t+1,k+1,np.round(stage[t][k].w.varValue,1),np.round(stage[t][k].y.varValue,1),np.round(stage[t][k].theta.varValue,2)))   
    #backward 
    itter+=1
    print("---------------------------------iteration{}---------------------------------".format(itter))
    print("DIR : BACKWARD")          
    for t in reversed(range(3)):
        for k in range(2**t):
            print("-----------------------------------------------")
            print(print("t:{}/k:{}".format(t+1,k+1)))
            print("-----------------------------------------------")
            print("NLDS({},{}) before cut".format(t+1,k+1))
            print(stage[t][k].prob)
            #obtaining dual variables
            pi1 = stage[t+1][2*k].pi
            pi2 = stage[t+1][2*k+1].pi
            pi = [pi1,pi2]
            sigma1 = stage[t+1][2*k].sigma
            sigma2 = stage[t+1][2*k+1].sigma   
            print("pi=({},{}) / sigma=({},{})".format(pi1,pi2,sigma1,sigma2))
            # calculating cut parameters
            if t== 2:
                for i in range(T.shape[0]):
                    stage[t][k].E[stage[t][k].l] += 0.5*(pi[i]*T[i])
            else:
                for i in range(T.shape[0]):
                    stage[t][k].E[stage[t][k].l] += -0.5*(pi[i]*T[i])
                #stage[t][k].E[stage[t][k].l] = 0.5*(pi1*np.array([[-1.25,-1.14]]) + pi2*np.array([[-1.06,-1.12]]))
            if t == 2:
                for i in range(T.shape[0]):
                    stage[t][k].e[stage[t][k].l] += 0.5*pi[i]*80  
            else :
                stage[t][k].e[stage[t][k].l] = 0
            if len(sigma1)!=0: 
                for i in range(len(sigma1)):
                    stage[t][k].e[stage[t][k].l]+= 0.5*sigma1[i]*stage[t+1][2*k].e[i]
            if len(sigma2)!=0: 
                for i in range(len(sigma2)):
                    stage[t][k].e[stage[t][k].l]+= 0.5*sigma2[i]*stage[t+1][2*k+1].e[i]
            #stage[t][k].e[stage[t][k].l] = np.round(stage[t][k].e[stage[t][k].l],2)                    
            print("E:{},e:{}".format(stage[t][k].E[stage[t][k].l],stage[t][k].e[stage[t][k].l][0]))
            print("theta:{},w:{}".format(np.round(stage[t][k].theta.varValue,3),np.round((stage[t][k].e[stage[t][k].l] - (stage[t][k].E[stage[t][k].l][0]*stage[t][k].x[(1)].varValue) - (stage[t][k].E[stage[t][k].l][1]*stage[t][k].x[(2)].varValue)),2)))
            # add cut
            if  str(list(stage[t][k].prob.constraints.items())[1][1]) == 'theta = 0' :
                print("we add the cut below to NLDS({},{})".format(t+1,k+1))
                print("cut : theta + {}*x1 + {}*x2 >= {}".format(stage[t][k].E[stage[t][k].l][0],stage[t][k].E[stage[t][k].l][1],stage[t][k].e[stage[t][k].l][0]))
                stage[t][k].add_cut(stage[t][k].E[stage[t][k].l][0]*stage[t][k].x[(1)] + (stage[t][k].E[stage[t][k].l][1]*stage[t][k].x[(2)]) + stage[t][k].theta >= stage[t][k].e[stage[t][k].l][0])
                print("NLDS({},{}) after cut".format(t+1,k+1))
                stage[t][k].solve()
                print(stage[t][k].prob)
                stage[t][k].l += 1
            elif str(list(stage[t][k].prob.constraints.items())[1][1]) != 'theta = 0' :
                if  np.round(stage[t][k].theta.varValue,3) < np.round((stage[t][k].e[stage[t][k].l] - (stage[t][k].E[stage[t][k].l][0]*stage[t][k].x[(1)].varValue) - (stage[t][k].E[stage[t][k].l][1]*stage[t][k].x[(2)].varValue)),3):
                    print("we add the cut below to NLDS({},{})".format(t+1,k+1))
                    print("cut : theta + {}*x1 + {}*x2 >= {}".format(stage[t][k].E[stage[t][k].l][0],stage[t][k].E[stage[t][k].l][1],stage[t][k].e[stage[t][k].l][0]))
                    stage[t][k].add_cut(stage[t][k].E[stage[t][k].l][0]*stage[t][k].x[(1)] + (stage[t][k].E[stage[t][k].l][1]*stage[t][k].x[(2)]) + stage[t][k].theta >= stage[t][k].e[stage[t][k].l][0])
                    #if t!=0 :
                    stage[t][k].solve()
                    print("NLDS({},{}) after cut".format(t+1,k+1))
                    print(stage[t][k].prob)
                    stage[t][k].l += 1
                else:
                    print("no cuts will be added to NLDS({},{})".format(t+1,k+1))
                    stage[t][k].E[stage[t][k].l] = 0 
                    stage[t][k].e[stage[t][k].l] = 0
    itter+=1
    flag2 = len(stage[0][0].cuts)
    if flag2-flag1 == 0:
         j = False
         for t in range(4):
            for k in range(2**(t)):
                if t != 3:
                    print("t:{}/k:{}/x1:{}/x2:{}/theta:{}".format(t+1,k+1,np.round(stage[t][k].x[(1)].varValue,1),np.round(stage[t][k].x[(2)].varValue,1),np.round(stage[t][k].theta.varValue,1)))
                elif t ==3:
                    print("t:{}/k:{}/w:{}/y:{}/theta:{}".format(t+1,k+1,np.round(stage[t][k].w.varValue,1),np.round(stage[t][k].y.varValue,1),np.round(stage[t][k].theta.varValue,1)))
obj=0
for i in range(len(stage[3])):
    obj+= stage[3][i].y.varValue - 4*stage[3][i].w.varValue
print("////////////////////////////////////////////////////////")
print("The Objective Function Is Equal to :",1/8*obj)               
                
                
            
          
                    
                    
                
                
        
                
            
            
            
            
        
    
        
  


