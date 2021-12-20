#This is the code for obtaining the numerical results .
import numpy as np
import math
import itertools
from collections import Counter
import sys
from numpy import linalg
from numba import jit
import time
from multiprocessing import Process, Queue
import csv
from itertools import permutations 
class Numerical:
    def __init__(self,nmode):

        Ehbykb = 3.1577464*100000
        #Temprt = np.array([100,1000,10000,100000,1000000,10000000,100000000])
        Temprt = 1000/Ehbykb
        w,FCQ3,FCQ4 = self.readSindoPES("./data/prop_no_3.hs",nmode)
        beta = 1/(Temprt)
        f = np.zeros(w.shape)
        for i in range(len(w)):
            f[i] = 1/(math.exp(beta*w[i])-1)

        self.FTPTnumeric(w,f,FCQ3,FCQ4,nmode)
        self.GFnumeric(w,f,FCQ3,FCQ4,nmode)

#S: singly D:doubly T:triply Q:quadruply the name match with the ones in the table:S01:1.01

    def FTPTnumeric(self,w,f,FCQ3,FCQ4,nmode):
        A01 = 0
        A02 = 0
        S01 = 0
        S07 = 0
        S02 = 0
        S08 = 0
        S03 = 0
        S04 = 0
        S09 = 0
        S05 = 0
        S10 = 0
        S06 = 0
        D01 = 0
        D09 = 0
        D02 = 0
        D07 = 0
        D03 = 0
        D15 = 0
        D05 = 0
        D11 = 0
        D12 = 0
        D10 = 0
        D08 = 0
        D04 = 0
        D16 = 0
        D13 = 0
        D14 = 0
        D06 = 0
        T01 = 0
        T05 = 0
        T02 = 0
        T06 = 0
        T03 = 0
        T07 = 0
        T04 = 0
        T08 = 0
        Q01 = 0
        Q02 = 0
        Q03 = 0
        Q04 = 0
        Q05 = 0
        Q06 = 0
        Q07 = 0
        Q08 = 0
        for i in range(nmode):
            S08 += -FCQ3[i,i,i]**2*(6*f[i]**2 + 6*f[i] + 1)/(4*w[i]) 
            S03 += -FCQ3[i,i,i]**2*(3*f[i]**2 + 3*f[i] + 1)/(18*w[i])

            A01 += FCQ4[i,i,i,i]*(2*f[i]+1)**2/8

            S10 += -FCQ4[i,i,i,i]**2*(32*f[i]**3 + 48*f[i]**2 + 22*f[i] + 3)/(48*w[i])
            S06 += -FCQ4[i,i,i,i]**2*(4*f[i]**3 + 6*f[i]**2 + 4*f[i] + 1)/(96*w[i])
            for j in range(nmode):
                if (j!=i):
                    S01 += -3*FCQ3[i,j,j]**2*(8*f[j]**2 + 8*f[j] + 1)/(12*w[i])
                    S02 += -FCQ3[i,j,j]*FCQ3[i,i,i]*(4*f[i]*f[j] + 2*f[i] + 2*f[j] + 1)/(2*w[i])
                    S04 += -FCQ4[i,i,j,j]**2*(16*f[i]*f[j]**2 + 16*f[i]*f[j] + 2*f[i] + 8*f[j]**2 + 8*f[j] + 1)/(16*w[i]) #6*6
                    S05 += -FCQ4[i,i,j,j]*FCQ4[i,i,i,i]*(8*f[i]**2*f[j] + 4*f[i]**2 + 8*f[i]*f[j] + 4*f[i] + 2*f[j] + 1)/(8*w[i])#6*1*2
                    D01 += -3*FCQ3[i,j,j]**2*(f[i] + f[j]**2 + 2*f[j]*(f[i] + 1) + 1)/(6*(w[i] + 2*w[j]))
                    D09 +=  3*FCQ3[i,j,j]**2*(2*f[i]*f[j] + f[i] - f[j]**2)/(6*(w[i] - 2*w[j])) 
                    
                    A02 +=  FCQ4[i,i,j,j]*(2*f[i]+1)*(2*f[j]+1)/8

                    D13 += -FCQ4[i,i,j,j]**2*(2*f[i]**2*f[j] + f[i]**2 + 2*f[i]*f[j]**2 + 4*f[i]*f[j] + 2*f[i] + f[j]**2 + 2*f[j] + 1)/(8*(w[i] + w[j]))/2#XXX 6*6 ???
                    D14 +=  FCQ4[i,i,j,j]**2*(f[i]**2 + 2*f[i]*f[j]*(f[i] - f[j]) - f[j]**2)/(8*(w[i] - w[j]))/2# 6*6???
                    D15 += -FCQ4[i,j,j,j]**2*(f[i] + 12*f[j]**2 + 6*f[j]*(f[i]*f[j] + f[i] + f[j]**2) + 7*f[j] + 1)/(4*(w[i] + w[j]))#4*4*2
                    D16 +=  FCQ4[i,j,j,j]**2*(f[i] + 6*f[j]*(f[i]*f[j] + f[i] - f[j]**2 - f[j]) - f[j])/(4*(w[i] - w[j]))#4*4*2
                    D05 += -FCQ4[i,j,j,j]*FCQ4[i,i,i,j]*(2*f[i]**2 + 4*f[i]*f[j]*(f[i] + f[j]) + 8*f[i]*f[j] + 3*f[i] + 2*f[j]**2 + 3*f[j] + 1)/(4*(w[i] + w[j]))#4*4
                    D06 +=  FCQ4[i,j,j,j]*FCQ4[i,i,i,j]*(2*f[i]**2 + 4*f[i]*f[j]*(f[i] - f[j]) + f[i] - 2*f[j]**2 - f[j])/(4*(w[i] - w[j]))# 4*4
                    D11 += -2*FCQ4[i,j,j,j]**2*(f[i] + f[j]**3 + 3*f[j]*(f[i]*f[j] + f[i] + f[j] + 1) + 1)/(12*(w[i] + 3*w[j]))#4*4*2
                    D12 +=  2*FCQ4[i,j,j,j]**2*(3*f[i]*f[j]*(f[j] + 1) + f[i] - f[j]**3)/(12*(w[i] - 3*w[j]))#4*4*2
                for k in range(nmode):
                    if( k!= i and k!=j and j!=i):
                        S07 += -FCQ3[i,k,k]*FCQ3[i,j,j]*(4*f[j]*f[k] + 2*f[j] + 2*f[k] + 1)/(4*w[i])
                        T01 += -FCQ3[i,j,k]**2*(f[i]*f[j] + f[i]*f[k] + f[i] + f[j]*f[k] + f[j] + f[k] + 1)/(w[i] + w[j] + w[k])/6
                        T05 +=  FCQ3[i,j,k]**2*(f[i]*f[j] - f[i]*f[k] - f[j]*f[k] - f[k])/(w[i] + w[j] - w[k])/6
                        T02 +=  FCQ3[i,j,k]**2*(-f[i]*f[j] + f[i]*f[k] - f[j]*f[k] - f[j])/(w[i] - w[j] + w[k])/6
                        T06 +=  FCQ3[i,j,k]**2*(f[i]*f[j] + f[i]*f[k] + f[i] - f[j]*f[k])/(w[i] - w[j] - w[k])/6

                        D10 +=  FCQ4[i,j,k,k]**2*(f[i] - f[j] + 8*f[k]*(f[i]*f[k] + f[i] - f[j]*f[k] - f[j]))/(8*(w[i] - w[j]))#6 * 6???XXX 
                        D02 += -FCQ4[i,j,k,k]**2*(f[i] + f[j] + 8*f[k]*(f[i]*f[k] + f[i] + f[j]*f[k] + f[j] + f[k] + 1) + 1)/(8*(w[i] + w[j]))#6 * 6 ???
                        D03 += -FCQ4[i,j,k,k]*FCQ4[i,j,j,j]*(2*f[i]*f[j] + 2*f[i]*f[k] + f[i] + 2*f[j]**2 + 4*f[j]*f[k]*(f[i] + f[j]) + 6*f[j]*f[k] + 3*f[j] + 2*f[k] + 1)/(2*(w[i] + w[j]))#6*4*2*2
                        D04 +=  FCQ4[i,j,k,k]*FCQ4[i,j,j,j]*(2*f[i]*f[j] + 2*f[i]*f[k] + f[i] - 2*f[j]**2 + 4*f[j]*f[k]*(f[i] - f[j]) - 2*f[j]*f[k] - f[j])/(2*(w[i] - w[j]))#6*4*2*2

                        S09 += -FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(8*f[i]*f[k]*f[j] + 4*f[i]*f[k] + 4*f[i]*f[j] + 2*f[i] + 4*f[k]*f[j] + 2*f[k] + 2*f[j] + 1)/(16*w[i])#6*6
                        T03 += -2*FCQ4[i,j,k,k]**2*(f[i]*f[j] + f[i]*f[k]**2 + f[i] + f[j]*f[k]**2 + f[j] + f[k]**2 + 2*f[k]*(f[i]*f[j] + f[i] + f[j] + 1) + 1)/(8*(w[i] + w[j] + 2*w[k]))# 6*6 *2 can't believe this 2
                        T07 +=  2*FCQ4[i,j,k,k]**2*(2*f[i]*f[j]*f[k] + f[i]*f[j] - f[i]*f[k]**2 - f[j]*f[k]**2 - f[k]**2)/(8*(w[i] + w[j] - 2*w[k]))
                        T04 += -2*FCQ4[i,j,k,k]**2*(f[i]*f[j] - f[i]*f[k]**2 + f[j]*f[k]**2 + 2*f[j]*f[k]*(f[i] + 1) + f[j])/(8*(w[i] - w[j] + 2*w[k]))
                        T08 +=  2*FCQ4[i,j,k,k]**2*(f[i]*f[j] + f[i]*f[k]**2 + 2*f[i]*f[k]*(f[j] + 1) + f[i] - f[j]*f[k]**2)/(8*(w[i] - w[j] - 2*w[k]))
                    for l in range(nmode):
                        if(l!=i and l!=j and l!=k and i!=j and i!=k and j!=k):
                            D07 += -FCQ4[i,j,l,l]*FCQ4[i,j,k,k]*(2*f[i]*f[k] + 2*f[i]*f[l] + f[i] + 2*f[j]*f[k] + 2*f[j]*f[l] + f[j] + 4*f[k]*f[l]*(f[i] + f[j] + 1) + 2*f[k] + 2*f[l] + 1)/(8*(w[i] + w[j]))
                            D08 +=  FCQ4[i,j,l,l]*FCQ4[i,j,k,k]*(2*f[i]*f[k] + 2*f[i]*f[l] + f[i] - 2*f[j]*f[k] - 2*f[j]*f[l] - f[j] + 4*f[k]*f[l]*(f[i] - f[j]))/(8*(w[i] - w[j]))
                            Q01 += -FCQ4[i,j,k,l]**2*(f[i]*f[j]*f[k] + f[i]*f[j]*f[l] + f[i]*f[j] + f[i]*f[k]*f[l] + f[i]*f[k] + f[i]*f[l] + f[i] + f[j]*f[k]*f[l] + f[j]*f[k] + f[j]*f[l] + f[j] + f[k]*f[l] + f[k] + f[l] + 1)/(24*(w[i] + w[j] + w[k] + w[l]))
                            Q02 += FCQ4[i,j,k,l]**2*(f[i]*f[j]*f[k] - f[i]*f[j]*f[l] - f[i]*f[k]*f[l] - f[i]*f[l] - f[j]*f[k]*f[l] - f[j]*f[l] - f[k]*f[l] - f[l])/(24*(w[i] + w[j] + w[k] - w[l]))
                            Q03 += FCQ4[i,j,k,l]**2*(-f[i]*f[j]*f[k] + f[i]*f[j]*f[l] - f[i]*f[k]*f[l] - f[i]*f[k] - f[j]*f[k]*f[l] - f[j]*f[k] - f[k]*f[l] - f[k])/(24*(w[i] + w[j] - w[k] + w[l]))
                            if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                                Q04 += FCQ4[i,j,k,l]**2*(f[i]*f[j]*f[k] + f[i]*f[j]*f[l] + f[i]*f[j] - f[i]*f[k]*f[l] - f[j]*f[k]*f[l] - f[k]*f[l])/(24*(w[i] + w[j] - w[k] - w[l]))
                            Q05 += FCQ4[i,j,k,l]**2*(-f[i]*f[j]*f[k] - f[i]*f[j]*f[l] - f[i]*f[j] + f[i]*f[k]*f[l] - f[j]*f[k]*f[l] - f[j]*f[k] - f[j]*f[l] - f[j])/(24*(w[i] - w[j] + w[k] + w[l]))
                            if (w[i] - w[j] - w[k] + w[l]!= 0 ):
                                Q06 += FCQ4[i,j,k,l]**2*(f[i]*f[j]*f[k] - f[i]*f[j]*f[l] + f[i]*f[k]*f[l] + f[i]*f[k] - f[j]*f[k]*f[l] - f[j]*f[l])/(24*(w[i] - w[j] + w[k] - w[l]))
                            if (w[i] - w[j] - w[k] + w[l]!= 0 ):
                                Q07 += FCQ4[i,j,k,l]**2*(-f[i]*f[j]*f[k] + f[i]*f[j]*f[l] + f[i]*f[k]*f[l] + f[i]*f[l] - f[j]*f[k]*f[l] - f[j]*f[k])/(24*(w[i] - w[j] - w[k] + w[l]))
                            Q08 += FCQ4[i,j,k,l]**2*(f[i]*f[j]*f[k] + f[i]*f[j]*f[l] + f[i]*f[j] + f[i]*f[k]*f[l] + f[i]*f[k] + f[i]*f[l] + f[i] - f[j]*f[k]*f[l])/(24*(w[i] - w[j] - w[k] - w[l]))
        result3rd = S01 +S07 +S02 +S08 +S03 +D01 +D09 +T01 +T05 +T02 +T06 
        result4th = S04 +S09 +S05 +S10 +S06 +D02 +D07 +D03 +D15 +D05 +D11 +D12 +D10 +D08 +D04 +D16 +D13 +D14 +D06 +T03 +T07 +T04 +T08 +Q01 +Q02 +Q03 +Q04 +Q05 +Q06 +Q07 +Q08
        print("FTPT")
        #print("3rd....")
        #print(result3rd)
        print("4th....")
        print(result4th)
        #print(D12+D11)
        #print(D15+D16)
        #print(D03+D04)
        print(D05+D06)
        #print(S04+D13+D14)
        #print(D02+D10+T03+T04+T07+T08)
        #print(T03+T04+T07+T08)
        #print(D02+D10)
        #print(D11+D12+D15+D16)
        #print(D07+D08)


    def GFnumeric(self,w,f,FCQ3,FCQ4,nmode):
        #second order diagram:
        A1  = 0
        A2  = 0
        B2N = 0
        B2D = 0
        C2  = 0
        D1  = 0
        D2  = 0
        D3  = 0
        D4  = 0
        D5  = 0
        D6  = 0
        D7  = 0
        D8  = 0
        #GP1 = 0 #ijkk ijll
        GP6 = 0 #iijj iiii 2BD
        GP3 = 0 #ijii ijjj 2BN

        GP2 = 0 #ijkk ijjj 2BN
        GP7 = 0 #iikk iijj 2BD
        GP4 = 0 #iiii^2 2B 2D
        GP5 = 0 #iijj^2 2B 2D
        GP8 = 0 #ijjj^2 2B 2D
        GP9 = 0 #ijkk^2 2B 2D
        test4th1 = 0
        test4th2 = 0
        count= 0
        for i in range(nmode):
            for j in range(nmode):
                #A1 += FCQ4[i,i,j,j]*(2*f[i]+1)*(2*f[j]+1)/8
                for k in range(nmode):
                    GP6t1 = [i,i,j,j]
                    GP6t2 = [i,i,k,k]
                    if ((len(set(GP6t1))==2 and len(set(GP6t2))==1) or(len(set(GP6t1))==1 and len(set(GP6t2))==2)):
                        GP6+= - FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(f[i]+1/2)/w[i]/8
                    GP7t1 = [i,i,j,j]
                    GP7t2 = [i,i,k,k]
                    if ((len(set(GP7t1))==2 and len(set(GP7t2))==2) and j!=k):
                        GP7 += - FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(f[i]+1/2)/w[i]/8
                    GP4t1 = [i,i,j,j]
                    GP4t2 = [i,i,k,k]
                    if ((len(set(GP4t1))==1 and len(set(GP4t2))==1)):
                        GP4+= -FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(f[i]+1/2)/w[i]/8
                    GP5t1 = [i,i,j,j]
                    GP5t2 = [i,i,k,k]
                    if ((len(set(GP5t1))==2 and len(set(GP5t2))==2) and j==k):
                        GP5 += - FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(f[i]+1/2)/w[i]/8
                    B2D += -FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(f[i]+1/2)/w[i]/8
                    #A2 += -FCQ3[i,j,j]*FCQ3[i,k,k]*(2*f[j]+1)*(2*f[k]+1)/w[i]/4     
                    #C2 += -FCQ3[i,j,k]**2*((f[i]*f[j]+f[j]*f[k]+f[i]*f[k]+f[i]+f[j]+f[k]+1)/(w[i]+w[j]+w[k])-(f[i]*f[j]+f[j]*f[k]-f[i]*f[k]+f[j])/(w[j]-w[i]-w[k])-(f[i]*f[j]-f[k]*f[j]-f[i]*f[k]-f[k])/(w[j]+w[i]-w[k])-(f[k]*f[j]-f[i]*f[j]-f[i]*f[k]-f[i])/(w[j]-w[i]+w[k]))/6
                    for l in range(nmode):
                        if(i!=j):
                            GP3t1 = [i,j,k,k]
                            GP3t2 = [i,j,l,l]
                            if (len(set(GP3t1))==2 and len(set(GP3t2))==2 and k!=l ):
                                GP3 += -FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(w[i]*(2*f[j]+1) - w[j]*(2*f[i]+1))/(w[i]**2-w[j]**2)/8
                            GP2t1 = [i,j,k,k]
                            GP2t2 = [i,j,l,l]
                            if ((len(set(GP2t1))==3 and len(set(GP2t2))==2) or (len(set(GP2t1))==2 and len(set(GP2t2))==3)):
                                GP2 += -FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(w[i]*(2*f[j]+1) - w[j]*(2*f[i]+1))/(w[i]**2-w[j]**2)/8
                            GP9t1 = [i,j,k,k]
                            GP9t2 = [i,j,l,l]
                            if (len(set(GP9t1))==3 and len(set(GP9t2))==3 and k==l):
                                GP9 += -FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(w[i]*(2*f[j]+1) - w[j]*(2*f[i]+1))/(w[i]**2-w[j]**2)/8
                            GP8t1 = [i,j,k,k]
                            GP8t2 = [i,j,l,l]
                            if (len(set(GP8t1))==2 and len(set(GP8t2))==2 and k==l):
                                GP8 += -FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(w[i]*(2*f[j]+1) - w[j]*(2*f[i]+1))/(w[i]**2-w[j]**2)/8
                            B2N+= -FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(w[i]*(2*f[j]+1) - w[j]*(2*f[i]+1))/(w[i]**2-w[j]**2)/8
                        GP4t3 = [i,j,k,l]
                        if ( len(set(GP4t3))==1):
                            GP4 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)+f[i]*(f[l]+1)*(f[j]+1)-f[k]*f[j]*f[l])/(-w[i]+w[j]+w[k]+w[l])/24
                            GP4 += -FCQ4[i,j,k,l]**2*(f[l]*f[k]*(f[j]+f[i]+1)+f[l]*(f[i]+1)*(f[j]+1)-f[i]*f[j]*f[k])/(w[i]+w[j]+w[k]-w[l])/24
                            GP4 += -FCQ4[i,j,k,l]**2*((f[k]+1)*(f[i]+1)*(f[l]+1)*(f[j]+1)-f[i]*f[j]*f[k]*f[l])/(w[i]+w[j]+w[k]+w[l])/24
                            GP4 += -FCQ4[i,j,k,l]**2*(f[k]*f[j]*(f[i]+f[l]+1)+f[j]*(f[i]+1)*(f[l]+1)-f[i]*f[k]*f[l])/(w[i]-w[j]+w[k]+w[l])/24
                            GP4 += -FCQ4[i,j,k,l]**2*(f[k]*(f[i]+1)*(f[j]+1)*(f[l]+1)-f[i]*f[j]*f[l]*(f[k]+1))/(w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] - w[k] + w[l]!= 0 ):
                                GP4 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/(-w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] + w[k] - w[l]!= 0 ):
                                GP4 += -FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/(-w[i]+w[j]+w[k]-w[l])/24
                            if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                                GP4 += -FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/(w[i]+w[j]-w[k]-w[l])/24
                        GP5t3 = [i,j,k,l]
                        if ( len(set(GP5t3))==2 and GP5t3.count(list(set(GP5t3))[0])==2 and GP5t3.count(list(set(GP5t3))[1])==2):#need to exclude ijjj -> set would be 2 too
                            GP5 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)+f[i]*(f[l]+1)*(f[j]+1)-f[k]*f[j]*f[l])/(-w[i]+w[j]+w[k]+w[l])/24
                            GP5 += -FCQ4[i,j,k,l]**2*(f[l]*f[k]*(f[j]+f[i]+1)+f[l]*(f[i]+1)*(f[j]+1)-f[i]*f[j]*f[k])/(w[i]+w[j]+w[k]-w[l])/24
                            GP5 += -FCQ4[i,j,k,l]**2*((f[k]+1)*(f[i]+1)*(f[l]+1)*(f[j]+1)-f[i]*f[j]*f[k]*f[l])/(w[i]+w[j]+w[k]+w[l])/24
                            GP5 += -FCQ4[i,j,k,l]**2*(f[k]*f[j]*(f[i]+f[l]+1)+f[j]*(f[i]+1)*(f[l]+1)-f[i]*f[k]*f[l])/(w[i]-w[j]+w[k]+w[l])/24
                            GP5 += -FCQ4[i,j,k,l]**2*(f[k]*(f[i]+1)*(f[j]+1)*(f[l]+1)-f[i]*f[j]*f[l]*(f[k]+1))/(w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] - w[k] + w[l]!= 0 ):
                                GP5 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/(-w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] + w[k] - w[l]!= 0 ):
                                GP5 += -FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/(-w[i]+w[j]+w[k]-w[l])/24
                            if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                                GP5 += -FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/(w[i]+w[j]-w[k]-w[l])/24

                        GP9t3 = [i,j,k,l]
                        if ( len(set(GP9t3))==3):
                            GP9 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)+f[i]*(f[l]+1)*(f[j]+1)-f[k]*f[j]*f[l])/(-w[i]+w[j]+w[k]+w[l])/24
                            GP9 += -FCQ4[i,j,k,l]**2*(f[l]*f[k]*(f[j]+f[i]+1)+f[l]*(f[i]+1)*(f[j]+1)-f[i]*f[j]*f[k])/(w[i]+w[j]+w[k]-w[l])/24
                            GP9 += -FCQ4[i,j,k,l]**2*((f[k]+1)*(f[i]+1)*(f[l]+1)*(f[j]+1)-f[i]*f[j]*f[k]*f[l])/(w[i]+w[j]+w[k]+w[l])/24
                            GP9 += -FCQ4[i,j,k,l]**2*(f[k]*f[j]*(f[i]+f[l]+1)+f[j]*(f[i]+1)*(f[l]+1)-f[i]*f[k]*f[l])/(w[i]-w[j]+w[k]+w[l])/24
                            GP9 += -FCQ4[i,j,k,l]**2*(f[k]*(f[i]+1)*(f[j]+1)*(f[l]+1)-f[i]*f[j]*f[l]*(f[k]+1))/(w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] - w[k] + w[l]!= 0 ):
                                GP9 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/(-w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] + w[k] - w[l]!= 0 ):
                                GP9 += -FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/(-w[i]+w[j]+w[k]-w[l])/24
                            if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                                GP9 += -FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/(w[i]+w[j]-w[k]-w[l])/24
                        GP8t3 = [i,j,k,l]
                        if ( len(set(GP8t3))==2 and GP8t3.count(list(set(GP8t3))[0])!=2 and GP8t3.count(list(set(GP8t3))[1])!=2):
                            GP8 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)+f[i]*(f[l]+1)*(f[j]+1)-f[k]*f[j]*f[l])/(-w[i]+w[j]+w[k]+w[l])/24
                            GP8 += -FCQ4[i,j,k,l]**2*(f[l]*f[k]*(f[j]+f[i]+1)+f[l]*(f[i]+1)*(f[j]+1)-f[i]*f[j]*f[k])/(w[i]+w[j]+w[k]-w[l])/24
                            GP8 += -FCQ4[i,j,k,l]**2*((f[k]+1)*(f[i]+1)*(f[l]+1)*(f[j]+1)-f[i]*f[j]*f[k]*f[l])/(w[i]+w[j]+w[k]+w[l])/24
                            GP8 += -FCQ4[i,j,k,l]**2*(f[k]*f[j]*(f[i]+f[l]+1)+f[j]*(f[i]+1)*(f[l]+1)-f[i]*f[k]*f[l])/(w[i]-w[j]+w[k]+w[l])/24
                            GP8 += -FCQ4[i,j,k,l]**2*(f[k]*(f[i]+1)*(f[j]+1)*(f[l]+1)-f[i]*f[j]*f[l]*(f[k]+1))/(w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] - w[k] + w[l]!= 0 ):
                                GP8 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/(-w[i]+w[j]-w[k]+w[l])/24
                            if (-w[i] + w[j] + w[k] - w[l]!= 0 ):
                                GP8 += -FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/(-w[i]+w[j]+w[k]-w[l])/24
                            if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                                GP8 += -FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/(w[i]+w[j]-w[k]-w[l])/24
                        D1 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)+f[i]*(f[l]+1)*(f[j]+1)-f[k]*f[j]*f[l])/(-w[i]+w[j]+w[k]+w[l])/24
                        D3 += -FCQ4[i,j,k,l]**2*(f[l]*f[k]*(f[j]+f[i]+1)+f[l]*(f[i]+1)*(f[j]+1)-f[i]*f[j]*f[k])/(w[i]+w[j]+w[k]-w[l])/24
                        D4 += -FCQ4[i,j,k,l]**2*((f[k]+1)*(f[i]+1)*(f[l]+1)*(f[j]+1)-f[i]*f[j]*f[k]*f[l])/(w[i]+w[j]+w[k]+w[l])/24
                        D6 += -FCQ4[i,j,k,l]**2*(f[k]*f[j]*(f[i]+f[l]+1)+f[j]*(f[i]+1)*(f[l]+1)-f[i]*f[k]*f[l])/(w[i]-w[j]+w[k]+w[l])/24
                        D8 += -FCQ4[i,j,k,l]**2*(f[k]*(f[i]+1)*(f[j]+1)*(f[l]+1)-f[i]*f[j]*f[l]*(f[k]+1))/(w[i]+w[j]-w[k]+w[l])/24
                        if (-w[i] + w[j] - w[k] + w[l]!= 0 ):
                            D7 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/(-w[i]+w[j]-w[k]+w[l])/24
                        if (-w[i] + w[j] + w[k] - w[l]!= 0 ):
                            D2 += -FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/(-w[i]+w[j]+w[k]-w[l])/24
                        if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                            D5 += -FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/(w[i]+w[j]-w[k]-w[l])/24
        GFresult3rd = A2 + C2 
        GFresult4th = B2N +B2D +D1 +D2 +D3 +D4 +D5 +D6 +D7 +D8 
        print("GF")
        #print("3rd....")
        #print(GFresult3rd)
        print("4th....")
        #print(GFresult4th)
        #print(B2D+ 3*(D1 +D2 +D3 +D6 +D7 +D8)+3*(D4 +D5) )#2.13 2.14 1.04
        #print(6*(D4+D5)) 2.13 2.14This six I am not sure. If there are no other bugs then should be this one 
        #print(B2D+3*(D3+D8+D1+D6)) #1.04 
        #print(B2N + 6*(D2+D3+D7+D8)) #2.02 2.10
        #print(6*(D5+D6+D4+D1))# 3.03 04 07 08
        #print(2*(D1+D4)) #2.11 12
        #print(B2N + 2*(D2+D3+D5+D6+D7+D8))# 2.15 2.16
        #print(B2N*2) 2.03 2.04
        #print(B2N*4) #2.05 2.06
        #----------------
        #print(count)
        #print(B2N)
        #print(GP4+GP5+GP6+GP7)
        #print(GP4+GP5+GP9+GP8)
        print(B2N+B2D+D1 +D2 +D3 +D4 +D5 +D6 +D7 +D8)
        #print(GP2+GP3+GP9+GP8)
        print(GP2+GP3+GP4+GP5+GP6+GP7+GP8+GP9)

    def readSindoPES(self,filepath,nmode):
        w_omega = np.zeros(nmode)
        FCQ3 = np.zeros((nmode,nmode,nmode)) #Coefficient in Q (normal coordinates)
        #XXX Coefficient includes the 1/2 1/3! 1/4! in the front!!
        #Dr.Yagi used dimensionless q as unit so we need to transfer from q to Q by times sqrt(w1*w2*.../hbar^(...))
        FCQ4 = np.zeros((nmode,nmode,nmode,nmode))
        with open(filepath) as f:
            flines = f.readlines()
            for idx in range(len(flines)):
                if( len(flines[idx].split())>1):

                    if (flines[idx].split()[1] == "Hessian(i,i)"):
                        tl = flines[idx+1].split()#shortcut for this line
                        leng=  len(tl)
                        if (leng == 2):
                            for i in range(nmode):
                                tl2 = flines[idx+1+i].split()
                                w_omega[i] = math.sqrt(float(tl2[1]))
                                #print("Hessian",math.sqrt(float(tl2[1])/(1.88973**2*math.sqrt(1822.888486**2)))*219474.63)
                    if (flines[idx].split()[1] == "Cubic(i,i,i)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            FCQ3[int(tl[0])-1,int(tl[0])-1,int(tl[0])-1] = float(tl[1])
                            #print("Cubic3",tl[1])
                    if (flines[idx].split()[1] == "Cubic(i,i,j)"):
                        for i in range(nmode*2):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[1])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ3[i[0],i[1],i[2]] = float(tl[2])
                            #print("Cubic2",tl[2])
                    if (flines[idx].split()[1] == "Cubic(i,j,k)"):
                        tl = flines[idx+1].split()#shortcut for this line
                        listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[2])-1]
                        perm = permutations(listidx)
                        for i in list(perm):
                            FCQ3[i[0],i[1],i[2]] = float(tl[3])
                        #print("Cubic1",tl[3])

                    if (flines[idx].split()[1] == "Quartic(i,i,i,i)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            FCQ4[int(tl[0])-1,int(tl[0])-1,int(tl[0])-1,int(tl[0])-1] = float(tl[1])
                            #print("Quar4",tl[1])
                    if (flines[idx].split()[1] == "Quartic(i,i,j,j)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[1])-1,int(tl[1])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ4[i[0],i[1],i[2],i[3]] = float(tl[2])
                            #print("Quar22",tl[2])
                    if (flines[idx].split()[1] == "Quartic(i,i,i,j)"):
                        for i in range(nmode*2):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[0])-1,int(tl[1])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ4[i[0],i[1],i[2],i[3]] = float(tl[2])
                            #print("Quar21",tl[2])
                    if (flines[idx].split()[1] == "Quartic(i,i,j,k)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[1])-1,int(tl[2])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ4[i[0],i[1],i[2],i[3]] = float(tl[3])
                            #print("Quar3",tl[3])
        FCQ3 = np.true_divide(FCQ3,(1.88973**3*math.sqrt(1822.888486**3)))    
        FCQ4 = np.true_divide(FCQ4,(1.88973**4*math.sqrt(1822.888486**4)))    
        #XXX TODO : remember to scale the force constants 
        w_omega = np.true_divide(w_omega,math.sqrt(1.88973**2*1822.888486))    
        return w_omega,FCQ3,FCQ4
    

test =  Numerical(3)
