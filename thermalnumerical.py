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
        Temprt = np.array([100,1000,10000,100000,1000000,10000000,100000000])
        #Temprt = np.array([10000000])
        Temprt = Temprt/Ehbykb
        w,FCQ3,FCQ4 = self.readSindoPES("./data/prop_no_3.hs",nmode)
        for ii in range(len(Temprt)):
            print("::::::::::::::")
            #ret12 = self.GFnumeric(w,Temprt[ii],FCQ3,FCQ4,nmode)
            ret12 = self.SQnumeric(w,Temprt[ii],FCQ3,FCQ4,nmode)
            retOmg0 = self.Bose_EinsteinStat(Temprt[ii],w)
            print(retOmg0+ret12)


    def SQnumeric(self,w,Temprt,FCQ3,FCQ4,nmode):
        beta = 1/(Temprt)
        f = np.zeros(w.shape)
        for i in range(len(w)):
            f[i] = 1/(math.exp(beta*w[i])-1)
        #print("checking f",f)
        #second order diagram:
        Atest = 0
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
        D7degen= 0
        D5degen= 0
        D2degen= 0
        count= 0
        for i in range(nmode):
            for j in range(nmode):
                Atest+=FCQ4[i,i,j,j]/8
                #print("test case")
                #print(FCQ4[i,i,j,j])
                #print(2*f[i]+1)
                #print(2*f[j]+1)
                #print((2*f[i]+1)*(2*f[j]+1))
                #print("test case done")
                A1 += FCQ4[i,i,j,j]*(2*f[i]+1)*(2*f[j]+1)/8
                for k in range(nmode):
                    B2D += -FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(f[i]+1/2)/w[i]/8 
                    #B2D += -beta*FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(2*f[i]**2+2*f[i]+1)/2
                    A2 += -FCQ3[i,j,j]*FCQ3[i,k,k]*(2*f[j]+1)*(2*f[k]+1)/w[i]/4     
                    C2 += -FCQ3[i,j,k]**2*((f[i]*f[j]+f[j]*f[k]+f[i]*f[k]+f[i]+f[j]+f[k]+1)/(w[i]+w[j]+w[k])-(f[i]*f[j]+f[j]*f[k]-f[i]*f[k]+f[j])/(w[j]-w[i]-w[k])-(f[i]*f[j]-f[k]*f[j]-f[i]*f[k]-f[k])/(w[j]+w[i]-w[k])-(f[k]*f[j]-f[i]*f[j]-f[i]*f[k]-f[i])/(w[j]-w[i]+w[k]))/6
                    for l in range(nmode):
                        if(i!=j):
                            #B2N+= -FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(w[i]*(2*f[j]+1) - w[j]*(2*f[i]+1))/(w[i]**2-w[j]**2)/8
                            B2N +=  FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(- (f[i]+f[j]+1)/(w[i]+w[j]) + (f[j]-f[i])/(w[i]-w[j])/2)/8
                        D4 += -FCQ4[i,j,k,l]**2*((f[k]+1)*(f[i]+1)*(f[l]+1)*(f[j]+1)-f[i]*f[j]*f[k]*f[l])/(w[i]+w[j]+w[k]+w[l])/24
                        D3 += -FCQ4[i,j,k,l]**2*(f[l]*f[k]*(f[j]+f[i]+1)+f[l]*(f[i]+1)*(f[j]+1)-f[i]*f[j]*f[k])/(w[i]+w[j]+w[k]-w[l])/6
                        #if (-w[i] + w[j] - w[k] + w[l]!= 0 ):
                        #    D7 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/(-w[i]+w[j]-w[k]+w[l])/24
                        #else:
                        #    D7degen+=  beta*FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/24/2

                        #if (-w[i] + w[j] + w[k] - w[l]!= 0 ):
                        #    D2 += -FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/(-w[i]+w[j]+w[k]-w[l])/24
                        #else:
                        #    D2degen+= beta*FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/24/2

                        if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                            D5 += -FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/(w[i]+w[j]-w[k]-w[l])/16
                        else:
                            #print(beta*FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)+f[i]*f[j]*(f[l]+f[k]+1))/16/2)
                            D5degen+= -beta*FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/16/2 
        GFresult3rd = A2 + C2 
        GFresult4th = B2N +B2D +D1 +D2 +D3 +D4 +D5 +D6 +D7 +D8 + D5degen
        ret= A1+GFresult4th+GFresult3rd
        #print("degen is",D7degen+D5degen+D2degen)
        print("first",A1)
        #print(A1)
        print("second",GFresult4th+GFresult3rd)
        #print(GFresult4th+GFresult3rd+D7degen+D5degen+D2degen)
        return ret

    def GFnumeric(self,w,Temprt,FCQ3,FCQ4,nmode):
        beta = 1/(Temprt)
        f = np.zeros(w.shape)
        for i in range(len(w)):
            f[i] = 1/(math.exp(beta*w[i])-1)
        #print("checking f",f)
        #second order diagram:
        Atest = 0
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
        D7degen= 0
        D5degen= 0
        D2degen= 0
        count= 0
        for i in range(nmode):
            for j in range(nmode):
                Atest+=FCQ4[i,i,j,j]/8
                #print("test case")
                #print(FCQ4[i,i,j,j])
                #print(2*f[i]+1)
                #print(2*f[j]+1)
                #print((2*f[i]+1)*(2*f[j]+1))
                #print("test case done")
                A1 += FCQ4[i,i,j,j]*(2*f[i]+1)*(2*f[j]+1)/8
                for k in range(nmode):
                    B2D += -FCQ4[i,i,j,j]*FCQ4[i,i,k,k]*(2*f[j]+1)*(2*f[k]+1)*(f[i]+1/2)/w[i]/8 
                    A2 += -FCQ3[i,j,j]*FCQ3[i,k,k]*(2*f[j]+1)*(2*f[k]+1)/w[i]/4     
                    C2 += -FCQ3[i,j,k]**2*((f[i]*f[j]+f[j]*f[k]+f[i]*f[k]+f[i]+f[j]+f[k]+1)/(w[i]+w[j]+w[k])-3*(f[i]*f[j]+f[j]*f[k]-f[i]*f[k]+f[j])/(w[j]-w[i]-w[k]))/6
                    for l in range(nmode):
                        if(i!=j):
                            B2N+= -FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*(2*f[k]+1)*(2*f[l]+1)*(w[i]*(2*f[j]+1) - w[j]*(2*f[i]+1))/(w[i]**2-w[j]**2)/8
                        D1 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)+f[i]*(f[l]+1)*(f[j]+1)-f[k]*f[j]*f[l])/(-w[i]+w[j]+w[k]+w[l])/24
                        D4 += -FCQ4[i,j,k,l]**2*((f[k]+1)*(f[i]+1)*(f[l]+1)*(f[j]+1)-f[i]*f[j]*f[k]*f[l])/(w[i]+w[j]+w[k]+w[l])/24
                        D3 += -FCQ4[i,j,k,l]**2*(f[l]*f[k]*(f[j]+f[i]+1)+f[l]*(f[i]+1)*(f[j]+1)-f[i]*f[j]*f[k])/(w[i]+w[j]+w[k]-w[l])/24
                        D6 += -FCQ4[i,j,k,l]**2*(f[k]*f[j]*(f[i]+f[l]+1)+f[j]*(f[i]+1)*(f[l]+1)-f[i]*f[k]*f[l])/(w[i]-w[j]+w[k]+w[l])/24
                        D8 += -FCQ4[i,j,k,l]**2*(f[k]*(f[i]+1)*(f[j]+1)*(f[l]+1)-f[i]*f[j]*f[l]*(f[k]+1))/(w[i]+w[j]-w[k]+w[l])/24
                        if (-w[i] + w[j] - w[k] + w[l]!= 0 ):
                            D7 += -FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/(-w[i]+w[j]-w[k]+w[l])/24
                        else:
                            D7degen+=  beta*FCQ4[i,j,k,l]**2*(f[i]*f[k]*(f[j]+f[l]+1)-f[j]*f[l]*(f[i]+f[k]+1))/24/2

                        if (-w[i] + w[j] + w[k] - w[l]!= 0 ):
                            D2 += -FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/(-w[i]+w[j]+w[k]-w[l])/24
                        else:
                            D2degen+= beta*FCQ4[i,j,k,l]**2*(f[i]*f[l]*(f[k]+f[j]+1)-f[j]*f[k]*(f[i]+f[l]+1))/24/2

                        if (w[i] + w[j] - w[k] - w[l]!= 0 ):
                            D5 += -FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/(w[i]+w[j]-w[k]-w[l])/24
                        else:
                            D5degen+= beta*FCQ4[i,j,k,l]**2*(f[k]*f[l]*(f[j]+f[i]+1)-f[i]*f[j]*(f[l]+f[k]+1))/24/2 
        GFresult3rd = A2 + C2 
        GFresult4th = B2N +B2D +D1 +D2 +D3 +D4 +D5 +D6 +D7 +D8 
        ret= A1+GFresult4th+GFresult3rd
        #print("degen is",D7degen+D5degen+D2degen)
        print("first",A1)
        #print(A1)
        print("second",GFresult4th+GFresult3rd)
        #print(GFresult4th+GFresult3rd+D7degen+D5degen+D2degen)
        return ret
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
        w_omega = np.true_divide(w_omega,math.sqrt(1.88973**2*1822.888486))    
        #XXX  : remember to scale the force constants 
        for i in range(nmode):
            for j in range(nmode):
                for k in range(nmode):
                    FCQ3[i,j,k]= FCQ3[i,j,k]/math.sqrt(2*w_omega[i])/math.sqrt(2*w_omega[j])/math.sqrt(2*w_omega[k])
                    for l in range(nmode):
                        FCQ4[i,j,k,l]= FCQ4[i,j,k,l]/math.sqrt(2*w_omega[i])/math.sqrt(2*w_omega[j])/math.sqrt(2*w_omega[k])/math.sqrt(2*w_omega[l])

        return w_omega,FCQ3,FCQ4
    
    def Bose_EinsteinStat(self,Temprt,w_omega):
        b_beta= 1/Temprt
        #f_i 
        #f_i = np.zeros(len(w_omega))
        #for i in range(len(w_omega)):
        #    f_i[i] = 1/(1-math.exp(-b_beta*w_omega[i]))
        #print(f_i)
        #partition function
        #GPF_Xi = 1
        #for ii in range(len(w_omega)):
        #    #GPF_Xi *=  math.exp(-b_beta*eachw/2)/(1-math.exp(-b_beta*eachw))
        #    GPF_Xi *=  math.exp(-b_beta*w_omega[ii]/2)*f_i[ii]
        #grand potential 
        GP_Omg = 0
        for eachw in w_omega:
            GP_Omg += 0.5*eachw +  math.log(1-math.exp(-b_beta*eachw))/b_beta
        #internal energy
        IE_U = 0
        for eachw in w_omega:
            IE_U += 0.5*eachw +  eachw*math.exp(-b_beta*eachw)/(1-math.exp(-b_beta*eachw))
        #entropy
        entropy_S = 0
        for eachw in w_omega:
            entropy_S += - math.log(1-math.exp(-b_beta*eachw)) + eachw*math.exp(-b_beta*eachw)/(Temprt*(1-math.exp(-b_beta*eachw)))
        print("total OMG",GP_Omg)
        return GP_Omg

test =  Numerical(3)
