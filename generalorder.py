#This is the code for obtaining the general order analytical results .
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
'''
The general order is based on the recursion of <E_N^n>
E_N^1 = <N|V|N> = F_iijj
E_N^2 = <N|V|M><M|V|N> this one may be tricky since M needs to be the distinct result
1. first need to get the permutation of N in the limit of 16.  < 0-16, 0-16, 0-16>
1.5. evaluate E_N^0 and thermal average to compare with BE[16]
2. evaluate E_N^1 then thermal average.
3. evaluate E_N^2 then thermal average.
'''
class generalorder:
    def __init__(self,nmode):
        Ehbykb = 3.1577464*100000
        #Temprt = np.array([100,1000,10000,100000,1000000,10000000,100000000])
        #Temprt = np.array([10000000])
        Temprt = 100
        Temprt = Temprt/Ehbykb
        w_omega,FCQ3,FCQ4,FCQ3scale,FCQ4scale = self.readSindoPES("./data/prop_no_3.hs",nmode)
        maxn = 16 
        maxorder = 5
        linrComb = self.loopfn(nmode,maxn)
        Evlst = self.EvaluationList(nmode,w_omega,maxn,maxorder)
        Omg0 = self.EN0(w_omega,maxn,Evlst,linrComb,Temprt)
        print(Omg0)
        #Omg1 = self.EN1(w_omega,maxn,Evlst,linrComb,Temprt,FCQ4,nmode)
        #print(Omg1)
        print("_________________________________________________________")
        Omg2 = self.EN2(w_omega,maxn,Evlst,linrComb,Temprt,FCQ3,FCQ4,nmode,maxorder=5) 
        print("general order 2 omega is ",Omg2)    
        print("_________________________________________________________")
        Omg1 = self.EN1_2(w_omega,maxn,Evlst,linrComb,Temprt,FCQ4,nmode,FCQ4scale)
        print("general order 1 omega is ",Omg1)
            
        #print(len(linrComb))
        #print(linrComb)

    def loopfn(self,n,maxn):
        if n>1:
            rt = []
            for x in range(maxn):
                k = self.loopfn(n-1,maxn)
                for i in range(len(k)):
                    k[i].append(x)
                rt += k
            return rt
        else:
            rt = []
            for x in range(maxn):
                rt.append([x])
            return rt
    #pass test
    def EN0(self,w_omega,maxn,Evlst,linrComb,Temprt):
        beta = 1/(Temprt)
        Xisum = 0.0
        for i in range(len(linrComb)):
            E_Nsum = 0.0
            for j in range(len(linrComb[i])):
                E_Nsum += (linrComb[i][j]+0.5) * w_omega[j]
            Xisum += math.exp(-beta*E_Nsum)
        Omg = - math.log(Xisum)/beta
        return Omg

        #get omega first
    def EN1(self,w_omega,maxn,Evlst,linrComb,Temprt,FCQ4,nmode):

        beta = 1/(Temprt)
        #iterate through all Force constants
        FCQ3permute = np.zeros((nmode*nmode*nmode,nmode+1))
        FCQ4permute = np.zeros((nmode*nmode*nmode*nmode,nmode+1))
        a=0
        b=0
        for ii in range(nmode):
            for jj in range(nmode):
                for kk in range(nmode):
                    #eachcount3 = Counter([ii,jj,kk])
                    #for idx in range(nmode):
                    #    FCQ3permute[a,idx] = eachcount3[idx]#store frequency of each mode
                    #a+=1
                    for ll in range(nmode):
                        eachcount4 = Counter([ii,jj,kk,ll])
                        for idxx in range(nmode):
                            FCQ4permute[b,idxx] = eachcount4[idxx]#store frequency of each mode
                        FCQ4permute[b,3] = FCQ4[ii,jj,kk,ll]/24#store scaled Force constant 
                        b+=1

        # EN1 = <N|V|N>
        #<EN1> = (EN1 exp(-beta*EN0))/(exp(-beta*EN0) 
        Xidenom = 0.0
        Xinome = 0.0
        test = 0.0
        for i in range(len(linrComb)):#iterate N
            E_Nsum = 0.0
            E_N0sum = 0.0
            for jj in range(nmode):
                E_N0sum += (linrComb[i][jj]+0.5) * w_omega[jj]
            for FCQ4idx in range(FCQ4permute.shape[0]): #sum of force constants permutation
                modemult = 1
                for j in range(nmode):#3 mode need to multiply them,if diff =0 and operator number is 0 then result is 1
                    modemult *= Evlst[j,int(FCQ4permute[FCQ4idx,j]),linrComb[i][j],0] 
                    if(modemult == 0):
                        break
                E_Nsum += modemult*FCQ4permute[FCQ4idx,3]
            Xinome += E_Nsum*math.exp(-beta*E_N0sum)
            Xidenom += math.exp(-beta*E_N0sum)
        return Xinome/Xidenom

    def EN1_2(self,w_omega,maxn,Evlst,linrComb,Temprt,FCQ4,nmode,FCQ4scale):
        beta = 1/(Temprt)
        f = np.zeros(w_omega.shape)
        for i in range(len(w_omega)):
            f[i] = 1/(math.exp(beta*w_omega[i])-1)

        A1 = 0.0
        for i in range(nmode):
            for j in range(nmode):
                A1 += FCQ4scale[i,i,j,j]*(2*f[i]+1)*(2*f[j]+1)/8
        print("infinite <EN1> is ",A1)
        #test
        A1 = 0.0
        for i in range(nmode):
            for j in range(nmode):
                ninome = 0.0
                nidenome = 0.0
                for ni in range(10):
                    ninome += (1+2*ni)*math.exp(-beta*ni*w_omega[i])
                    nidenome += math.exp(-beta*ni*w_omega[i])
                njnome = 0.0
                njdenome = 0.0
                for nj in range(10):
                    njnome += (1+2*nj)*math.exp(-beta*nj*w_omega[j])
                    njdenome += math.exp(-beta*nj*w_omega[j])
                A1 += FCQ4scale[i,i,j,j]*(ninome/nidenome)*(njnome/njdenome)/8
        print("16 analytical <EN1> is ",A1)
        Xidenom = 0.0
        Xinome = 0.0
        for i in range(len(linrComb)):
            E_N0sum = 0.0
            sumofoperator = 0.0
            for j in range(nmode):
                E_N0sum += (linrComb[i][j]) * w_omega[j]
            for ii in range(nmode):
                for jj in range(nmode):
                    for kk in range(nmode):
                        for ll in range(nmode):
                            multply = 1
                            eachcount = Counter([ii,jj,kk,ll])
                            for modeidx in range(nmode):
                                n = linrComb[i][modeidx] 
                                numberofmodeinFC = eachcount[modeidx]
                                multply*= Evlst[modeidx,numberofmodeinFC,n,0]
                            multply*=FCQ4[ii,jj,kk,ll]
                            sumofoperator+=multply/24
            Xinome += sumofoperator*math.exp(-beta*E_N0sum)
            Xidenom += math.exp(-beta*E_N0sum)
        return Xinome/Xidenom

    def EN2(self,w_omega,maxn,Evlst,linrComb,Temprt,FCQ3,FCQ4,nmode,maxorder=5): 
        #strategy for optimization:
        #1. no need to parse FC index then parse each mode,we can parse each mode diff and let FC index goes from 0 to 4
        #2. only store non-zero FC in a serial vector and store its index
        #the key point of this algo is to enumerate the permutation
        #<N|V|M><M|V|N> M is not same as N
        beta = 1/(Temprt)
        Xidenom = 0.0
        Xinome = 0.0
        for i in range(len(linrComb)):#N, and E_N0sum is done here
            sumofoperator = 0.0
            E_N0sum=0.0
            for E0idx in range(nmode):
                E_N0sum += linrComb[i][E0idx] * w_omega[E0idx]
            E_Nsum =0.0#sum for each N, needing sum of all M
            for j in range(len(linrComb)):# M runs over N except for M=N
                if(i==j): 
                    continue
                Nhs = np.array(linrComb[i])
                Mhs = np.array(linrComb[j])
                n = np.maximum(Nhs,Mhs)
                diff = np.abs(Nhs-Mhs)
                vecQ3 = [] 
                vecQ4 = []
                for ii in range(nmode):
                    for jj in range(nmode):
                        for kk in range(nmode):
                            multplyQ3 = 1
                            eachcount3 = Counter([ii,jj,kk])
                            for modeidx3 in range(nmode):
                                if(diff[modeidx3]>=maxorder-1):
                                    multplyQ3 = 0
                                    break
                                numberofmodeinFC = eachcount3[modeidx3]
                                multplyQ3 *= Evlst[modeidx3,numberofmodeinFC,n[modeidx3],diff[modeidx3]]
                            multplyQ3*=FCQ3[ii,jj,kk]/6
                            if(multplyQ3!=0):
                                vecQ3.append(multplyQ3)
                            for ll in range(nmode):
                                multplyQ4 = 1
                                eachcount = Counter([ii,jj,kk,ll])
                                for modeidx in range(nmode):
                                    if(diff[modeidx]>=maxorder):
                                        multplyQ4 = 0
                                        break
                                    numberofmodeinFC = eachcount[modeidx]
                                    multplyQ4 *= Evlst[modeidx,numberofmodeinFC,n[modeidx],diff[modeidx]]
                                if(multplyQ4!=0):
                                    multplyQ4*=FCQ4[ii,jj,kk,ll]/24
                                    vecQ4.append(multplyQ4)
                for each1 in vecQ3:
                    for each2 in vecQ3:
                        sumofoperator+= each1*each2
                for each1 in vecQ4:
                    for each2 in vecQ4:
                        sumofoperator+= each1*each2
            Xinome += sumofoperator*math.exp(-beta*E_N0sum)
            Xidenom += math.exp(-beta*E_N0sum)
            return Xinome/Xidenom
        


    def EvaluationList(self,nmode,w_omega,maxn,maxorder):
        #I used the combination to determine which operator can give us result.
        #The 1st is to indicate which normal mode is it.
        #The 2nd is to indicate which operator: 0-4 : 0, Q, Q^2, Q^3, Q^4. Here we used QFF so the max order of operator is 4 and total number is 4
        #The 3rd is to the which level n is, n is the bigger one than n' 
        #The 4th is the difference between n and n'
        Evlst = np.zeros((nmode,maxorder,maxn,maxorder))
        for i in range(nmode):
            for n in range(maxn):
                #Evlst[i,0,n,0] = - w_omega[i]*(n+0.5)
                #Evlst[i,0,n,2] = w_omega[i]*math.sqrt(n*(n-1))/2
                Evlst[i,0,n,0] = 1
                Evlst[i,1,n,1] = math.sqrt(n/2/w_omega[i])
                Evlst[i,2,n,0] = (n+0.5)/w_omega[i]
                Evlst[i,2,n,2] = math.sqrt(n*(n-1))/2/w_omega[i]
                Evlst[i,3,n,1] = 3*n/2/w_omega[i]*math.sqrt(n/2/w_omega[i])
                Evlst[i,3,n,3] = math.sqrt(n*(n-1)*(n-2))/(2*w_omega[i]*math.sqrt(2*w_omega[i]))
                Evlst[i,4,n,0] = (6*n*(n+1)+3)/4/(w_omega[i]**2)
                Evlst[i,4,n,2] =  (n-0.5)*math.sqrt(n*(n-1))/(w_omega[i]**2)
                Evlst[i,4,n,4] = math.sqrt(n*(n-1)*(n-2)*(n-3))/4/(w_omega[i]**2)
        return Evlst

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
        FCQ3scale = np.copy(FCQ3)
        FCQ4scale = np.copy(FCQ4)
        #XXX  : remember to scale the force constants 
        for i in range(nmode):
            for j in range(nmode):
                for k in range(nmode):
                    FCQ3scale[i,j,k]= FCQ3scale[i,j,k]/math.sqrt(2*w_omega[i])/math.sqrt(2*w_omega[j])/math.sqrt(2*w_omega[k])
                    for l in range(nmode):
                        FCQ4scale[i,j,k,l]= FCQ4scale[i,j,k,l]/math.sqrt(2*w_omega[i])/math.sqrt(2*w_omega[j])/math.sqrt(2*w_omega[k])/math.sqrt(2*w_omega[l])
        return w_omega,FCQ3,FCQ4,FCQ3scale,FCQ4scale

    def Bose_EinsteinStat(self,Temprt,w_omega):
        b_beta= 1/Temprt
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

test = generalorder(3) 
