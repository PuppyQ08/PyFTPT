#This is the code for obtaining the general order analytical results .
import numpy as np
import math
import itertools as itl
from collections import Counter
import sys
from numpy import linalg
from numba import jit
import time
#from multiprocessing import Process, Queue
import csv
from itertools import permutations 
import glob,os
'''
General order algorithm, based on the HCPT and recursion for E_N^n and Omega.
we define three type of matrices:
    -* V_NM : Part of FCI matrix, < N | V | M > Hermitian.
        it is formulated as V[N,M] squared matrix, can be stored in file and reuse.
    -* deltaE_MN : 1/(E_N - E_M) not Hermitian,
        it is formulated as DE_MN[M,N] squared matrix, can be stored in file and reuse
    - a_MN^(n-1) : < M | Psi_N^(n-1) > not Hermitian. a_NN^(n) = \delta_n0. initial case: a_MN^(0) = \delta_MN
        it is formulated as a[n,M,N]. 
    - E_N^(n) : it is a non diagonal matrix. But we only need diagonal value for thermal averaging.
        E_N[n,N,N]
    - E0: vector Exp(-beta*E0) Used for thermal average E_N^(n).
        E0[T,N]

    all matrices have N*N (N=len(linrComb)) 
'''
class generalorder:
    def __init__(self,nmode):
        np.set_printoptions(precision=12)
        Ehbykb = 3.1577464*100000
        self.Ehcs = Ehbykb
        #Temprt = np.array([100,10**2.5,1000,10**3.5,10000,100000,1000000,10000000,100000000])
        #bug: 10000 result is changing with the beta size.
        #Temprt = np.array([10,1000,1234,2354])
        #Temprt = np.array([10,1000,10000])#,10000,100000,1000000])
        Temprt = np.array([10,30,60,100,300,600,1000,3000,6000,10000,100000,1000000,10000000,100000000])#,10000,100000,1000000])
        #Temprt = 100
        Temprt = Temprt/Ehbykb
        beta = 1/Temprt
        w_omega,FCQ3,FCQ4,FCQ3scale,FCQ4scale = self.readSindoPES("./data/prop_no_3.hs",nmode)
        maxn = 16 
        maxorder = 5
        norder = 8 

        Lambd = 0.01
        self.dictlambd = {1:0,-3*Lambd:1,-2*Lambd:2,-Lambd:3,0:4,Lambd:5,2*Lambd:6,3*Lambd:7}
        #preparation 
        linrComb = self.loopfn(nmode,maxn)
        self.FCIsize = len(linrComb)
        Evlst = self.EvaluationList(nmode,w_omega,maxn,maxorder)
        #initialize the a, E_N^(n) , Omega^(n) matrix
        self.E_N = np.zeros((norder+1,self.FCIsize,self.FCIsize))
        self.Omg = np.zeros((norder+1,len(Temprt)))# like perturbation order of 4, then 0,1,2,3,4th order we have 5 in total
        self.U = np.zeros((norder+1,len(Temprt)))# like perturbation order of 4, then 0,1,2,3,4th order we have 5 in total
        #also initialize self.E_N[0] here
        self.Omg[0],self.E0exp = self.EN0(w_omega,maxn,Evlst,linrComb,Temprt)
        self.T_EN_d= np.sum(self.E0exp,axis=1)#denominator in the size of Temperature
        self.U[0] = self.thermalavg(self.E_N[0,:,:])

        #XXX browse the directory
        self.globlist = glob.glob("./data/*.npy")
        print(self.globlist)

        #XXX generate V and DE and store in npy file
        V_NMpath =  "./data/V_NM"+str(maxn)+".npy"
        DE_MNpath = "./data/DE_MN"+str(maxn)+".npy"
        if(V_NMpath in self.globlist):
            self.V_NM = np.load("./data/V_NM"+str(maxn)+".npy")
        else:
            self.V_EN_matrix_build(w_omega,maxn,Evlst,linrComb,FCQ3,FCQ4,nmode,maxorder)

        if(DE_MNpath in self.globlist):
            self.DE_MN = np.load("./data/DE_MN"+str(maxn)+".npy")
        else:
            self.V_EN_matrix_build(w_omega,maxn,Evlst,linrComb,FCQ3,FCQ4,nmode,maxorder)
        
        #XXX build a matrix EN^(n) matrix iteratively and store
        #self.a_iteration(norder,linrComb,w_omega,maxn)
        norderEpath = './data/EN_'+str(norder)+'_N'+str(maxn)+'.npy'
        if( norderEpath in self.globlist):
            print("find all the EN files then read in")
            #just read EN no longer need a
            self.E_Nreadin(norder,maxn)
        else:
            self.a_iteration(norder,linrComb,w_omega,maxn)
        
        #XXX do Omega calculation based on recursion
        #np.save(Omgpath,self.Omg)
        '''
        Omgpath = "./data/Omg_"+str(norder)+".npy"
        if(Omgpath in self.globlist):
            print("find Omg "+str(norder)+" file read  in")
            self.Omg= np.load(Omgpath)
        else:
            self.OmgCal(norder,beta)
            np.save(Omgpath,self.Omg)
        print("_________________________________________________________")
        print("general order 0 omega is",self.Omg[0])
        '''
        self.OmgCal(norder,beta)
        
        #XXX do U calculation based on recursion
        '''
        Upath = "./data/U_"+str(norder)+".npy"
        if(Upath in self.globlist):
            print("find U "+str(norder)+" file read  in")
            self.U = np.load(Upath)
        else:
            self.UCal(norder,beta)
            np.save(Upath,self.U)
        print("_________________________________________________________")
        print("general order 0 U is",self.U[0])
        print("lets see if we can get it done by 10")
        '''
        self.UCal(norder,beta)
        
        #XXX VCI calculation

        #XXX generate H0 matrix and store in npy.
        H0path = "./data/H0_matrix.npy"
        if(H0path in self.globlist):
            self.H0 = np.load("./data/H0_matrix.npy")
        else:
            self.H0matrix(w_omega,maxn,Evlst,linrComb,nmode,maxorder)

        #XXX FCI calculation (checked)
        
        FCI = self.VCIlambda(1)
        OmgFCI,UFCI = self.eachpoint(FCI,beta)
        print("_________FCI result is__________")
        print("Omega is ",OmgFCI)
        print("U is ",UFCI)
        #XXX test rewrite
        #Omgnew = self.newseven(beta,Lambd)


        Omg7T,U7T = self.sevenpoint(beta,Lambd)
        #7 perturbation orders 0-6
        Omg_VCIresults = np.zeros((7,Temprt.shape[0]))
        Omg_VCItemp = self.derivative(Omg7T,Lambd)
        Omg_VCIresults[0,:] = Omg7T[3,:]
        Omg_VCIresults[1:7,:] = Omg_VCItemp
        print("-_____________________")
        print("VCI result of Omega is order/ temprature")
        for i in range(len(Temprt)):
            print("at "+str(Temprt[i]*Ehbykb)+" K temp, we have 0-6 order results:")
            print("0-6 order is ", Omg_VCIresults[:,i])
        
        U_VCIresults = np.zeros((7,Temprt.shape[0]))
        U_VCItemp = self.derivative(U7T,Lambd)
        U_VCIresults[0,:] = U7T[3,:]
        U_VCIresults[1:7,:] = U_VCItemp
        print("-_____________________")
        print("VCI result of U is order/ temprature")
        for i in range(len(Temprt)):
            print("+++++++++")
            print("at "+str(Temprt[i]*Ehbykb)+" K temp, we have 0-6 order results:")
            print("0-6 is ", U_VCIresults[:,i])
         
        '''
        #XXX save csv
        #for Omg and U changing with 10 - 100000 Temperature change
        for i in range(3):# 0 1 2 perturbation order
            fname ="./data/Omg_01"+str(i)+"_perT.csv"
            with open(fname,'w') as csvfile:
                csvwriter = csv.writer(csvfile,delimiter ='&')
                for j in range(Temprt.shape[0]):
                    #general order first then FCI 
                    csvwriter.writerow([self.Omg[i,j],Omg_VCIresults[i,j]])

        for i in range(3):# 0 1 2 perturbation order
            fname ="./data/U_01"+str(i)+"_perT.csv"
            with open(fname,'w') as csvfile:
                csvwriter = csv.writer(csvfile,delimiter ='&')
                for j in range(Temprt.shape[0]):
                    #general order first then FCI 
                    csvwriter.writerow([self.U[i,j],U_VCIresults[i,j]])
        '''
        #for 0 - 8 orders results
        #idxlist= [10,11,12,13]
        #idxlist= [i]
        
        for idx in range(14):
            fname= "./data/GenOrder_OU_01"+str(idx)+".csv"
            with open(fname,'w') as csvfile:
                csvwriter = csv.writer(csvfile,delimiter ='&')
                for j in range(norder+1):
                    if(j<=6):
                        csvwriter.writerow([self.Omg[j,idx],Omg_VCIresults[j,idx],self.U[j,idx],U_VCIresults[j,idx]])
                    else: 
                        csvwriter.writerow([self.Omg[j,idx],self.U[j,idx]])
                csvwriter.writerow([np.sum(self.Omg[:,idx]),np.sum(self.U[:,idx])])
                csvwriter.writerow([OmgFCI[idx],UFCI[idx]])

    def newseven(self,beta,Lambd):
        n3 = self.VCIlambda(-3*Lambd)
        n2 = self.VCIlambda(-2*Lambd)
        n1 = self.VCIlambda(-Lambd)
        n0 = self.VCIlambda(0)
        p1 = self.VCIlambda(Lambd)
        p2 = self.VCIlambda(2*Lambd)
        p3 = self.VCIlambda(3*Lambd)
        #print([n3,n2,n1,n0,p1,p2,p3])
        sevenpoint = np.zeros((7,beta.shape[0]))
        for i in range(beta.shape[0]):
            On3 = self.neweach(n3,beta[i])
            On2 = self.neweach(n2,beta[i])
            On1 = self.neweach(n1,beta[i])
            On0 = self.neweach(n0,beta[i])
            Op1 = self.neweach(p1,beta[i])
            Op2 = self.neweach(p2,beta[i])
            Op3 = self.neweach(p3,beta[i])
            sevenpoint[:,i]= np.array([On3,On2,On1,On0,Op1,Op2,Op3])

            #each order
            print("for temperature "+ str(1/beta[i]*self.Ehcs)+ " T")
            O1 = (-1/60*On3 + 3/20*On2 -3/4* On1 + 3/4*Op1 -3/20* Op2 + 1/60* Op3)/Lambd
            O2 = (1/90*On3 - 3/20*On2 +3/2* On1 - 49/18*On0 + 3/2*Op1 - 3/20* Op2 + 1/90* Op3)/Lambd/Lambd/2
            O3 = (1/8*On3 - On2 +13/8* On1  - 13/8*Op1 +  Op2 - 1/8* Op3)/Lambd/Lambd/Lambd/6
            O4 = (-1/6*On3 + 2*On2 -13/2* On1+28/3*On0 - 13/2*Op1 + 2* Op2 - 1/6* Op3)/Lambd/Lambd/Lambd/Lambd/24
            O5 = (-1/2*On3 + 2*On2 -5/2* On1  + 5/2*Op1 -  2*Op2 + 1/2* Op3)/Lambd/Lambd/Lambd/Lambd/Lambd/120
            O6 = (On3 - 6*On2 +15* On1-20*On0 + 15*Op1 - 6* Op2 +  Op3)/Lambd/Lambd/Lambd/Lambd/Lambd/Lambd/720
            arry= [O1,O2,O3,O4,O5,O6]
            print(arry)
        test = self.derivative(sevenpoint,Lambd)
        for i in range(len(beta)):
            print("at "+str(1/beta[i]*self.Ehcs)+" K temp,  results:")
            print( test[:,i])


    def sevenpoint(self,beta,lambd):
        #it can be matrilized here but I didn't since it is easier to debug
        sevenlambd = [-3*lambd,-2*lambd,-lambd,0,lambd,2*lambd,3*lambd]
        Omg7T = np.zeros((7,beta.shape[0]))
        U7T = np.zeros((7,beta.shape[0]))
        for i in range(7):
            w = self.VCIlambda(sevenlambd[i])
            Omg7T[i,:],U7T[i,:] = self.eachpoint(w,beta)
        return Omg7T,U7T

    def neweach(self,w,beta):
        Xi = np.sum(np.exp(-w*beta))
        Omg = -np.log(Xi)/beta
        return Omg

    def eachpoint(self,w,beta):
        Xiarray = np.exp(-beta[:,np.newaxis]*w)
        Xi = np.sum(Xiarray,axis=1)#Xi in the size of Temperature
        Omg = - np.log(Xi)/beta
        Unu = w*Xiarray
        U = np.sum(Unu,axis=1)/Xi# in the size of temperature 
        S = beta*(U - Omg)
        return Omg,U
        

    def VCIlambda(self,lambd):
        VCIpath = "./data/VCIlambd_01_eigenvalue_"+str(self.dictlambd[lambd])+".npy"
        if VCIpath in self.globlist:
            w = np.load(VCIpath)
        else:
            VCI = self.H0 + self.V_NM*lambd
            w,v = linalg.eigh(VCI)
            np.save(VCIpath,w)
        #VCI = self.H0 + self.V_NM*lambd
        #w,v = linalg.eigh(VCI)
        return w



    #sevenpointPerT is seven * temperature size, orderbyseven is perturbation order *seven.
    #the result is perturbation order * temperature size
    def derivative(self,sevenpointPerT,Lambd):
        # the order of seven input is f(-3) f(-2) f(-1) f(0) f(1) f(2) f(3)
        orderbyseven = np.zeros((6,7))#perturbation order * seven points coefficient matrix
        #orderbyseven[0,:] = np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60])/Lambd
        #orderbyseven[1,:] = np.array([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90])/Lambd/Lambd/2
        #orderbyseven[2,:] = np.array([1/8,-1,13/8,0,-13/8,1,-1/8])/Lambd/Lambd/Lambd/6
        #orderbyseven[3,:] = np.array([-1/6,2,-13/2,28/3,-13/2,2,-1/6])/Lambd/Lambd/Lambd/Lambd/24

        #orderbyseven[4,:] = np.array([-1/2,2,-5/2,0,5/2,-2,1/2])/Lambd/Lambd/Lambd/Lambd/Lambd/120

        #orderbyseven[5,:] = np.array([1,-6,15,-20,15,-6,1])/Lambd/Lambd/Lambd/Lambd/Lambd/Lambd/720
        orderbyseven[0,:] = np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60])
        orderbyseven[1,:] = np.array([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90])
        orderbyseven[2,:] = np.array([1/8,-1,13/8,0,-13/8,1,-1/8])
        orderbyseven[3,:] = np.array([-1/6,2,-13/2,28/3,-13/2,2,-1/6])

        orderbyseven[4,:] = np.array([-1/2,2,-5/2,0,5/2,-2,1/2])

        orderbyseven[5,:] = np.array([1,-6,15,-20,15,-6,1])
        result = np.dot(orderbyseven,sevenpointPerT)
        result[0,:] = result[0,:]/Lambd
        result[1,:] = result[1,:]/Lambd/Lambd/2
        result[2,:] = result[2,:]/Lambd/Lambd/Lambd/6
        result[3,:] = result[3,:]/Lambd/Lambd/Lambd/Lambd/24
        result[4,:] = result[4,:]/Lambd/Lambd/Lambd/Lambd/Lambd/120
        result[5,:] = result[5,:]/Lambd/Lambd/Lambd/Lambd/Lambd/Lambd/720
        return result

    def H0matrix(self,w_omega,maxn,Evlst,linrComb,nmode,maxorder):
        self.H0 = np.zeros((self.FCIsize,self.FCIsize))
        for i in range(len(linrComb)):
            Nhs = np.array(linrComb[i])
            for j in range(i,len(linrComb)):# M runs over N half
                sumofoperator = 0.0
                Mhs = np.array(linrComb[j])
                n = np.maximum(Nhs,Mhs)
                diff = np.abs(Nhs-Mhs)
                for optidx in range(nmode):# three kinetic operators
                    #parse each mode in |xxxx>
                    multply = 1 #the multiply product of each mode
                    multplyw = 1
                    for modeidx in range(nmode):
                        if (modeidx == optidx and diff[modeidx] < maxorder): #the operator works on the correspoding Q
                            multply *= -0.5*Evlst[modeidx,5,n[modeidx],diff[modeidx]]
                            multplyw *= 0.5*Evlst[modeidx,2,n[modeidx],diff[modeidx]]
                        else: #check if they are orthogonal if not, then zero
                            if (diff[modeidx]!=0):
                                multply *= 0 
                                multplyw *=0
                                break
                    multplyw *=(w_omega[optidx]**2)
                    sumofoperator += multply
                    sumofoperator += multplyw
                self.H0[i,j]=self.H0[j,i] = sumofoperator
        np.save("./data/H0_matrix.npy",self.H0)





    '''
    We don't do Omg and U together, although it would be faster but it would be messy and buggy.
    It needs run OmgCal first as we will use self.Omg
    We will see how far we can reach 
    '''
    def UCal(self,norder,beta):
        #when i = 1  it is for the first order Omega and it goes from 1- n
        for i in range(1,norder+1):
            T_EN = self.thermalavg(self.E_N[i,:,:])
            allbetaterm = np.zeros(beta.shape)#used for summing all the beta terms in the size of temperature
            for j in range(1,i+1): #total number of beta terms adding them to T_EN
                permutelist = self.permute_Betaterm(i,j+1)#j+1 is the number of E_N in each beta term
                extraE0 = self.permuteE0(i,j)
                betatemp = np.zeros(beta.shape)#used for summing permutation in the size of temperature
                for k in permutelist+extraE0:#sum all the permutation
                    ENtemp = np.identity(self.FCIsize)#used for multiply E_N^i E_N^j E_N^k...
                    Utemp = np.ones(beta.shape[0])
                    #for each permutation we iterate and thermal average them
                    for l in range(j+1):
                        ENtemp*=self.E_N[k[l],:,:]
                        if(l==j):
                            Utemp*= self.U[k[l],:]
                        else:
                            Utemp*= self.Omg[k[l],:]
                    betatemp += self.thermalavg(ENtemp) - Utemp
                allbetaterm+= np.power(-beta,j)/math.factorial(j) * betatemp 
            T_EN += allbetaterm
            self.U[i] = T_EN
            print("__________________________")
            print("The "+str(i)+" order U^{%} is")
            print(T_EN)

    '''
    calculate the norder (0-norder+1) Omg based on the Omega recursion.
    '''
    def OmgCal(self,norder,beta):
        #do <EN^(n)> first, pass!
        #when i = 1  it is for the first order Omega and it goes from 1- n
        for i in range(1,norder+1):
            T_EN = self.thermalavg(self.E_N[i,:,:])
            print("the <ENn> itself is",T_EN)
            if(i>1):
                allbetaterm = np.zeros(beta.shape)#used for summing all the beta terms in the size of temperature
                #XXX debug here.
                print("for n of " + str(i)+" xxxxxxx")
                for j in range(1,i): #total number of beta terms adding them to T_EN
                    permutelist = self.permute_Betaterm(i,j+1)#j+1 is the number of E_N in each beta term
                    #print("for the "+str(j)+"th beta term we have:")
                    #print(permutelist)
                    betatemp = np.zeros(beta.shape)#used for summing permutation in the size of temperature
                    for k in permutelist:#sum all the permutation
                        ENtemp = np.identity(self.FCIsize)#used for multiply E_N^i E_N^j E_N^k...
                        Omgtemp = np.ones(beta.shape)
                        #for each permutation we iterate and thermal average them
                        for l in range(j+1):
                            ENtemp*=self.E_N[k[l],:,:]
                            Omgtemp*= self.Omg[k[l],:]
                        betatemp += self.thermalavg(ENtemp) - Omgtemp
                    allbetaterm+= np.power(-beta,j)/math.factorial(j+1) * betatemp 
                T_EN += allbetaterm
            self.Omg[i] = T_EN

            print("__________________________")
            print("The "+str(i)+" order Omega^{%} is")
            print(T_EN)
    '''
    this is the function to generate the indices for the beta term in Omega recursion
    '''
    #norder is the real n not minus 1
    def permute_Betaterm(self,norder,nofterm):
        temp = list(itl.product(range(1,norder),repeat = nofterm))
        tempfixed = [i for i in temp if sum(i) == norder]
        return tempfixed
    def permuteE0(self,norder,nofterm):
        temp = list(itl.product(range(1,norder+1),repeat = nofterm))
        tempfixed = [i+(0,) for i in temp if sum(i) == norder]
        return tempfixed


    '''
    return the <input> in the size of temperature
    '''
    def thermalavg(self, inputmatrix):
        return np.sum(np.diag(inputmatrix)*self.E0exp,axis=1)/self.T_EN_d
    #a^(n-1) is for EN^(n) evaluation
    #so for each order we evaluate EN first then calculate a
    #n is from 1 - norder with index of 0 - norder-1

    def E_Nreadin(self,norder,maxn):
        print("find EN "+ str(norder) + " just read in every EN")
        for i in range(1,norder+1):
            Ethisorder = './data/EN_'+str(i)+'_N'+str(maxn)+'.npy'
            self.E_N[i,:,:] = np.load(Ethisorder)

    def a_iteration(self,norder,linrComb,w_omega,maxn):
        self.a = np.zeros((norder,self.FCIsize,self.FCIsize))
        #norder >=1  and end at n-1
        for i in range(1,norder):
            Ethisorder = './data/EN_'+str(i)+'_N'+str(maxn)+'.npy'
            athisorder = './data/a_'+str(i)+'_N'+str(maxn)+'.npy'
            if(Ethisorder in self.globlist):
                print("find EN "+ str(i) + " just read in")
                self.E_N[i,:,:] = np.load(Ethisorder)
            else:
                self.E_N_build(i,maxn)

            if(athisorder in self.globlist):
                print("find a "+ str(i) + " just read in")
                self.a[i,:,:] = np.load(athisorder)
            else:
                self.a_build(i,maxn)
            #self.E_N_build(i,maxn)
            #self.a_build(i,maxn)
        norderEpath = './data/EN_'+str(norder)+'_N'+str(maxn)+'.npy'
        if( norderEpath in self.globlist):
            self.E_N[norder,:,:] = np.load(norderEpath)
        else:
            self.E_N_build(norder,maxn)
        #self.E_N_build(norder,maxn)

    def E_N_build(self,i,maxn):
        if(i==1):
            self.E_N[i,:,:] = self.V_NM
        else:
            self.E_N[i,:,:] = np.dot(self.V_NM,self.a[i-1,:,:])#when i = 2 ,a[2] is the a^(2).
        Epath = './data/EN_'+str(i)+'_N'+str(maxn)+'.npy'
        np.save(Epath,self.E_N[i,:,:])

    def a_build(self,i,maxn):
        if(i==1):
            self.a[i,:,:] = np.multiply(self.DE_MN,self.V_NM)#element-wise multiplication
        else:
            temp = np.dot(self.V_NM,self.a[i-1,:,:])
            for j in range(1,i):
                temp1 = np.diag(np.diag(self.E_N[i-j,:,:]))
                temp -= np.dot(self.a[j,:,:],temp1)
                #temp -= np.dot(self.a[j,:,:],self.E_N[i-j,:,:])
            self.a[i,:,:] = np.multiply(self.DE_MN,temp)
        apath = './data/a_' +str(i)+'_N'+str(maxn)+'.npy'
        np.save(apath,self.a[i,:,:])
        


    def V_EN_matrix_build(self,w_omega,maxn,Evlst,linrComb,FCQ3,FCQ4,nmode,maxorder):
        V_NM = np.zeros((len(linrComb),len(linrComb)))
        DE_MN = np.zeros(V_NM.shape)
        for i in range(len(linrComb)):#N, and E_N0sum is done here
            #E_Nsum =np.zeros(beta.shape) 
            #E_N0sum=0.0
            #for E0idx in range(nmode):
            #    E_N0sum += linrComb[i][E0idx] * w_omega[E0idx]
            #E_Nsum = 0.0#sum for each N, needing sum of all M
            #delta E
            Nhs = np.array(linrComb[i])
            for j in range(len(linrComb)):# M runs over N
                Mhs = np.array(linrComb[j])
                diffreal = np.sum((Mhs-Nhs)*w_omega)
                if(diffreal == 0):
                    DE_MN[i,j] = 0
                else:
                    DE_MN[i,j] = 1/diffreal
            for j in range(i,len(linrComb)):# M runs over N half
                sumofoperator = 0.0
                Mhs = np.array(linrComb[j])
                n = np.maximum(Nhs,Mhs)
                diff = np.abs(Nhs-Mhs)
                for ii in range(nmode):
                    for jj in range(nmode):
                        for kk in range(nmode):
                            multplyQ3 = 1
                            eachcount3 = Counter([ii,jj,kk])
                            for modeidx3 in range(nmode):
                                if(diff[modeidx3]>=maxorder-1):#magic number we set maxorder = 5 (number of operators:partial Q, Q, Q2, Q3, Q4.)
                                    multplyQ3 = 0
                                    break
                                numberofmodeinFC = eachcount3[modeidx3]
                                multplyQ3 *= Evlst[modeidx3,numberofmodeinFC,n[modeidx3],diff[modeidx3]]
                            multplyQ3*=FCQ3[ii,jj,kk]/6
                            sumofoperator += multplyQ3
                            for ll in range(nmode):
                                multplyQ4 = 1
                                eachcount = Counter([ii,jj,kk,ll])
                                for modeidx in range(nmode):
                                    if(diff[modeidx]>=maxorder):
                                        multplyQ4 = 0
                                        break
                                    numberofmodeinFC = eachcount[modeidx]
                                    multplyQ4 *= Evlst[modeidx,numberofmodeinFC,n[modeidx],diff[modeidx]]
                                multplyQ4*=FCQ4[ii,jj,kk,ll]/24
                                sumofoperator += multplyQ4
                V_NM[i,j] = V_NM[j,i]=sumofoperator
            np.save("./data/V_NM"+str(maxn)+".npy",V_NM)
            np.save("./data/DE_MN"+str(maxn)+".npy",DE_MN)


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
        Xisum =np.zeros(beta.shape) 
        for i in range(len(linrComb)):
            E_Nsum = 0.0
            for j in range(len(linrComb[i])):
                E_Nsum += (linrComb[i][j]+0.5) * w_omega[j]
            self.E_N[0,i,i] = E_Nsum
            Xisum += np.exp(-beta*E_Nsum)
        Omg = - np.log(Xisum)/beta

        #expE0 generating
        E0 = np.sum(linrComb*w_omega,axis=1)
        E0exp = np.exp(-beta[:,np.newaxis]*E0)
        return Omg,E0exp



    def EvaluationList(self,nmode,w_omega,maxn,maxorder):
        #I used the combination to determine which operator can give us result.
        #The 1st is to indicate which normal mode is it.
        #The 2nd is to indicate which operator: 0-4 : 0, Q, Q^2, Q^3, Q^4. Here we used QFF so the max order of operator is 4 and total number is 4
        #The 3rd is to the which level n is, n is the bigger one than n' 
        #The 4th is the difference between n and n'
        Evlst = np.zeros((nmode,maxorder+1,maxn,maxorder))
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
                Evlst[i,5,n,0] = - w_omega[i]*(n+0.5)
                Evlst[i,5,n,2] = w_omega[i]*math.sqrt(n*(n-1))/2
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
a= time.time()
test = generalorder(3) 
print("time is: ")
print((time.time()-a)," /sec", (time.time()-a)/60, " /min")
