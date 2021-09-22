import numpy as np
import sympy as sym
import sys

Ii = sym.symbols('Ii')
Ij = sym.symbols('Ij')
Ik = sym.symbols('Ik')
Il = sym.symbols('Il')
wi = sym.symbols('wi')
wj = sym.symbols('wj')
wk = sym.symbols('wk')
wl = sym.symbols('wl')
fi = sym.symbols('fi')
fj = sym.symbols('fj')
fk = sym.symbols('fk')
fl = sym.symbols('fl')
Qi = sym.symbols('Qi')
Qj = sym.symbols('Qj')
Qk = sym.symbols('Qk')
Ql = sym.symbols('Ql')
D0 = sym.symbols('D0')
D1 = sym.symbols('D1')
D2 = sym.symbols('D2')
D3 = sym.symbols('D3')
D4 = sym.symbols('D4')
D1n = sym.symbols('D1n')
D2n = sym.symbols('D2n')
D3n = sym.symbols('D3n')
D4n = sym.symbols('D4n')

class ListofPTterms:
    def __init__(self,diff,fc,expression):
        self.operatorlst = [Qi,Qj,Qk,Ql]
        self.freqlst = [wi,wj,wk,wl]
        self.qtnumberlst = [Ii,Ij,Ik,Il]
        self.BEfactorlst = [fi,fj,fk,fl]
        self.diffsymlst = [D0,D1,D2,D3,D4,D4n,D3n,D2n,D1n]
        #we will have only one diff for each data sturcutre class, even at second step we merge two terms with reverse sign diff, after merging, they all have same diff.
        self.diff = diff
        #operator for each term, at second steps, two terms with reverse sign diff should have same operator list, otherwise, report error
        self.fclst = [fc]
        #expressions for each operator list
        self.explst = [expression]

    def mergesamediff(self,PTterms):
        fclst = PTterms.fclst
        explst = PTterms.explst
        diff = PTterms.diff
        #check diff
        if (self.diff == diff):
            print("same diff merging running")
            self.fclst = self.fclst + fclst
            self.explst = self.explst + explst
        else:
            sys.exit("first merge need two terms with same diff")

    def diffexp(self,iptlst):
        #iptlst is (1,0,-1) -> -wi+wk D1 is -1*wi
        diffexpreturn = 0
        for i in range(len(iptlst)):
            diffexpreturn -= iptlst[i]*self.freqlst[i]
        return diffexpreturn

    def mergereversediff(self,PTterms):
        fclst_sd = PTterms.fclst_samediff
        explst_sd = PTterms.explst_samediff
        self.explst_revers = []
        diff = PTterms.diff
        if (np.array_equal(np.array(self.diff),-1*np.array(diff))):
            print("reverse diff merging running")
            for i in range(len(fclst_sd)):
                if(np.array_equal(np.array(fclst_sd[i]),np.array(self.fclst_samediff[i]))):
                    #Here: we introduce the denomenator
                    self.explst_revers.append(self.explst_samediff[i]/self.diffexp(self.diff)+explst_sd[i]/self.diffexp(diff))
        else:
            sys.exit("second merge need two terms with reverse diff")
    #substitute Im with fm 
    def subsIm_fm(self,thermAverules):
        leng =len(self.explst_revers)
        self.explst_fm = [0]*leng
        #expand first:
        for i in range(leng):
            self.explst_fm[i] = sym.expand(sym.expand(self.explst_revers[i]).subs(thermAverules))



    #iterate through the fc and exp list to obtain <Phi|V|Phi>**2 only for second order now, probably generalize to third order in the future.
    def iterate_samediff(self):
        self.fclst_samediff = []
        self.explst_samediff = []
        for i in range(len(self.fclst)):
            for j in range(len(self.fclst)):
                self.fclst_samediff.append(self.fclst[i]+self.fclst[j])
                self.explst_samediff.append(sym.simplify(self.explst[i]*self.explst[j]))

    def fclst_Qform(self,lstipt):
        leng = len(lstipt)
        fcprintlst = [0]*leng
        if (leng<=4):
            for j in range(leng):
                #here fclst[i] is (1,0,2) like, so would be QiQk**2
                fcprintlst[j] =self.operatorlst[j]**lstipt[j]
        else:
            #this is specifically for 3rd and 4th (max) order force constants, so I used this trick
            for j in range(leng):
                #here fclst_samediff[i] is (1,0,2,2,0,1) like, so would be QiQk**2 Qi**2 Qk
                fcprintlst[j] =self.operatorlst[int(j%(leng/2))]**lstipt[j]

        return fcprintlst

    def printout(self,whichstage):
        print("++++++++++++++++++++++++++++++")
        print("The diff is ",self.diff," ",[self.diffsymlst[x] for x in self.diff])
        print("each of corresponding terms")
        if(whichstage==0):
            for i in range(len(self.fclst)):
                fcprintlst = self.fclst_Qform(self.fclst[i])
                print("The fc is ",self.fclst[i]," ",fcprintlst)
                print("The expression is", self.explst[i]) 
                print("---------------")
        #print the term after first merge of same diff. 
        if(whichstage==1):
            for i in range(len(self.fclst_samediff)):
                fcprintlst = self.fclst_Qform(self.fclst_samediff[i])
                print("The fc is", self.fclst_samediff[i], " ",fcprintlst)
                print("The expression is", self.explst_samediff[i]) 
        #print the term after Im fm substitution.
        if (whichstage == 2):
            for i in range(len(self.fclst_samediff)):
                fcprintlst = self.fclst_Qform(self.fclst_samediff[i])
                print("The fc is", self.fclst_samediff[i], " ",fcprintlst)
                print("The revers expression is", self.explst_revers[i]) 
                print("The subs expression is", self.explst_fm[i]) 




    





