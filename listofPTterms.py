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
        elif (np.array_equal(np.array(self.diff),-1*np.array(diff))):
            print("reverse diff merging running")
            # TBD
        else:
            sys.exit("first merge need two terms with same diff")

    #iterate through the fc and exp list to obtain <Phi|V|Phi>**2 only for second order now, probably generalize to third order in the future.
    def iterate_samediff(self):
        self.fc_samediff = []
        self.explst_samediff = []
        for i in range(len(self.diff)):
            for j in range(len(self.diff)):
                self.fc_samediff.append(self.diff[i]+self.diff[j])
                self.explst_samediff.append(sym.simplify(self.explst[i]*self.explst[j]))

    def fclst_Qform(self,lstipt):
        fcprintlst = [0]*len(lstipt)
        for j in range(len(lstipt)):
            #here fclst[i] is (1,0,2) like, so would be QiQk**2
            fcprintlst[j] =self.operatorlst[j]**lstipt[j]
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
        if(whichstage==1):
            for i in range(len(self.fc_samediff)):
                print("The fc is", self.fc_samediff[i], " ",[self.operatorlst[x] for x in self.fc_samediff[i]])
                print("The expression is", self.explst_samediff[i]) 


    





