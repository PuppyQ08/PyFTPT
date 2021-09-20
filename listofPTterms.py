import numpy as np
import sympy as sym
import sys


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
        fclst = PTterms.fc
        explst = PTterms.explst
        diff = PTterms.diff
        #check diff
        if (self.diff == diff):
            print("same diff merging running")
            self.fclst = self.fclst + fclst
            self.explst = self.explst + explst
        else if (np.array_equal(np.array(self.diff),-1*np.array(diff))):
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

    def printout(self,whichstage):
        print("The diff is ",self.diff," ",[self.diffsymlst[x] for x in self.diff])
        print("each of corresponding terms")
        if(whichstage==0):
            for i in range(len(self.fclst)):
                print("The fc is", self.fclst[i], " ",[self.operatorlst[x] for x in self.fclst[i]])
                print("The expression is", self.explst[i]) 
        if(whichstage==1):
            for i in range(len(self.fc_samediff)):
                print("The fc is", self.fc_samediff[i], " ",[self.operatorlst[x] for x in self.fc_samediff[i]])
                print("The expression is", self.explst_samediff[i]) 


    





