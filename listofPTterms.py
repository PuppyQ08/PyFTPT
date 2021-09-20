import numpy as np
import sympy as sym
import sys


class ListofPTterms:
    def __init__(self,diff,fc,expression):
        #we will have only one diff for each data sturcutre class, even at second step we merge two terms with reverse sign diff, after merging, they all have same diff.
        self.diff = diff
        #operator for each term, at second steps, two terms with reverse sign diff should have same operator list, otherwise, report error
        self.fclst = [fc]
        #expressions for each operator list
        self.explst = [expression]

    def mergesamediff(self,diff,fc,expression):
        #check diff
        if (self.diff == diff):
            print("checking pass")
            self.fclist.append(fc)
            self.explst.append(expression)
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

    





