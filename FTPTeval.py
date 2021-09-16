import sympy as sym
import itertools

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
D0 = sym.symbols('D0')
D1 = sym.symbols('D1')
D2 = sym.symbols('D2')
D3 = sym.symbols('D3')
D4 = sym.symbols('D4')
D1n = sym.symbols('D1n')
D2n = sym.symbols('D2n')
D3n = sym.symbols('D3n')
D4n = sym.symbols('D4n')

class ThermalAvg:
    def __init__(self):
        #start with one mode excited wave function
        #operator combinations 
        self.fc3rd_origin,self.fc4th_origin = self.fcoperator()
        #difference combinations
        self.diff3rd_origin,self.diff4th_orgin = self.diffgen()
        self.diffsymlst = [D0,D1,D2,D3,D4,D4n,D3n,D2n,D1n]
        self.thermAverules = self.thermAvgeval()#return a list of dict
        self.BornHuangrules = self.BHruleeval()#return a list of dict

    def BHrulehelper(self,Qm,Im,wm):
        tempdict = {D0*Qm: 

    def BHruleeval(self):
        #Dx*Qm**y - > Im


    def thermAvgeval(self):
        #Im -> fm

    def diffgen(self):
        difflst = [0,1,2,3,4,-4,-3,-2,-1]
        iterdiff3rd = list(itertools.product(difflst,repeat=3))
        iterdiff4th = list(itertools.product(difflst,repeat=4))
        return iterdiff3rd,iterdiff4th

    def fcoperator(self):
        #|1 0 2> for QiQk**2 the number is the multiplicity of each mode
        lst3rd = [0,1,2,3]
        iter3rdtemp = list(itertools.product(lst3rd,repeat=3))
        iter3rd = []
        #filter out those with sum = 3
        for i in range(len(iter3rdtemp)):
            if (sum(list(iter3rdtemp[i])) == 3):
                iter3rd.append(iter3rdtemp[i])
        #same with 4th:
        lst4th = [0,1,2,3,4]
        iter4thtemp = list(itertools.product(lst4th,repeat=4))
        iter4th = []
        #filter out those with sum = 3
        for i in range(len(iter4thtemp)):
            if (sum(list(iter4thtemp[i])) == 4):
                iter4th.append(iter4thtemp[i])
        print(iter3rd)
        print(iter4th)
        return iter3rd,iter4th



test = ThermalAvg()
