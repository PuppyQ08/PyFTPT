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

class ThermalAvg:
    def __init__(self):
        #operator combinations 
        self.fc3rd_origin,self.fc4th_origin = self.fcoperator()
        #difference combinations
        self.diff3rd_origin,self.diff4th_orgin = self.diffgen()
        #list of each symbols
        self.operatorlst = [Qi,Qj,Qk,Ql]
        self.freqlst = [wi,wj,wk,wl]
        self.qtnumberlst = [Ii,Ij,Ik,Il]
        self.BEfactorlst = [fi,fj,fk,fl]
        self.diffsymlst = [D0,D1,D2,D3,D4,D4n,D3n,D2n,D1n]
        #two rules for substituding
        self.thermAverules = self.thermAvgeval()#return a list of dict
        self.BornHuangrules = self.BHruleeval()#return a list of dict
        #start with one mode excited wave function
        self.onemodewvfn()


    def onemodewvfn(self):
        #the one mode exicted wave function thermal average means two things:
        #1, the operator group should have non-zero first element. 
        #2, the difference group should have non-zero first element and zero rest elements.
        fc3rd1mode = []
        diff3rd1mode = []
        #do 3rd first
        for i in range(len(self.fc3rd_origin)):
            if (self.fc3rd_origin[i][0] != 0):
                fc3rd1mode.append(self.fc3rd_origin[i])
        for i in range(len(self.diff3rd_origin)):
            if (self.diff3rd_origin[i][0]!=0 and self.diff3rd_origin[i][1]==0 and self.diff3rd_origin[i][2]==0):
                diff3rd1mode.append(self.diff3rd_origin[i])
        #do it
        numorder = 3
        for i in range(len(diff3rd1mode)):
            for j in range(len(fc3rd1mode)):
                valueofeachmode = 1
                for modeidx in range(numorder):
                    tempvalue = self.Dx_Qm(diff3rd1mode[i],fc3rd1mode[j],modeidx)
                    tempvaluesub1 = tempvalue.subs(self.BornHuangrules[modeidx])#sub DxQm 
                    tempvaluesub2 = tempvaluesub1.subs(self.BornHuangrules[4])#sub Dx itself
                    if (tempvaluesub2 == 0 ):
                        valueofeachmode = 0
                        break
                if (valueofeachmode != 0):
                    print("found")
                    print(diff3rd1mode[i],fc3rd1mode[j])

    def Dx_Qm(self,diff,fc,modeidx):
        eachDxQm = self.diffsymlst[diff[modeidx]]*self.operatorlst[modeidx]**fc[modeidx]
        return eachDxQm

        


    def thermAvghelper(self,Qm,Im,fm):
        tempdict = {Im:fm,Im**2:fm*(fm+2),Im**3:fm*(6*fm**2+6*fm+1),Im**4:24*fm**4+36*fm**3+14*fm**2+fm}
        return tempdict

    def thermAvgeval(self):
        #Im -> fm
        lstofthermalAvg = []
        for i in range(len(self.operatorlst)):
            lstofthermalAvg.append(self.thermAvghelper(self.operatorlst[i],self.qtnumberlst[i],self.BEfactorlst[i]))
        return lstofthermalAvg

    def BHrulehelper(self,Qm,Im,wm):
        tempdict = {D0*Qm:0,D0*Qm**2:(Im+sym.Rational(1,2))/wm,D0*Qm**3:0,D0*Qm**4:(6*Im*(Im+1)+3)/wm/wm*sym.Rational(1,4),
                    D1*Qm:sym.sqrt((Im+1)/wm*sym.Rational(1,2)),D1*Qm**2:0,D1*Qm**3:3*((Im+1)/wm*sym.Rational(1,2))**sym.Rational(3,2),D1*Qm**4:0,
                    D2*Qm:0,D2*Qm**2:sym.sqrt((Im+2)*(Im+1))/wm*sym.Rational(1,2),D2*Qm**3:0,D2*Qm**4:(Im+sym.Rational(3,2))*sym.sqrt((Im+2)*(Im+1))/wm/wm,
                    D3*Qm:0,D3*Qm**2:0,D3*Qm**3:sym.sqrt((Im+3)*(Im+2)*(Im+1))*(sym.Rational(1,2)/wm)**sym.Rational(3,2),D3*Qm**4:0,
                    D4*Qm:0,D4*Qm**2:0,D4*Qm**3:0,D4*Qm**4:sym.sqrt((Im+4)*(Im+3)*(Im+2)*(Im+1))/wm/wm*sym.Rational(1,4),
                    D1n*Qm:sym.sqrt(Im/wm*sym.Rational(1,2)),D1n*Qm**2:0,D1n*Qm**3:3*(Im/wm*sym.Rational(1,2))**sym.Rational(3,2),D1n*Qm**4:0,
                    D2n*Qm:0,D2n*Qm**2:sym.sqrt(Im*(Im-1))/wm*sym.Rational(1,2),D2n*Qm**3:0,D2n*Qm**4:(Im-sym.Rational(1,2))*sym.sqrt(Im*(Im-1))/wm/wm,
                    D3n*Qm:0,D3n*Qm**2:0,D3n*Qm**3:sym.sqrt(Im*(Im-1)*(Im-2))*(sym.Rational(1,2)/wm)**sym.Rational(3,2),D3n*Qm**4:0,
                    D4n*Qm:0,D4n*Qm**2:0,D4n*Qm**3:0,D4n*Qm**4:sym.sqrt(Im*(Im-1)*(Im-2)*(Im-3))/wm/wm*sym.Rational(1,4)}
        return tempdict

    def BHruleeval(self):
        #Dx*Qm**y - > Im
        lstofBHdict = []
        for i in range(len(self.operatorlst)):
            lstofBHdict.append(self.BHrulehelper(self.operatorlst[i],self.qtnumberlst[i],self.freqlst[i]))
        lstofBHdict.append({D0:1,D1:0,D2:0,D3:0,D4:0,D1n:0,D2n:0,D3n:0,D4n:0})
        return lstofBHdict


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
        return iter3rd,iter4th



test = ThermalAvg()
