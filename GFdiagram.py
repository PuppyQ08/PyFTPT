import sympy as sym
wi = sym.symbols('wi',positive=True,real=True)
wj = sym.symbols('wj',positive=True,real=True)
wk = sym.symbols('wk',positive=True,real=True)
wl = sym.symbols('wl',positive=True,real=True)
fi = sym.symbols('fi')
fj = sym.symbols('fj')
fk = sym.symbols('fk')
fl = sym.symbols('fl')
def pp(ipt):
    #print(sym.expand(sym.simplify(ipt)))
    print(ipt)

diagram2C = (fi*fj+fj*fk+fi*fk+fi+fj+fk+1)/(wi+wj+wk)-(fi*fj+fj*fk-fi*fk+fj)/(wj-wi-wk)-(fi*fj-fk*fj-fi*fk-fk)/(wj+wi-wk)-(fk*fj-fi*fj-fi*fk-fi)/(wj-wi+wk)
sub1 = diagram2C.subs({fk:fj,wk:wj})
#sub1 = diagram2C.subs({fk:fj,wk:wj})
#sub1 = diagram2C.subs({fk:fi,wk:wi,fj:fi,wj:wi})
D1 = (fj*fk*(fj+fl+1)+fi*(fl+1)*(fj+1)-fk*fj*fl)/(-wi+wj+wk+wl)
D2 = (fi*fl*(fk+fj+1)-fj*fk*(fi+fl+1))/(-wi+wj+wk-wl)
D3 = (fl*fk*(fj+fi+1)+fl*(fi+1)*(fj+1)-fi*fj*fk)/(wi+wj+wk-wl)
D4 = ((fk+1)*(fi+1)*(fl+1)*(fj+1)-fi*fj*fk*fl)/(wi+wj+wk+wl)
D5 = (fk*fl*(fj+fi+1)-fi*fj*(fl+fj+1))/(wi+wj-wk-wl)
D6 = (fk*fj*(fi+fl+1)+fj*(fi+1)*(fl+1)-fi*fk*fl)/(wi-wj+wk+wl)
D7 = (fi*fk*(fj+fl+1)-fj*fl*(fi+fk+1))/(-wi+wj-wk+wl)
D8 = (fk*(fi+1)*(fj+1)*(fl+1)-fi*fj*fl*(fk+1))/(wi+wj-wk+wl)
diagram2D = -sym.Rational(1,24)*(D1+D2+D3+D4+D5+D6+D7+D8)
diagram2B_1 = -sym.Rational(1,8)*(2*fk+1)*(2*fl+1)*(fi+1/2)/wi
#______________1.01 _1.06
#sub1 = diagram2D.subs({fk:fi,wk:wi,fj:fi,wj:wi,fl:fi,wl:wi})
#sub2 = sym.simplify(sub1)
#sub3 = sub2 + sym.Rational(1,16)*(2*fi+1)**3/wi
#pp(sub3)
#______________1.04 2.05 2.13___
#PT1 = (2*fi**2*fj+fi**2+2*fi*fj**2+4*fi*fj+2*fi+fj**2+2*fj+1)/32/(wi+wj)
#PT2 = (fi**2+2*fi*fj*(fi-fj)-fj**2)/32/(wi-wj)
#PT3 = -(16*fi*fj**2+16*fi*fj+2*fi+8*fj**2+8*fj+1)/64/wi
#sub1 = diagram2D.subs({fl:fk,wl:wk})#,fk:fi,wk:wi})
#pp(sub1)
#print(sub1)
#sub2 = diagram2B_1.subs({fk:fj,fl:fk})
#pp(PT1+PT2+PT3)
#pp(sub1+sub2)
#________________________2.07 2.08
#PT1 = (2*fi*fk+2*fi*fl+fi-2*fj*fk+2*fj*fl-fj+4*fk*fl*(fi-fj))/8/(wi-wj)
#PT2 = (2*fi*fk+2*fi*fl+fi+2*fj*fk+2*fj*fl+fj+4*fk*fl*(fi+fj+1)+2*fk+2*fl+1)/8/(wi+wj)
#pp(sym.expand(sym.simplify(PT1+PT2),numer =True))
#test = (2*fk+1)*(2*fl+1)*(wi*(2*fj+1)-wj*(2*fi+1))
#pp(sym.expand(test))
#_______________________2.03 2.04
PT1 = -(2*fi*fj+2*fi*fl+fi+2*fj**2+4*fj*fl*(fi+fj)+6*fj*fl+3*fj+2*fl+1)/8/(wi+wj)
PT2 = (2*fi*fj+2*fi*fl+fi-2*fj**2+4*fj*fl*(fi-fj)-2*fj*fl-fj)/8/(wi-wj)
diag2B = -(2*fk+1)*(2*fl+1)*(wi*(2*fj+1)-wj*(2*fi+1))
#diag2B = diag2B.subs({fk:fj})
#pp(sym.expand(sym.simplify(PT1+PT2),numer =True))
#pp(sym.expand(diag2B))
#________________________2.05 2.06
#PT1 = PT1.subs({fl:fi})
#PT2 = PT2.subs({fl:fi})
#diag2B = diag2B.subs({fl:fj,fk:fi})
#pp(sym.expand(sym.simplify(PT1+PT2),numer =True))
#pp(sym.expand(diag2B))
#____________1.09
diag2B = -(2*fk+1)*(2*fl+1)*(fi+sym.Rational(1,2))/wi/8
PT = - (8*fi*fk*fl+4*fi*fk+4*fi*fl+2*fi+4*fk*fl+2*fk+2*fl+1)/wi/16
pp(sym.expand(diag2B))
pp(sym.expand(PT))
