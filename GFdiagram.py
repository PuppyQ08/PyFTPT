import sympy as sym
wi = sym.symbols('wi',positive=True,real=True)
wj = sym.symbols('wj',positive=True,real=True)
wk = sym.symbols('wk',positive=True,real=True)
wl = sym.symbols('wl',positive=True,real=True)
fi = sym.symbols('fi')
fj = sym.symbols('fj')
fk = sym.symbols('fk')
fl = sym.symbols('fl')

diagram2C = (fi*fj+fj*fk+fi*fk+fi+fk+1)/(wi+wj+wk)-(fi*fj+fj*fk-fi*fk+fj)/(wj-wi-wk)-(fi*fj-fk*fj-fi*fk-fk)/(wj+wi-wk)-(fk*fj-fi*fj-fi*fk-fi)/(wj-wi+wk)
sub1 = diagram2C.subs({fk:fj,wk:wj})
print(sub1)
