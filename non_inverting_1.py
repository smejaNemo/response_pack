#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:53:16 2017

@author: kot
"""

import numpy as np
import sympy as sy
from sympy.solvers import solve
import matplotlib.pyplot as plt

#from sympy.functions import re


init_printing(use_unicode=True)

C = sy.Symbol('C',positive=True)
omega = sy.Symbol('omega',positive=True)
R1 = sy.Symbol('R1',real=True,positive=True)
R2 = sy.Symbol('R2',real=True,positive=True)
Z1 = sy.Symbol('Z1',imaginary=True)
Z2 = sy.Symbol('Z2',imaginary=True)
ZC = sy.Symbol('ZC',imaginary=True)
V = sy.Symbol('V',imaginary=True)
H = sy.Symbol('H',imaginary=True)

ZC = -sy.I * (1/(omega*C))
Z1 = R1
Z2 = ZC*R2/(ZC + R2)

V = 1 + (Z1/Z2)

H = sy.limit(V,omega,0) / V
#pprint(sy.limit(V,omega,0))
#pprint(sy.Abs(H))
H = H.subs(R1,1000)
H = H.subs(R2,1000)
H = H.subs(C,1E-6)
#pprint(sy.Abs(H.subs(omega,100000)))
fc = float(sy.solve(sy.Abs(H)-1/sy.sqrt(2),omega)[0] / (2*np.pi))

freq = np.linspace(1,fc + fc/4,int(fc)/2)
h_freq = np.zeros(len(freq))

for f in freq:
    h_freq[np.where(freq == f)] = sy.Abs(H.subs(omega,f*2*np.pi))



print(fc)
X_line_V = np.ones(len(h_freq)) * fc
Y_line_V = np.linspace(h_freq.min(),h_freq.max(),len(h_freq))

Y_line_H = np.ones(len(h_freq)) * sy.Abs(H.subs(omega,fc*2*np.pi))
X_line_H = freq
plt.plot(np.log10(freq),20*np.log10(h_freq))
plt.plot(np.log10(X_line_V),20*np.log10(Y_line_V))
plt.plot(np.log10(X_line_H),20*np.log10(Y_line_H))
plt.show()

#==============================================================================
# pprint(H)
# pprint()
#==============================================================================

#pprint(sy.Abs(V))
#pprint(bool(C > 0))





#pprint(relation)
#pprint(limit(relation,omega,oo))
#relation = limit(relation,R2,R1)
#pprint(solve(Abs(relation)-(1/sqrt(2)),omega))#,dict=True))




# example of calculations 
# set value for R1 = 100 Ohm 
#relation = relation.subs(R1,100)
# set value for R2 = 10 Ohm
#relation = relation.subs(R2,10)
# set cutoff frequency 200Hz
#relation = relation.subs(omega,200/(2*pi))
#C = solve(Abs(relation)-(1/sqrt(2)),C)[1]
#pprint(float(C)*1E6)



