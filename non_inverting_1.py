#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:53:16 2017

@author: kot
"""

import numpy as np

#from sympy import I, symbols, init_printing, pprint, Abs, re, im, limit, lambdify,sqrt, pi, Eq, simplify,rsolve, S

import sympy as sy

from sympy.solvers import solve
import matplotlib.pyplot as plt

#from sympy.functions import re


sy.init_printing(use_unicode=True)

C = sy.Symbol('C',positive=True)
omega = sy.Symbol('omega',positive=True)
R1 = sy.Symbol('R1',real=True,positive=True)
R2 = sy.Symbol('R2',real=True,positive=True)
Z1 = sy.Symbol('Z1',imaginary=True)
Z2 = sy.Symbol('Z2',imaginary=True)
ZC = sy.Symbol('ZC',imaginary=True)
V = sy.Symbol('V',imaginary=True)
H = sy.Symbol('H',imaginary=True)


R1, R2, C, omega = sy.symbols('R1 R2 C omega', positive=True, real=True)

R1 = sy.Symbol('R1',positive=True,real=True)
R2 = sy.Symbol('R2',positive=True,real=True)
C = sy.Symbol('C',positive=True,real=True)
omega = sy.Symbol('omega',pos=True,real=True)

 
Z1, Z2, ZC, V, relation = sy.symbols('Z1 Z2 ZC V relation', imaginary=True)
ZC = -sy.I*(1/(omega * C))
 
ZC = -sy.I * (1/(omega*C))
Z1 = R1*ZC/(R1 + ZC)
Z2 = R2
V = 1 + (Z1/Z2)

H = sy.limit(V,omega,0) / V
sy.pprint(sy.limit(V,omega,0))

# Set parameter
H = H.subs(R1,1000)
H = H.subs(R2,1000)
H = H.subs(C,100E-9)
solutuins = sy.solve(sy.Abs(H)-(2/sy.sqrt(2)),omega)
pprint(solutuins)
fc = float(solutuins[0] / (2*np.pi))
# 
freq = np.linspace(1,fc + fc/4,int(round(fc,0)/2))
h_freq = np.zeros(len(freq))
# 
for f in freq:
    h_freq[np.where(freq == f)] = sy.Abs(H.subs(omega,f*2*np.pi))
# 
# 
print('Cut-Off frequency: ' + str(round(fc,3)) + ' Hz')
X_line_V = np.ones(len(h_freq)) * fc
Y_line_V = np.linspace(h_freq.min(),h_freq.max(),len(h_freq))
# 
Y_line_H = np.ones(len(h_freq)) * float(sy.Abs(H.subs(omega,fc*2*np.pi)))
X_line_H = freq
plt.plot(freq,20*np.log10(h_freq))
plt.plot(X_line_V,20*np.log10(Y_line_V))
plt.plot(X_line_H,20*np.log10(Y_line_H))
plt.xlabel('Frequncy f, [Hz]',size=13)
plt.ylabel('Normalized Response-Function, [dB]',size=13)
plt.annotate(str(round(fc,2)),(int(fc+5),0-2.7))
plt.show()



