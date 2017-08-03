#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:53:16 2017

@author: kot
"""

import numpy as np
from sympy import I, symbols, init_printing, pprint, Abs, re, im, limit, lambdify,sqrt, pi, Eq, simplify,rsolve, S
from sympy.solvers import solve
#from sympy.functions import re


init_printing(use_unicode=True)


R1, R2, C, omega = symbols('R1 R2 C omega', positive=True, real=True)

R1 = S('R1',positive=True,real=True)
R2 = S('R2',positive=True,real=True)
C = S('C',positive=True,real=True)
omega = S('omega',pos=True,real=True)


Z1, Z2, ZC, V, relation = symbols('Z1 Z2 ZC V relation', imaginary=True)
ZC = -I*(1/(omega * C))
Z1 = R1
Z2 = R2 * ZC / (R2 + ZC)
V = 1 + Z1/Z2



relation = Abs(simplify(limit(V,omega,0) / V))
#relation = limit(relation,R2,R1)
pprint(relation)
pprint(simplify(limit(relation,omega,0)))
pprint(solve(relation,omega))




#==============================================================================
# # example of calculations 
# # set value for R1 = 100 Ohm 
# relation = relation.subs(R1,100)
# # set value for R2 = 10 Ohm
# relation = relation.subs(R2,10)
# # set cutoff frequency 200Hz
# relation = relation.subs(omega,200/(2*pi))
# C = solve(Abs(relation)-(1/sqrt(2)),C)[1]
# pprint(float(C)*1E6)
#==============================================================================



