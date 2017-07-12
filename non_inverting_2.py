#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:53:16 2017

@author: kot
"""

import numpy as np
from sympy import I, symbols, init_printing, pprint, Abs, re, im, limit, lambdify,sqrt, pi, simplify
from sympy.solvers import solve
#from sympy.functions import re


init_printing(use_unicode=True)


R4, R3, C, omega = symbols('R4 R3 C omega', real=True)
Z3, Z4, ZC, V, relation = symbols('Z3 Z4 ZC V relation', imaginary=True)
ZC = -I*(1/(omega * C))
Z3 = R4
Z4 = R3 * ZC / (R3 + ZC)
V = 1 + Z4/Z3



relation = limit(V,omega,0) / V
#relation = limit(relation,R2,R1)

pprint(limit(V,omega,0))
relation = Abs(relation)

relation = relation.subs(R3,R4)
realRelation = re(relation)
imRelation = im(relation)
absrelation = simplify(sqrt(realRelation**2 + imRelation**2))
#pprint( simplify(absrelation - (2/sqrt(2)) )**2 )
pprint(solve( absrelation - (2/sqrt(2)),C))


#pprint(  )

# example of calculations 
# set value for R4 = 100 Ohm 
relation = relation.subs(R4,100)
# set value for R3 = 10 Ohm
relation = relation.subs(R3,100)
# set cutoff frequency 200Hz
relation = relation.subs(omega,200/(2*pi))

C = solve( relation-(2/sqrt(2)),C)[1] 
pprint(float(C)*1E6)

