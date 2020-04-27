import scipy.integrate as si 
import matplotlib.pyplot as plt
import numpy as np

# Parameters
deltaA, deltaB = 1, 1
alphaA, alphaB = 250, 30
alphaA0, alphaB0 = 0.04, 0.004
gammaA, gammaB = 1, 0.5 # gammaB = 0.5 for functional clock; 1.5 for non-functional clock
kappaA, kappaB = 1, 1
KA, KB = 1, 1
n, m = 2, 4
kon, koff = 10, 10
ptot = 0
t_max = 50

pars = [deltaA, deltaB, alphaA, alphaB, alphaA0, alphaB0, gammaA, gammaB, kappaA, kappaB, KA, KB, n, m, kon, koff, ptot]

# Initial conditions
A = 0
B = 0
C = 0

y = [A,B,C]

# Functions as defined in Section 5.5 of textbook
def F1(A,B):
    return (alphaA*((A/KA)**n) + alphaA0)/(1+((A/KA)**n)+((B/KB)**m))
def F2(A):
    return (alphaB*((A/KA)**n) + alphaB0)/(1+((A/KA)**n))
def f1(A,B):
    return kappaA/deltaA*F1(A,B)
def f2(A):
    return kappaB/deltaB*F2(A)

# system of ODEs
def odeA(t, y, deltaA, deltaB, alphaA, alphaB, alphaA0, alphaB0, gammaA, gammaB, kappaA, kappaB, KA, KB, n, m, kon, koff, ptot):
    return [-gammaA*y[0] + kappaA/deltaA*(alphaA*((y[0]/KA)**n)+alphaA0)/(1+((y[0]/KA)**n)+((y[1]/KB)**m)) 
                - n*kon*(y[0]**n)*(ptot-y[2]) + n*koff*y[2],                        # -gammaA*y[0] + f1(A,B) - n*kon(A^n)*p + n*koff*C
            -gammaB*y[1] + kappaB/deltaB*(alphaB*((y[0]/KA)**n)+alphaB0)/(1+((y[0]/KA)**n)), # -gammaB*B + f2(A)
            kon*(y[0]**n)*(ptot-y[2]) - koff*y[2]                                 # kon(A^n)*p - koff*C
            ]  
def odeB(t, y, deltaA, deltaB, alphaA, alphaB, alphaA0, alphaB0, gammaA, gammaB, kappaA, kappaB, KA, KB, n, m, kon, koff, ptot):
    return [-gammaA*y[0] + kappaA/deltaA*(alphaA*((y[0]/KA)**n)+alphaA0)/(1+((y[0]/KA)**n)+((y[1]/KB)**m)), # -gammaA*y[0] + f1(A,B) 
            -gammaB*y[1] + kappaB/deltaB*(alphaB*((y[0]/KA)**n)+alphaB0)/(1+((y[0]/KA)**n))
                - m*kon*(y[1]**m)*(ptot-y[2]) + m*koff*y[2],                      # -gammaB*B + f2(A)- m*kon(B^m)*p + m*koff*C
            kon*(y[1]**m)*(ptot-y[2]) - koff*y[2]                                 # kon(B^m)*p - koff*C
            ]

# Compute and plot trial path
sol = si.solve_ivp(lambda t,y: odeA(t,y,*pars), (0,t_max), [0]*len(y), 
        method='RK45', dense_output=True)
t = np.linspace(0,t_max,500)
a = sol.sol(t)[0]
b = sol.sol(t)[1]
plt.plot(t,a.T)
plt.plot(t,b.T)
plt.legend(['A','B'])
plt.title("Functional Clock, p_tot = " + str(ptot))
plt.show()