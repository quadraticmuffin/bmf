import scipy.integrate as si 
import matplotlib.pyplot as plt
import numpy as np
import math

# Parameters
gamma = 0.01
koff = 10
kon = 10
ptot = 100
omega = 0.005
t_max = 30000

pars = [gamma, koff, kon, ptot]

# Initial conditions
X = 0
C = 0

y = [X, C]

def k(t):
    return gamma * (1+math.sin(omega * t))

# system of ODEs
def isolated(t, X, gamma):
    return [k(t)-gamma*X]
def connected(t, y, gamma, koff, kon, ptot):
    return [
        k(t) - gamma*y[0] + koff*y[1] - kon*(ptot-y[1])*y[0],
        -koff*y[1] + kon*(ptot-y[1])*y[0]
    ]

# Jacobian
j_isolated = [[-gamma]]
j_connected = [[-gamma - kon*(ptot-y[1]), koff + kon*y[0]],
    [kon*(ptot-y[1]), -koff - kon*y[0]]]

# # Compute and plot trial path
# sol = si.solve_ivp(lambda t,y: connected(t,y,*pars), (0,t_max), [0]*len(y), 
#         method='Radau', vectorized=True, dense_output=True, jac = j_connected)
# t = np.linspace(0,t_max,1000)
# z = sol.sol(t)[0]
# sol = si.solve_ivp(lambda t,x: isolated(t,x,gamma), (0,t_max), [0], 
#         method='Radau', vectorized=True, dense_output=True, jac = j_isolated)
# t = np.linspace(0,t_max,1000)
# f = sol.sol(t)[0]
# plt.plot(t,z.T)
# plt.plot(t,f.T)
# plt.legend(['connected', 'isolated'])
# plt.show()

# Measure steady state amplitude for values of omega: [0.001, 0.002,..., 0.01]
num_trials = 10
omegas = [0.001 * i for i in range(1,num_trials+1)]
amps_connected = [None] * num_trials
amps_isolated = [None] * num_trials
for i in range(num_trials):
    omega = omegas[i]

    sol = si.solve_ivp(lambda t,y: connected(t,y,*pars), (0,t_max), [0]*len(y), 
        method='Radau', vectorized=True, dense_output=True, jac = j_connected)
    t = np.linspace(20000,t_max,1000) # Only take solution after t=20000 to ensure we measure the amplitude after steady state
    z = sol.sol(t)[0]
    f = sol.sol(t)[0]
    amps_connected[i] = max(z)-min(z)

    sol = si.solve_ivp(lambda t,x: isolated(t,x,gamma), (0,t_max), [0], 
        method='Radau', vectorized=True, dense_output=True, jac = j_isolated)
    t = np.linspace(20000,t_max,1000) # Only take solution after t=20000 to ensure we measure the amplitude after steady state
    z = sol.sol(t)[0]
    amps_isolated[i] = max(z)-min(z)


plt.plot(omegas, amps_connected, 'b-')
plt.plot(omegas, amps_isolated, 'r-')
plt.legend(['connected', 'isolated'])
plt.xlabel('omega')
plt.ylabel('Amplitude')
plt.title('gamma = ' + str(gamma))
plt.show()