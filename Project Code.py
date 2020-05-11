import numpy as np 
import numpy.random as random
import matplotlib.pyplot as plt
import math as m
import random as rand
# Initial Conditions
X1, X1d, D1, C1 = 0, 0, 1, 0
X2, X2d, D2, C2 = 0, 0, 1, 0
q0 = np.matrix([[X1, X1d, D1, C1, X2, X2d, D2, C2]]).T

# Assume reactions are symmetric, 
# i.e. rate constants are similar between X1 and X2.
# fast reactions
alpha = 50 # X1 + X1 -> X1d, X2 + X2 -> X2d
delta = 25 # X1d -> X1 + X1, X2d -> X2 + X2
kappa = 50 # X1d + D2 -> C2, X2d + D1 -> C1
theta = 25 # C2 -> X1d + D2, C1 -> X2d + D1
# slow reactions
beta = 3   # D1 -> D1 + X1, D2 -> D2 + X2
gamma = 1  # X1 -> 0, X2 -> 0
# Volume
Omega = 1
# Number of Reactions to compute
r_max = 500000
sparsity = 500
# Propensity functions
# q = [X1, X1d, D1, C1, X2, X2d, D2, C2]
r = [
    lambda q: alpha * (q[0]*(q[0]-1)/2) / Omega,
    lambda q: alpha * (q[4]*(q[4]-1)/2) / Omega,
    lambda q: delta * q[1],
    lambda q: delta * q[5],
    lambda q: kappa * q[1] * q[6] / Omega,
    lambda q: kappa * q[5] * q[2] / Omega,
    lambda q: theta * q[7],
    lambda q: theta * q[3],
    lambda q: beta * q[2],
    lambda q: beta * q[6],
    lambda q: gamma * q[0],
    lambda q: gamma * q[4]
]
#Stoichiometric matrix
s = np.matrix([
    [-2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -2, 1, 0, 0],
    [2, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, -1, 0, 0],
    [0, -1, 0, 0, 0, 0, -1, 1],
    [0, 0, -1, 1, 0, -1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, -1],
    [0, 0, 1, -1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0]
    ]).T

# Gillespie's direct method (SSA algorithm)
def ssa(q0, r, s, r_max, sparsity):
    num_records = r_max/sparsity
    q = q0
    q_over_time = np.zeros((np.shape(q)[0], int(r_max/sparsity) + 1))
    q_over_time[:, 0:1] = q0
    t = 0
    times = [0] * (int(r_max/sparsity) + 1)
    for j in range(1, r_max+1):
        # Pick the reaction that happens first
        reaction_times = [None] * 12
        for i in range(12):
            if r[i](q) != 0:
                reaction_times[i] = random.exponential(1/r[i](q))
        dt = min(i for i in reaction_times if i is not None)
        i = reaction_times.index(dt)
        # prop_sum = sum(i(q) for i in r)
        # dt = random.exponential(1/prop_sum)
        # props = [i(q) for i in r]
        # i = rand.choices(
        #     population=list(range(12)),
        #     weights=[p/prop_sum for p in props],
        #     k=1
        # )[0]
        # Advance t and q according to the chosen reaction
        t += dt
        q += s[:, i:i+1]
        if j % sparsity == 0:
            record_idx = int(j/sparsity)
            times[record_idx] = np.asscalar(t)
            q_over_time[:, record_idx : record_idx+1] = q
            print (j/r_max)
        # Cumulatively record t and q
        
        # Temperature increase halfway through
        # if j == int(r_max/2):
        #     theta *= 1.5
        #     print ("Temperature increased")
        
    return times, q_over_time

# Return and plot reaction

# X1_over_time, X2_over_time = [], []
# for i in range(10):
#     times_i, qs_i = ssa(q0, r, s, r_max, sparsity)
#     X1_over_time.extend(qs_i[0].tolist())
#     X2_over_time.extend(qs_i[4].tolist())

def multiplot(): 
    fig, axs = plt.subplots(2, sharex=True)
    beta = 2
    # Probability Distributions
    times, qs = ssa(q0, r, s, r_max, sparsity)
    X1_over_time = qs[0].tolist()
    X2_over_time = qs[4].tolist()
    # print (X1_over_time)
    axs[0].plot(times, X1_over_time, 'r-', label="X1")
    axs[0].plot(times, X2_over_time, 'b-', label="X2")
    axs[0].set_title('beta/gamma = ' + str(beta/gamma))

    beta = 3
    # Probability Distributions
    times, qs = ssa(q0, r, s, r_max, sparsity)
    X1_over_time = qs[0].tolist()
    X2_over_time = qs[4].tolist()
    # print (X1_over_time)
    axs[1].plot(times, X1_over_time, 'r-', label="X1")
    axs[1].plot(times, X2_over_time, 'b-', label="X2")
    axs[1].set_title('beta/gamma = ' + str(beta/gamma))
# plt.hist2d(X1_over_time, X2_over_time, bins=[max(X1_over_time), max(X2_over_time)])
# plt.hist(X2_over_time, label = "X2")

times, qs = ssa(q0, r, s, r_max, sparsity)
X1_over_time = qs[0].tolist()
X2_over_time = qs[4].tolist()
# print (X1_over_time)
plt.plot(times, X1_over_time, 'r-', label="X1")
plt.plot(times, X2_over_time, 'b-', label="X2")
plt.legend()
plt.title('beta/gamma = ' + str(beta/gamma))
plt.show()