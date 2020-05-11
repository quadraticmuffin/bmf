import numpy as np 
import numpy.random as random
import matplotlib.pyplot as plt
import random as rand
# Initial Conditions
X1, X1d, D1, C1, Y1, G1 = 0, 0, 1, 0, 0, 0
X2, X2d, D2, C2, Y2, G2 = 0, 0, 1, 0, 0, 0
q0 = np.matrix([[X1, X1d, D1, C1, X2, X2d, D2, C2, Y1, G1, Y2, G2]]).T

# Assume reactions are symmetric, 
# i.e. rate constants are similar between X1 and X2.
# fast reactions
alpha_d = 50 # X1 + X1 -> X1d, X2 + X2 -> X2d
delta_d = 25 # X1d -> X1 + X1, X2d -> X2 + X2
alpha_c = 50 # X1d + D2 -> C2, X2d + D1 -> C1
delta_c = 25 # C2 -> X1d + D2, C1 -> X2d + D1
alpha_g = 50 # X1 + Y1 -> G1, X2 + Y2 -> G2
delta_g = 25 # G1 -> X1 + Y1, G2 -> X2 + Y2

# slow reactions
beta_x = 3   # D1 -> D1 + X1, D2 -> D2 + X2
beta_y = 10     # 0 -> Y1, 0 -> Y2
gamma = 1      # X1 -> 0, X2 -> 0, Y1 -> 0, Y2 -> 0
omega = 1      # G1 -> 0, G2 -> 0
# Volume
V = 1
# Number of Reactions to compute
r_max = 1000000
sparsity = 100
num_records = int(r_max/sparsity)
temp_dependence = 3
# Propensity functions
#       0   1   2   3   4   5    6   7   8   9   10  11
# q = [X1, X1d, D1, C1, X2, X2d, D2, C2, Y1, G1, Y2, G2]
r = [
    lambda q: alpha_d * (q[0]*(q[0]-1)/2) / V,
    lambda q: alpha_d * (q[4]*(q[4]-1)/2) / V,
    lambda q: delta_d * q[1],
    lambda q: delta_d * q[5],
    lambda q: alpha_c * q[1] * q[6] / V,
    lambda q: alpha_c * q[5] * q[2] / V,
    lambda q: delta_c * q[7],
    lambda q: delta_c * q[3],
    lambda q: beta_x * q[2],
    lambda q: beta_x * q[6],
    lambda q: gamma * q[0],
    lambda q: gamma * q[4],

    lambda q: alpha_g * q[0] * q[8] / V,
    lambda q: alpha_g * q[4] * q[10] / V,
    lambda q: delta_g * q[9],
    lambda q: delta_g * q[11],
    lambda q: omega * q[9],
    lambda q: omega * q[11],
    lambda q: beta_y,
    lambda q: beta_y,
    lambda q: gamma * q[8],
    lambda q: gamma * q[10],

]
#Stoichiometric matrix
s = np.matrix([
    [-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -2, 1, 0, 0, 0, 0, 0, 0],
    [2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, -1, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
    [0, 0, -1, 1, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
    [0, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],

    [-1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0]
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
        reaction_times = [None] * np.shape(q)[0]
        for i in range(np.shape(q)[0]):
            if r[i](q) != 0:
                reaction_times[i] = random.exponential(1/r[i](q))
        dt = min(i for i in reaction_times if i is not None)
        i = reaction_times.index(dt)
        # Advance t and q according to the chosen reaction
        t += dt
        q += s[:, i:i+1]
        global delta_c
        if j % sparsity == 0:
            record_idx = int(j/sparsity)
            times[record_idx] = np.asscalar(t)
            q_over_time[:, record_idx : record_idx+1] = q
            print (j/r_max)
        # Cumulatively record t and q
        
        # Temperature increase halfway through
        if j == int(r_max/2):
            global delta_c
            global delta_g
            global temp_dependence
            delta_c *= temp_dependence
            delta_g *= temp_dependence
            print ("Temperature increased: t = " + str(np.asscalar(t)))
        
    return times, q_over_time

# Return and plot reaction
# def multiplot(): 
    fig, axs = plt.subplots(2,1)
    delta_c = 25
    delta_g = 25
    # Probability Distributions
    times, qs = ssa(q0, r, s, r_max, sparsity)
    X1_over_time = qs[0].tolist()
    X2_over_time = qs[4].tolist()
    axs[0].plot(times, X1_over_time, 'r-', label="X1")
    axs[0].plot(times, X2_over_time, 'b-', label="X2")
    axs[0].set_title("Kc, Kg = " + str(delta_c/alpha_c) + ", " + str(delta_g/alpha_g))

    delta_c = 75
    delta_g = 75
    # Probability Distributions
    times, qs = ssa(q0, r, s, r_max, sparsity)
    X1_over_time = qs[0].tolist()
    X2_over_time = qs[4].tolist()
    axs[1].plot(times, X1_over_time, 'r-', label="X1")
    axs[1].plot(times, X2_over_time, 'b-', label="X2")
    axs[1].set_title("Kc, Kg = " + str(delta_c/alpha_c) + ", " + str(delta_g/alpha_g))

    # delta_c = 50
    # delta_g = 50
    # # Probability Distributions
    # times, qs = ssa(q0, r, s, r_max, sparsity)
    # X1_over_time = qs[0].tolist()
    # X2_over_time = qs[4].tolist()
    # axs[0,1].plot(times, X1_over_time, 'r-', label="X1")
    # axs[0,1].plot(times, X2_over_time, 'b-', label="X2")
    # axs[0,1].set_title("Kc, Kg = " + str(delta_c/alpha_c) + ", " + str(delta_g/alpha_g))

    # delta_c = 100
    # delta_g = 100
    # # Probability Distributions
    # times, qs = ssa(q0, r, s, r_max, sparsity)
    # X1_over_time = qs[0].tolist()
    # X2_over_time = qs[4].tolist()
    # axs[1,1].plot(times, X1_over_time, 'r-', label="X1")
    # axs[1,1].plot(times, X2_over_time, 'b-', label="X2")
    # axs[1,1].set_title("Kc, Kg = " + str(delta_c/alpha_c) + ", " + str(delta_g/alpha_g))

# END MULTIPLOT 

times, qs = ssa(q0, r, s, r_max, sparsity)
X1_over_time = qs[0].tolist()
X2_over_time = qs[4].tolist()

# Quantify "bimodality"
separation_before = 0
for i in range(int(num_records/2)):
    x1 = X1_over_time[i]
    x2 = X2_over_time[i]
    separation_before += (x1-x2)**2 / r_max

separation_after = 0
for i in range(int(num_records/2), num_records):
    x1 = X1_over_time[i]
    x2 = X2_over_time[i]
    separation_after += (x1-x2)**2 / r_max

print ("before: " + str(separation_before))
print ("after: " + str(separation_after))
# print (X1_over_time)
plt.plot(times, X1_over_time, 'r-', label="X1")
plt.plot(times, X2_over_time, 'b-', label="X2")
plt.legend()
# plt.title('beta_x/gamma = ' + str(beta_x/gamma))
plt.show()

1043