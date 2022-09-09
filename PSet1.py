import matplotlib.pyplot as plt
import numpy as np

from createNeuralNet import create_neural_net
from forwardPass import forward_pass
from backprop import backprop
from pongPhysics import pongPhysics
from plotDiagnostics import plotDiagnostics

np.random.seed(1)  # Set random seed for repeatability
save = True

# Hyperparameters found by Arnaud klipfel <akipfel3@gatech.edu>
alphaPG = 2e-1  # Learning rate for policy gradient
alphaQ = 1e-3  # Learning rate for Q network
numNodesPerLayerP = [8]  # Vector of the number of hidden nodes per layer, indexed from input -> output
numNodesPerLayerQ = [8]
I = 40000  # Number of episodes
T = 100  # Max time steps for the game (i.e., episode horizon)
Q_mini_dataset_size = 100  # Size of the mini_dataset used to update the Q nn
gamma = 0.95   # Discount factor
difficultyLevel = 1  # Difficulty level (see below)
numInputDims = 6  # State space

data_nn = (alphaPG, alphaQ, numNodesPerLayerP, numNodesPerLayerQ, Q_mini_dataset_size)
# NN creation
numOutputDims = 2  # Dimensionality of the outputs space (i.e., for ``move up'' vs. ``move down'')
# Create the nn for the policy. The output activation for the policy will be softmax.
nn = create_neural_net(numNodesPerLayerP, numInputDims, numOutputDims)
# Creates the nn for the Q-function. The output activation for the Q network will be linear.
nnQ = create_neural_net(numNodesPerLayerQ, numInputDims, numOutputDims)

# Loss plot for the Q network
L_Q = np.zeros((I, 1))
# Loss plot for the policy network
L = np.zeros((I, 1))
winLossRecord = np.zeros(I)
# Initialize visualization
fig, axes = plt.subplots(1, 3)
fig.suptitle(r"$\alpha$={}, $\alpha_Q$={}, Architecture P {} and Q {}, $|D'|$={}".format(data_nn[0], data_nn[1], data_nn[2],  data_nn[3], data_nn[4]))
line1, = axes[0].plot([], '-b', linewidth=3)
line2, = axes[0].plot([], '-r', linewidth=3)
line3, = axes[0].plot([], '.k', markersize=12)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
fig.canvas.draw()
axbackground = fig.canvas.copy_from_bbox(axes[0].bbox)
fig.canvas.flush_events()

# Replay buffer for the Q nn.
replay_buffer = []

# Main loop
for i in range(I):
    if i % 1000 == 0:
        print("Iteration {}".format(i))
    #################################################################
    #########                 COLLECT DATA                  #########
    #################################################################

    # Initialize the first state of the game. The state consists of the
    # following features.
    # 1) y-position of the player's paddle
    # 2) y-position of the opponent's paddle
    # 3) x-position of the ball
    # 4) y-position of the ball
    # 5) x-velocity of the ball
    # 6) y-velocity of the ball

    s = [0.5, 0.5, 0.5, 0.5, -1, 0]

    # Randomly initialize the y-velocity of the ball and normalize so that the speed of the ball is 1.

    vel_init = np.array([-1, np.random.uniform() - 0.5])
    vel_init = vel_init / np.sqrt(np.dot(vel_init, vel_init))
    s[4:] = vel_init

    # INITIALIZE A LIST TO STORE TRAJECTORY <S,A,R,S'> FOR EACH TIME STEP t in range(T)
    tau = []
    T_terminal = T
    for t in range(T):

        # Here, we draw actions from a policy (i.e., a probability mass function)
        probs = forward_pass(nn, np.array(s).reshape(6, 1), final_softmax=True)  # of size 2x1
        # Sample from policy.
        a_1 = np.random.choice(2, p=probs.ravel())  # SAMPLE FROM A PROBABILITY DISTRIBUTION DEFINED BY PROBS

        # For now player 2 is not going to take any actions
        a_2 = -1

        # Apply transition function and get our reward. Note: the fourth
        # input is a Boolean to determine whether to plot the game being played.
        PlottingBool = False
        s_prime, r = pongPhysics(s, a_1, a_2, PlottingBool, axes, fig, line1, line2, line3, axbackground)

        # Store the <s,a,r,s'> transition into a trajectory to train Q.
        # stores the reward for the first player.
        data = (np.array(s).reshape(6, 1),
                a_1,
                r[0],
                np.array(s_prime).reshape(6, 1))
        tau.append(data)
        replay_buffer.append(data)
        # Determine if the new state is a terminal state. If so, then quit
        # the game. If not, step forward into the next state.
        if r[0] == -1 or r[1] == -1:
            # The next state is a terminal state. Therefore, we should
            # record the outcome of the game in winLossRecord for game i.
            winLossRecord[i] = float(r[0] == 1)
            T_terminal = t+1
            break

        else:
            # Simply step to the next state
            s = s_prime

    ######################################################################
    ##############                 UPDATE POLICY            ##############
    ######################################################################

    for t in range(T_terminal):
        # get the state, s_t
        s = tau[t][0]
        # get the action, a_t
        a = tau[t][1]
        # get the probability of taking action a_t in state s_t, i.e. \pi(a_t | s_t)
        probs = forward_pass(nn, s, final_softmax=True)
        Pr_a_Given_s = probs[a]
        # Q(s,a) estimate
        A_t = forward_pass(nnQ, s)[a]
        # Compute the gradient for the neural network
        g = backprop(nn, s, a, 'Softmax')  # third entry should be the action
        # Update the neural network parameters (normalizing for batchSize).
        for j in range(len(nn)):
            # we divide by the proba since the log is in the gradient.
            nn[j][0] = nn[j][0] + (alphaPG *g[j][0]*A_t)/ (Pr_a_Given_s*T_terminal) # INSERT CODE HERE (1) # Update the neural network weights
            nn[j][1] = nn[j][1] + (alphaPG * g[j][1]*A_t)/(Pr_a_Given_s*T_terminal) # INSERT CODE HERE (2)  # Update the neural network biases
            # Store into L[i] a measure of how much the network is changing.
            
            L[i] += np.sum(np.abs((alphaPG * g[j][0]*A_t)/ (Pr_a_Given_s*T_terminal))) + np.sum(np.abs((alphaPG * g[j][1]*A_t)/ (Pr_a_Given_s*T_terminal)))  # INSERT CODE HERE (You just need to copy (1) and (2) from above here)
    # Normalize by the number of time steps.
    L[i] /= T_terminal

    ######################################################################
    ##############              UPDATE Q-function           ##############
    ######################################################################
    # Retrieves a subset of the replay buffer.
    nb_data = len(replay_buffer)
    if nb_data <= Q_mini_dataset_size:
        # if more data are asked to train the Q nn than available.
        ind = np.random.choice(nb_data, size=nb_data, replace=False)  # take all the data available.
    else:
        ind = np.random.choice(nb_data, size=Q_mini_dataset_size, replace=False)
    Dprime = [replay_buffer[k] for k in ind]  # extract data from dataset.
    # Update of the NN.
    card_Dprime = len(Dprime)
    for k in range(card_Dprime):
        # Sample a piece of data from Dprime.
        d = np.random.choice(card_Dprime)
        # Retrieves data.
        # get the state, s
        s = Dprime[d][0]
        # get the current action, a
        a = Dprime[d][1]
        # Reward, r
        r = Dprime[d][2]
        # Future state, s'
        s_prime = Dprime[d][3]
        # Computes the Y term in the Bellman residual (Y-Y_hat).
        # Gets Q(s,a), that we will change to the Y term in the MSE later.
        Y = forward_pass(nnQ, s)
        Y_hat = np.copy(Y)
        # Gets Q(s',a').
        Q_prime = forward_pass(nnQ, s_prime)
        # Replace the term for the action taken
        if r in [-1, 1]:
            Y[a] = r
        else:
            Y[a] = r + gamma*np.max(Q_prime) # INSERT CODE HERE
            # Backpropagation to get the gradient of the loss.
        gQ = backprop(nnQ, s, Y, 'MSE')
        # Updates the parameters of the NN.
        for j in range(len(nnQ)):
            nnQ[j][0] -= alphaQ * gQ[j][0] / card_Dprime  # Update the neural network weights
            nnQ[j][1] -= alphaQ * gQ[j][1] / card_Dprime  # Update the neural network biases
        L_Q[i] += 0.5 * np.mean((Y - Y_hat) ** 2)
    L_Q[i] /= card_Dprime
    # Plot loss per iter and win-loss record per iter.
    #
    if i % 1000 == 0:
        plotDiagnostics(L, winLossRecord, i, axes, fig, data_nn)
    # Learning rate decay for the policy nn.
    if i == 15000:
        alphaPG /= 2
    if i == 21000:
        alphaPG /= 2
    if i == 27000:
        alphaPG /= 2
    if i == 35000:
        alphaPG /= 2
    fig.canvas.flush_events()
