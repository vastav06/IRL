import numpy as np


def plotDiagnostics(L, winLossRecord, i, axes, fig, data=None):
    # Author: Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu
    # Date: 26 JUN 2020

    # This function takes as input a vector, L, containing the loss at each
    # iteration of training, the win-loss record, winLossRecord, at each
    # iteration of training, and the number of iterations, i.

    # Plot the loss after each iteration vs. the iteration number.
    y = L[:i]
    y_new = np.zeros((len(y), 1))
    for j in range(len(y)):
        y_new[j] = np.mean(y[max(j - 25, 0): min(j + 25, len(y))])
    axes[1].plot(range(0, i), y_new, '-.g')  # Can only use log scale for positive values
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Change in Params')
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()

    winLossRecord_new = np.zeros((len(y), 1))
    for j in range(i):
        winLossRecord_new[j] = np.mean(winLossRecord[max(j - 25, 0):min(j + 25, i)])
        
    axes[2].plot(range(0, len(winLossRecord_new)), winLossRecord_new, '-r')
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel('Games Played')
    axes[2].set_ylabel('Win/Loss')
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    fig.savefig('PQ_{}_{}_{}_{}_Dprime_{}.png'.format(data[0], data[1], data[2], data[3], data[4]))
