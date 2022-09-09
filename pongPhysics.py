import operator

import numpy as np


def execute_pongPhysics(x, a_1, a_2):
	dt = 0.1  # Time step of physics
	paddleWidth = 0.1  # Width of each player's paddle
	actionDistance = 0.1  # How far the paddle moves when commanded

	# Get ball state
	ball_pos = x[2:4]
	ball_vel = x[4:]

	# update ball state
	ball_pos_new = list(map(operator.add, ball_pos, [y * dt for y in ball_vel]))
	ball_vel_new = ball_vel
	# Correct ball height (bouncing) and velocity
	if ball_pos_new[1] < 0:
		ball_pos_new[1] = abs(ball_pos_new[1])
		ball_vel_new[1] = -ball_vel_new[1]
	elif ball_pos_new[1] > 1:
		ball_pos_new[1] = 2 - ball_pos_new[1]
		ball_vel_new[1] = -ball_vel_new[1]

	if a_1 == 0:
		vel_1 = -1
	elif a_1 == 1:
		vel_1 = 1

	player_1_new = x[0] + vel_1 * actionDistance

	vel_2 = 0  # For now, player 2 is not moving.
	player_2_new = x[1] + vel_2 * actionDistance

	# Clip player locations (y-direction) to be between [0,1]
	if player_1_new < 0:
		player_1_new = 0
	elif player_1_new > 1:
		player_1_new = 1

	if player_2_new < 0:
		player_2_new = 0
	elif player_2_new > 1:
		player_2_new = 1

	# Determine if player hit ball. If so, reverse direction. If not, game
	# over :(

	r_new = (0, 0)
	if ball_pos_new[0] <= 0:
		# Player 1 needs to have hit the ball or Player 1 loses point.
		if abs(player_1_new - ball_pos_new[1]) <= paddleWidth and ball_pos[0] > 0:
			# Reverse direction!
			ball_pos_new[0] = -ball_pos_new[0]
			ball_vel_new[0] = -ball_vel_new[0]

			# Add component of velocity (may not obey physics...)
			ball_vel_new[1] = ball_vel_new[1] + a_1 * actionDistance
		else:
			# Point Over
			r_new = (-1, 1)

	elif ball_pos_new[0] >= 1:
		# Player 2 needs to have hit the ball or Player 2 loses point.
		if abs(player_2_new - ball_pos_new[1]) <= paddleWidth and ball_pos[0] < 1:
			# reverse direction!
			ball_pos_new[0] = 2 - ball_pos_new[0]
			ball_vel_new[0] = -ball_vel_new[0]

			# Add component of velocity (may not obey physics...)
			ball_vel_new[1] = ball_vel_new[1] + a_2 * actionDistance
		else:
			# Point over
			r_new = (1, -1)

	# Create new state of the game
	x_new = [player_1_new, player_2_new]
	x_new.extend(ball_pos_new)
	x_new.extend(ball_vel_new)
	return np.array(x_new), r_new, player_1_new, player_2_new, ball_pos_new


def pongPhysics(x, a_1, a_2, plottingBool, axes, fig, line1, line2, line3, axbackground):
	# Author: Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu>
	# Date: 26 June 2020
	#
	# This function takes as input the current state of the game, x, the action
	# of each player, a_1 and a_2, and a Boolean, plottingBool, that determines
	# whether to plot the next state of the game.  This function returns the
	# next state of the game, x_new, as well as the reward, r_new, as given by:
	#
	# r_new[0] = R(x, a_1, x_new)
	# r_new[1] = R(x, a_2, x_new)

	x_new, r_new, player_1_new, player_2_new, ball_pos_new = execute_pongPhysics(x, a_1, a_2)

	paddleWidth = 0.1  # Width of each player's paddle

	# Plot the game
	if plottingBool:
		# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
		# plt.hold(False)
		# print(axes)
		# list(map(operator.add, player_1_new,[y*paddleWidth for y in [-1,1]]))
		axes[0].cla()
		# axes[0].plot([0,0], [player_1_new + x for x in [y*paddleWidth for y in [-1,1]]],'-b', linewidth=3)
		line1.set_data([0, 0], [player_1_new + x for x in [y * paddleWidth for y in [-1, 1]]])
		# axes.hold(True)
		# axes[0].plot([1,1], [player_2_new + x for x in [y*paddleWidth for y in [-1,1]]],'-r', linewidth=3)
		line2.set_data([1, 1], [player_2_new + x for x in [y * paddleWidth for y in [-1, 1]]])
		# axes[0].plot(ball_pos_new[0],ball_pos_new[1],'.k', markersize=12)
		line3.set_data(ball_pos_new[0], ball_pos_new[1])
		fig.canvas.restore_region(axbackground)
		axes[0].draw_artist(line1)
		axes[0].draw_artist(line2)
		axes[0].draw_artist(line3)
		fig.canvas.blit(axes[0].bbox)

		# axes[0].set_xlim(0,1)
		# axes[0].set_ylim(0,1)
		# fig.canvas.draw()
		fig.canvas.flush_events()
		fig.show()
	return x_new, r_new
