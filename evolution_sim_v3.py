"""
Aaron Tayal
evolution_sim_v3
Started

Description: This version will introduce brain systems into the robots.  It will allow robots to
interact with each other.

Questions for future versions:
-How to organize parameters, and store param combos?
-Is there a better way other than passing params to functions in useful?
-How can I build an interactive window for data analysis?  For example, I want to be able to see an animated
histogram while I watch the animation of the physical world of the robots
-How to organize the genes of the robots and how the genes are interpreted in the context of phenotype
-How to use maxDistToActivate to effectively limit the number of distances that need to be calculated between eyes
and centers, without creating large discontinuities in the distance->activation graph
-How to build the neural network of the robots.  Should the number of nodes be easily adjustable?
-How to visually represent the connections of the neural network
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
import evolution_sim__useful_functions_v3 as useful


# ---------------------------------------------------------------------------------------------------------------------

# World set-up parameters:
amountOfTime = 1000  # Number of iterations
dimBox = 40  # y dimension = x dimension of the box which holds the robots
wallBounceProportion = 1  # When a circle hits a wall, it is accelerated in either a north, east, south, or west
# direction.  The wallBounceProportion is between 0 and 1 and designates what percentage of the velocity in the
# direction of acceleration is absorbed by the wall (reducing the velocity of the robot in that direction)
wallBounceMultiplier = 0.5 + (0.5 * wallBounceProportion)  # showing this here for transparency.
frictionConst = 0.2  # Friction constant to increase or decrease the effect of friction. Note that right now the
# friction force is depending on a particular motor's velocity, that is, increases linearly with the velocity of motor

# ---------------------------------------------------------------------------------------------------------------------

# Phenotype bounds (lower, upper)
radiusBounds = (1, 1)
massBounds = (1, 1)  # **not** fraction of radius^2
motorDistBounds = (1, 1)  # fraction of the radius
eyeAngleBounds = (np.pi/2, np.pi/2)
eyeDistBounds = (1, 1)  # fraction of the radius
redBounds = (0, 1)  # between 0 and 1 to avoid error
greenBounds = (0, 1)
blueBounds = (0, 1)
phenBounds = np.array((radiusBounds, massBounds, motorDistBounds, eyeAngleBounds, eyeDistBounds, redBounds, greenBounds,
                    blueBounds))
# ---------------------------------------------------------------------------------------------------------------------

# Robot miscellaneous parameters
vMagInit = 0  # initial speed of center of the robots (meters/second)
genesPerSupergene = 2   # number of bits in one supergene, where each supergene controls a phenotype trait

minDistToEyeSquared = 0.5  # to prevent extremely high numbers for 1/(distToEye**2)
maxDistToActivate = 10  # half the edge of a square centered at the robot k, where robots within the square will add to
# eye activation for robot k, while robots outside of the square will not, foods within the square will add to ...
waitTimeRep = 50  # time between possible reproduction for each indiv. robot
initRobotEnergy = 40  # the energy that each robot will have when spawned at time = 0
metabolism = -0.04
maxEnergy = 100
searchStartBoxEdge = radiusBounds[1] * 5  # distance away to start looking for a spot for offspring
phenotypeToAnalyze = 5  # this is the phenotype which will be displayed on graphs at the end
robotToTrack = 0  # This is the storage index of the robot which will be animated (showing its neural network working)

maxRobots = 60
initNumberOfRobots = 10
# ---------------------------------------------------------------------------------------------------------------------
# Robot neural net parameters

robEyeActivAdjust = 1
foodEyeActivAdjust = 1
propelAdjust = 0.2  # to adjust the propelling force that the robots' motors exert

init_weights_0th_lay = np.array(((1, 0, 0, 0, 0.00125),
                                 (0, 1, 0, 0, 0.00125),
                                 (0, 0, 1, 0, -0.00125),
                                 (0, 0, 0, 1, -0.00125),
                                 (0, 0, 0, 0, 0)), np.float16)

init_weights_1st_lay = np.array(((0, 1, 0, 1, 0),
                                 (1, 0, 1, 0, 0)), np.float16)

init_biases_0th_lay = np.zeros((5), np.float16)
init_biases_1st_lay = np.zeros((2), np.float16)

# ---------------------------------------------------------------------------------------------------------------------
# Food parameters
foodRadius = 0.3
foodEnergy = 15
# Equation for food growth: dF/dt = foodRate*F*(1-F/foodCarryingCapac) where F is the number of living food
foodCarryingCapac = 20
foodRate = 0.04
initNumberOfFood = 10

# ---------------------------------------------------------------------------------------------------------------------


# The following arrays are current, meaning they run on the alive_index, meaning that they report data about robots that
# are alive at a given time.

bookkeep = np.full((amountOfTime, maxRobots, 2), -1, np.int16)
# Third dimension info:
# 0: tells the bookkeeping_index of each of the robots that are alive at a given time.
# 1: tells the age of the robot

robotAlive = np.full((amountOfTime, maxRobots), False, dtype=bool)  # Whether or not each robot is in existence
# at each point in time

pos = np.full((amountOfTime, maxRobots, 7), -1, np.float16)  # Reports position:
# 0: x pos of center
# 1: y pos of center
# 2: direction
# 3: x pos left eye
# 4: y pos left eye
# 5: x pos right eye
# 6: y pos right eye

vel = np.full((amountOfTime, maxRobots, 3), -1, np.float16)  # Reports velocity:
# 0: x comp. veloc.
# 1: y comp. veloc.
# 2: angular veloc.

accel = np.full((amountOfTime, maxRobots, 3), -1, np.float16)  # Reports acceleration:
# 0: x comp. accel.
# 1: y comp. accel.
# 2: angular accel.

collision = np.full((amountOfTime, maxRobots, 2), -1, dtype=np.int16)  # The third dimension of collision_c
# is for whether we are talking about running into the wall (0) or another robot (1).  The values in this array for the
# wall collisions are either -1 for no collision or 1 for collision.  The values for the robot collisions are the
# alive_index of the robot colliding with, or else -1 for no collision.

rep = np.full((amountOfTime, maxRobots), -1, dtype=np.int16)
# This array tells the alive_index of the robot that the a certain robot reproduced with, or else -1 for no one,
# at a particular time

can_rep = np.full((amountOfTime, maxRobots), False, dtype=bool)
# This array stores whether a robot can reproduce at a certain time, so that we can draw a circle for the animation

gen = np.full((amountOfTime, maxRobots, 8, genesPerSupergene, 2), -1, np.int16)  # Tells the genotype.  The
# third dimension designates supergene and the fourth dimension designates individual genes in the supergene and the
# fifth dimension designates "maternal" or "paternal" (the point is that the robots are diploid)

phen = np.full((amountOfTime, maxRobots, 8), -1, np.float16)  # Reports the phenotype.  The third dimension is
# which supergene we are talking about

energy = np.full((amountOfTime, maxRobots), -1, np.float16)

sensor_neurons = np.full((amountOfTime, maxRobots, 5), -1, np.float16)  # This records the degree to which the eye
# neurons light up.  This helps to train the neural net of new robots based on the behavior of the parents.
# The last dimension is 0:robots left eye, 1:robots right eye, 2:food left eye, 3: food right eye, 4: energy

inter_neurons = np.full((amountOfTime, maxRobots, 5), -1, np.float16)

motor_neurons = np.full((amountOfTime, maxRobots, 2), -1, np.float16)  # This records the degree to which the motor
# neurons light up.  This helps to train the neural net of new robots based on the behavior of the parents.

weights_0th_lay = np.full((amountOfTime, maxRobots, 5, 5), -1, np.float16)

weights_1st_lay = np.full((amountOfTime, maxRobots, 2, 5), -1, np.float16)

biases_0th_lay = np.full((amountOfTime, maxRobots, 5), -1, np.float16)

biases_1st_lay = np.full((amountOfTime, maxRobots, 2), -1, np.float16)

# ---------------------------------------------------------------------------------------------------------------------
# food source array

foodPos = np.full((amountOfTime, foodCarryingCapac, 2), -1, np.float16)  # x and y coordinates of each food.

foodAlive = np.full((amountOfTime, foodCarryingCapac), False, dtype=bool)  # Whether or not each food is in existence
# at each point in time

# ---------------------------------------------------------------------------------------------------------------------

for s in range(initNumberOfRobots):  # Create a starting population of robots.

    randomGenotype1 = useful.create_random_genotype(8, genesPerSupergene)
    randomGenotype2 = useful.create_random_genotype(8, genesPerSupergene)
    (gen[0, s], phen[0, s], pos[0, s], vel[0, s], energy[0, s], ) = \
        useful.initialize_robot(new_robot_index=s, genotype_1=randomGenotype1, genotype_2=randomGenotype2,
                                phenotype_bounds=phenBounds, robot_positions=pos[0, :s],
                                robot_radii=phen[0, :s, 0], dim_box=dimBox, v_mag_init=vMagInit,
                                init_robot_energy=initRobotEnergy, alives=np.where(robotAlive[0, :]==True)[0],
                                pos_to_be_born_near="none")
    bookkeep[0, s, :] = (s, 0)  # The robots start at age = 0 seconds
    robotAlive[:, s] = True  # Set robot s to be alive for the rest of time (for now)
    (weights_0th_lay[0, s], weights_1st_lay[0, s], biases_0th_lay[0, s], biases_1st_lay[0, s]) = \
        (init_weights_0th_lay, init_weights_1st_lay, init_biases_0th_lay, init_biases_1st_lay)

nextBookkeepIndex = initNumberOfRobots  # note that if there are 3 robots initially, then their bookkeeping indices are
# 0, 1, and 2, so the next bookkeeping index (for the fourth robot) would be 3.


for f in range(initNumberOfFood):
    foodAlive[:, f] = True  # set food f to be alive for the rest of time (for now)
    foodPos[:, f, :] = (np.random.rand()*dimBox, np.random.rand()*dimBox)  # Set the position of food f for the rest of
    # time (this will possibly be altered later)

print("Done initializing")

# ---------------------------------------------------------------------------------------------------------------------

# Below this line is where the main calculation for the sim is made, iterating from time = 0 to time = amountOfTime - 1
for t in range(amountOfTime-1):  # start with t = 0, go through t = (amountOfTime - 2)

    if t % 100 == 0:  # If the time t is a multiple of 100,
        print("t = ", t)  # Print the time t.  This is so that I can monitor the progress of the calculation.

    F = np.sum(foodAlive[t])  # number of food that are alive at time t
    # print("F = ", F)
    dF_dt = foodRate*F*(1-F/foodCarryingCapac)  # rate of addition of new food
    # print("dF_dt = ", dF_dt)
    if np.random.rand() < (dF_dt - math.floor(dF_dt)):  # For example, if dF_dt = 2.2, then the random between 0 and 1
        # would be less than 0.2 one fifth of the time, so then dF_dt_integer would be 3 one fifth of the time
        dF_dt_integer = math.ceil(dF_dt)
    else:
        dF_dt_integer = math.floor(dF_dt)
    # print("dF_dt_integer = ", dF_dt_integer)
    for f in range(dF_dt_integer):
        minFalseIndex = np.array(np.where(foodAlive[t, :] == False))[0][0]
        foodAlive[t:, minFalseIndex] = True  # find the first "False" in foodAlive[t], and make it into a "True" for the
        # rest of time
        foodPos[t:, minFalseIndex, :] = (np.random.rand()*dimBox, np.random.rand()*dimBox)  # find that same index in
        # foodPos[t] and find a new random location

    alives = np.where(robotAlive[t, :] == True)[0]  # all indices of the robots alive at time t

    for k in alives:  # This loop will calculate the acceleration due to *********************************
        # collisions and motors for each robot k that is alive

        (pos[t], vel[t], accel[t], collision[t]) = \
            useful.evaluate_collisions(k=k, alives=alives,
                                       time_t_phen_c=phen[t], time_t_pos_c=pos[t], time_t_vel_c=vel[t],
                                       time_t_accel_c=accel[t], time_t_collision_c=collision[t], dim_box=dimBox,
                                       wall_bounce_multiplier=wallBounceMultiplier,
                                       init_number_of_robots=initNumberOfRobots)
        # the above function calculates collisions with the wall or another robot and adjusts arrays of possibly both
        # robots in a collision, in addition to calculating the accelerations due to collisions

        (foodAlive, deltaFoodEnergy) = useful.eat_food(time=t, food_pos=foodPos,
                                                       food_alive=foodAlive,
                                                       food_energy=foodEnergy, food_radius=foodRadius,
                                                       robot_center_xy=pos[t, k, 0:2],
                                                       robot_radius=phen[t, k, 0])
        # find overlaps between food and robot k, and then set those particles to "eaten[t + 1, f] = True" and compute a
        # deltaFoodEnergy value for the robot k

        (deltaAccel, deltaAccelEnergy, sensor_neurons[t, k], inter_neurons[t, k], motor_neurons[t, k]) = \
            useful.motor_acceleration(k=k, alives=alives, time_t_pos_c=pos[t],
                                      time_t_vel_c=vel[t], time_t_phen_c=phen[t],
                                      robot_energy=energy[t, k],
                                      init_robot_energy=initRobotEnergy,
                                      min_dist_to_eye_squared=minDistToEyeSquared,
                                      max_dist_to_activate=maxDistToActivate,
                                      friction_const=frictionConst,
                                      init_number_of_robots=initNumberOfRobots,
                                      propel_adjust=propelAdjust,
                                      rob_eye_activ_adjust=robEyeActivAdjust,
                                      food_eye_activ_adjust=foodEyeActivAdjust,
                                      time_t_food_pos=foodPos[t],
                                      time_t_food_alive=foodAlive[t],
                                      k_weights_0th_lay=weights_0th_lay[t, k],
                                      k_weights_1st_lay=weights_1st_lay[t, k],
                                      k_biases_0th_lay=biases_0th_lay[t, k],
                                      k_biases_1st_lay=biases_1st_lay[t, k])
        # Above, we use the positions of other robots and food, as well as info about robot k's brain, to figure out
        # how the robot chooses to propel itself.  Then, we factor in the force of friction and finally use Newton's
        # Second Law to find the acceleration of robot k (including the angular acceleration).  Along the way, we
        # record how the robot's neural net is thinking, and how much energy the robot uses as it uses its motors.

        accel[t, k, :] = accel[t, k, :] + deltaAccel
        # change x and y and angular components of the acceleration for robot k (alive_index) at time t, due to
        # propelling forces from motors as well as friction forces

        energy[t + 1, k] = energy[t, k] + deltaFoodEnergy + deltaAccelEnergy + metabolism  # deltaAccelEnergy and
        # metabolism are negative
        if energy[t + 1, k] > maxEnergy:
            energy[t + 1, k] = maxEnergy


    #  Note that the following code must be run in a ***new*** "k" loop.
    #  The idea for the following is to calculate the arrays for time t + 1 *as if* none of the robots are going to
    # reproduce at time t

    for k in alives:  # Here is where to update the velocity and position for *all alive robots k*
        gen[t + 1, k] = gen[t, k]  # for all robots and all genes in the robots
        phen[t + 1, k] = phen[t, k]  # for all robots and all phenotypes
        (weights_0th_lay[t+1, k], weights_1st_lay[t+1, k], biases_0th_lay[t+1, k], biases_1st_lay[t+1, k]) = \
            (weights_0th_lay[t, k], weights_1st_lay[t, k], biases_0th_lay[t, k], biases_1st_lay[t, k])
        vel[t + 1, k, 0:3] = vel[t, k, 0:3] + accel[t, k, 0:3]  # Calculate the velocity at time t + 1
        pos[t + 1, k, 0:3] = pos[t, k, 0:3] + vel[t + 1, k, 0:3]  # Calculate the position at time t + 1
        pos[t + 1, k, 3:7] = useful.get_eyes_pos(robot_x=pos[t + 1, k, 0], robot_y=pos[t + 1, k, 1],
                            robot_angle=pos[t + 1, k, 2], eye_angle=phen[t + 1, k, 3],
                            eye_dist=phen[t + 1, k, 4])
        bookkeep[t + 1, k, 1] = bookkeep[t, k, 1] + 1  # the robot is one second older in time (t + 1)

        bookkeep[t + 1, k, 0] = bookkeep[t, k, 0]  # keep the bookkeeping indices the same for now
        # The following for loop figures out which robots reproduce and creates the new robots

    # The following loop figures out which robots reproduce at time t
    newRobots = []
    for k in alives:
            kAge = bookkeep[t, k, 1]
            noRepArray = np.full(waitTimeRep, -1, np.int16)  # this array will be compared to the rep_c array for robot k
            if np.array_equal((rep[(t + 1 - waitTimeRep):(t + 1), k]), noRepArray) == True \
                    and energy[t + 1, k] > (initRobotEnergy) and kAge >= waitTimeRep:  # TODO: This is an opportunity to improve the conditions under which a robot can reproduce
                can_rep[t, k] = True
                j = collision[t, k, 1]  # robot collisions (-1 if no one, index of other robot if collided)
                if j != -1:  # if robot k collided with another robot at time t
                    jAge = bookkeep[t, j, 1]
                    if np.array_equal((rep[(t + 1 - waitTimeRep):(t + 1), j]), noRepArray) == True \
                            and jAge >= waitTimeRep and energy[t + 1, j] >= (initRobotEnergy):  # ******************
                        rep[t, k] = j
                        rep[t, j] = k
                        energy[t + 1, k] = energy[t + 1, k] - (initRobotEnergy / 2)  # TODO: right now, each robot contributes half the energy of the new robot
                        energy[t + 1, j] = energy[t + 1, j] - (initRobotEnergy / 2)
                        newRobots.append((k, j))

    # The following loop figures out which robots die between t and t+1
    for k in alives:
        if energy[t+1, k] <= 0:
            robotAlive[(t+1):,k] = False
            # TODO: make all the arrays for robot k at time t+1 equal -1

    # The following loop creates the newborn robots at time t+1
    for n in range(len(newRobots)):
        if np.all(robotAlive[t+1, :]):  # If there are no more places to fill with new robots
            print("Error: robot population exceeded array space")
            break
        d = (np.where(robotAlive[t+1, :]==False))[0][0]  # Find the first index that contains a dead robot at time t+1
        (gen[t + 1, d], phen[t + 1, d],
         pos[t + 1, d], vel[t + 1, d], energy[t + 1, d]) = \
            useful.initialize_robot(new_robot_index=d, genotype_1=gen[t, newRobots[n][0]],
                                    genotype_2=gen[t, newRobots[n][1]], phenotype_bounds=phenBounds,
                                    robot_positions=pos[t+1, :, :], robot_radii=phen[t+1, :, 0],
                                    alives=(np.where(robotAlive[t+1, :]==True))[0], dim_box=dimBox, v_mag_init=vMagInit,
                                    init_robot_energy=initRobotEnergy,
                                    search_start_box_edge=searchStartBoxEdge,
                                    pos_to_be_born_near=pos[t, newRobots[n][0], 0:2])
        # the parents (indices newRobots[n]) create a new robot at index d at time t+1
        robotAlive[(t + 1):, d] = True
        bookkeep[t + 1, d, :] = (nextBookkeepIndex, 0)  # The robots start at age = 0 seconds
        # Create new robot in the dead robot's place with bookkeeping_index = nextBookkeepIndex
        nextBookkeepIndex += 1

        (weights_0th_lay[t + 1, d], weights_1st_lay[t + 1, d], biases_0th_lay[t + 1, d], biases_1st_lay[t + 1, d]) = \
            (init_weights_0th_lay, init_weights_1st_lay, init_biases_0th_lay, init_biases_1st_lay)
        """
        training_data = 0  
        (weights_0th_lay[t+1, d], weights_1st_lay[t+1, d], biases_0th_lay[t+1, d], biases_1st_lay[t+1, d]) = \
            useful.train_neural_net(training_data=training_data)
        """
        # TODO: Create training data from parents' lives

print("total number of robots ever alive = ", nextBookkeepIndex)

# ---------------------------------------------------------------------------------------------------------------------
# animation

fig = plt.figure(figsize=(12, 6))  # create a figure instance fig which has dimensions
fig.patch.set(facecolor=(0.5, 0.5, 0.5))

# ---------------------------------------------------------------------------------------------------------------------
# Set up axHabitat:

axHabitat = fig.add_subplot(1, 2, 1, xlim=(0, dimBox), ylim=(0, dimBox), facecolor=(0.9, 0.9, 0.9))

time_display = axHabitat.text(x=dimBox - (dimBox / 6), y=dimBox - (dimBox / 20),
                        s="t = 0", color=(0.25, 0.25, 0.25, 0.5))  # create a text artist to display the time t
trackingCircle = axHabitat.add_patch(plt.Circle((0, 0), radius=1.5, edgecolor=(1, 1, 0, 0.5), facecolor="none"))

robotsList = []  # this will become a list of the matplotlib patches circles
leftEyesList = []
rightEyesList = []
foodList = []
centerList = []

for k in range(maxRobots):  # Put all the circle patches for the body, left eye, right eye, into the lists
    robotsList.append(plt.Circle(xy=(0, 0), radius=0,
                                 linewidth=1, edgecolor="none", facecolor=(0, 0, 0)))
    # the inside of each circle is colored a certain color, determined by the matrix P characters 5, 6, and 7
    centerList.append(plt.Circle(xy=(0, 0), radius=0, linewidth=1, edgecolor="none",
                                         facecolor="white"))
    leftEyesList.append(plt.Circle(xy=(0, 0), radius=0, linewidth=1, edgecolor="none",
                                   facecolor="white"))
    rightEyesList.append(plt.Circle(xy=(0, 0), radius=0, linewidth=1, edgecolor="none",
                                    facecolor="white"))

for f in range(foodCarryingCapac):
    foodList.append(plt.Circle(xy=(0, 0), radius=foodRadius, linewidth=1, edgecolor="none", facecolor="green"))

for robot in robotsList:
    axHabitat.add_patch(robot)  # add all of the circles to the axes ax
for center in centerList:
    axHabitat.add_patch(center)
for left in leftEyesList:
    axHabitat.add_patch(left)
for right in rightEyesList:
    axHabitat.add_patch(right)
for food in foodList:
    axHabitat.add_patch(food)

# ---------------------------------------------------------------------------------------------------------------------
# Set up axBrain:

axBrain = fig.add_subplot(1, 2, 2, xlim=(-10, 10), ylim=(-10, 10), facecolor=(0.9, 0.9, 0.9))
axBrain.add_patch(plt.Circle(xy=(0, 0), radius=9, linewidth=1, edgecolor="none", facecolor=(1, 0.5, 0.5)))

nodesList = [0 for layer in range(3)]
nodesList[0] = [0 for s in range(sensor_neurons.shape[2])]
nodesList[1] = [0 for i in range(inter_neurons.shape[2])]
nodesList[2] = [0 for m in range(motor_neurons.shape[2])]
connectionsList = [0 for layer in range(2)]
connectionsList[0] = [[0 for s in range(sensor_neurons.shape[2])] for i in range(inter_neurons.shape[2])]
connectionsList[1] = [[0 for i in range(inter_neurons.shape[2])] for m in range(motor_neurons.shape[2])]
#biasesList =

spread_sensor_neurons = np.linspace(-7, 7, sensor_neurons.shape[2])
spread_inter_neurons = np.linspace(-7, 7, inter_neurons.shape[2])
spread_motor_neurons = np.linspace(-7, 7, motor_neurons.shape[2])

for s in range(sensor_neurons.shape[2]):
    nodesList[0][s] = (plt.Circle(xy=(spread_sensor_neurons[s], 4), radius=0.25, facecolor="white"))

for i in range(inter_neurons.shape[2]):
    nodesList[1][i] = (plt.Circle(xy=(spread_inter_neurons[i], 0), radius=0.25, facecolor="white"))
    for s in range(sensor_neurons.shape[2]):
        connectionsList[0][i][s] = (plt.Line2D((spread_inter_neurons[i], spread_sensor_neurons[s]), (0, 4),
                                          linewidth=0.5, color="white"))

for m in range(motor_neurons.shape[2]):
    nodesList[2][m] = (plt.Circle(xy=(spread_motor_neurons[m], -4), radius=0.25, facecolor="white"))
    for i in range(inter_neurons.shape[2]):
        connectionsList[1][m][i] = (mlines.Line2D((spread_motor_neurons[m], spread_inter_neurons[i]), (-4, 0),
                                          linewidth=0.5, color="white"))


for layer in nodesList:
    for node in layer:
        axBrain.add_patch(node)

for layer in connectionsList:
    for destination in layer:
        for origin in destination:
            axBrain.add_artist(origin)

# ---------------------------------------------------------------------------------------------------------------------

def init():  # Not really sure what this does
    return(robotsList, leftEyesList, rightEyesList, centerList, foodList)

def animate(t):  # This function will be repeatedly called for each t
    time_display.set_text(("t = " + str(t)))  # update the text of time_display to the current t
    for k in range(maxRobots):
        if robotAlive[t, k] == True:
            # --------------------------color:
            energyColor = math.atan(energy[t, k] / initRobotEnergy) / (np.pi / 2)  # A sigmoid type function
            centerList[k].set_facecolor((energyColor, energyColor, energyColor))  # as the energy gets higher, the
            # center of the robot gets lighter
            if can_rep[t, k] == True:  # if the robot is capable of reproducing at time t
                robotsList[k].set_edgecolor("black")
            else:
                robotsList[k].set_edgecolor("none")

            # --------------------------initializing a robot:
            if bookkeep[t, k, 1] == 0:  # If the robot is age=0 at time t
                robotsList[k].set_radius(phen[t, k, 0])
                centerList[k].set_radius(phen[t, k, 0] / 2)
                leftEyesList[k].set_radius(phen[t, k, 0] / 5)  # just an aesthetic of how big the eyes should be
                rightEyesList[k].set_radius(phen[t, k, 0] / 5)
                robotsList[k].set_facecolor(
                    (phen[t, k, 5], phen[t, k, 6], phen[t, k, 7]))  # set the color of the robot

            #----------------------------location:
            robotsList[k].center = pos[t, k, 0:2]  # set the new locations of the circles at time t
            centerList[k].center = pos[t, k, 0:2]
            leftEyesList[k].center = pos[t, k, 3:5]
            rightEyesList[k].center = pos[t, k, 5:7]

        else:
            robotsList[k].center = (-10, -10)
            rightEyesList[k].center = (-10, -10)
            leftEyesList[k].center = (-10, -10)
            centerList[k].center = (-10, -10)

    for f in range(foodCarryingCapac):
        if foodAlive[t, f] != foodAlive[t-1, f] or t==0:
            if foodAlive[t, f] == False:
                foodList[f].set_facecolor("none")
            else:
                foodList[f].set_facecolor("green")
        if foodPos[t, f, 0] != foodPos[t-1, f, 0] or foodPos[t, f, 1] != foodPos[t-1, f, 1] or t==0:
            foodList[f].center = foodPos[t, f, :]  # TODO: I could move this into the above if statement

    k = robotToTrack
    trackingCircle.center = pos[t, k, 0:2]  # This puts a marker on the current robot being monitored
    if t != (amountOfTime - 1):
        for s in range(sensor_neurons.shape[2]):
            value = useful.zero_zero_sigmoid(sensor_neurons[t, k, s])
            if value<0:
                print("I found a problem :( the value is less than zero")  # TODO: I think the energy is going below zero (when a robot dies)
            nodesList[0][s].set_facecolor((value, value, value))

        for i in range(inter_neurons.shape[2]):
            value = useful.zero_zero_sigmoid(inter_neurons[t, k, i])
            nodesList[1][i].set_facecolor((value, value, value))
            for s in range(sensor_neurons.shape[2]):
                value = (1 + weights_0th_lay[t, k, i, s]) / 2
                connectionsList[0][i][s].set_color((value, value, value))

        for m in range(motor_neurons.shape[2]):
            value = useful.zero_zero_sigmoid(motor_neurons[t, k, m])
            nodesList[2][m].set_facecolor((value, value, value))
            for i in range(inter_neurons.shape[2]):
                value = (1 + weights_1st_lay[t, k, m, i]) / 2
                connectionsList[1][m][i].set_color((value, value, value))

    return(robotsList, leftEyesList, rightEyesList, foodList, connectionsList, nodesList, time_display)
    # give the list of circles patches, particles, and
    # the text artist time_display, back to FuncAnimation


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=amountOfTime, interval=50, blit=False)
# fig is the figure which was made
# animate is the function which will be repeatedly called
# interval is the interval between frames

plt.show()  # show the animation which was just created

# ---------------------------------------------------------------------------------------------------------------------
# analysis

p = phenotypeToAnalyze

fig2, axs2 = plt.subplots(1, 2, sharey=True, tight_layout=True)
#plt.style.use('ggplot')

numberBins = genesPerSupergene*2 + 1
if p == 4 or p == 2:
    binDist = radiusBounds[1] * (phenBounds[p, 1] - phenBounds[p, 0])/(genesPerSupergene*2)  # **************************** there may be problems with bins if radius varies
    xlimits = radiusBounds[0]*(phenBounds[p, 0] - 0.5*binDist), radiusBounds[1]*(phenBounds[p, 1] + 0.5*binDist)  # ***********************
else:
    binDist = (phenBounds[p, 1] - phenBounds[p, 0]) / (genesPerSupergene * 2)
    xlimits = ((phenBounds[p, 0] - 0.5 * binDist), (phenBounds[p, 1] + 0.5 * binDist))
binEdges = np.linspace(xlimits[0], xlimits[1], num=numberBins+1)

axs2[0].hist(phen[0, :, p], bins=binEdges, color="black")
axs2[0].set_xlabel("phenotype " + str(p))
axs2[0].set_xlim(xlimits)
axs2[0].set_ylabel("number of living robots at beginning")

axs2[1].hist(phen[amountOfTime-1, :, p], bins=binEdges, color="black")
axs2[1].set_xlabel("phenotype " + str(p))
axs2[1].set_xlim(xlimits)
axs2[1].set_ylabel("number of living robots at end")

plt.show()

"""
fig3 = plt.figure()
ax3 = plt.axes()
if p == 4 or p == 2:
    ax3.set_ylim(radiusBounds[0] * phenBounds[p, 0], radiusBounds[1] * phenBounds[p, 1])
else:
    ax3.set_ylim(phenBounds[p, 0], phenBounds[p, 1])

averagePhenotypeToAnalyze = np.zeros(amountOfTime, dtype=np.float16)
for t in range(amountOfTime):
    averagePhenotypeToAnalyze[t] = np.average(phen[t, :, p])  # this is broken because the program should jus look at alive robots
ax3.plot(np.linspace(0, amountOfTime - 1, amountOfTime), averagePhenotypeToAnalyze)
"""

fig3 = plt.figure()
ax3 = plt.axes()

robotPopulation = np.zeros(amountOfTime, dtype=np.int16)
foodPopulation = np.zeros(amountOfTime, dtype=np.int16)
for t in range(amountOfTime):
    robotPopulation[t] = np.sum(robotAlive[t, :])
    foodPopulation[t] = np.sum(foodAlive[t, :])
robotPopLine, = ax3.plot(np.linspace(0, amountOfTime - 1, amountOfTime), robotPopulation, 'black')
foodPopLine, = ax3.plot(np.linspace(0, amountOfTime - 1, amountOfTime), foodPopulation, 'green')
ax3.set_xlabel("time")
ax3.set_ylabel("population")
ax3.legend((robotPopLine, foodPopLine), ("robots", "food"))
plt.show()