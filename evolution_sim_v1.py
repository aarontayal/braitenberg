"""
evolution_sim_v1
Started 12/23/2018
Aaron Tayal

Description: The goal of this version is to combine the hardy_weinberg simulation and the
elastic_collisions_v10 sim into a new version which incorporates reproduction with a constant
population size, as well as food which can give robots energy and increase their resources to
reproduce.

The overall control algorithm from hardy_weinberg_v3 will stay in place.  The food placement and
energy change routines will come from elastic_collisions_v10.  When a new robot is born, some energy
goes from the parents to the offspring.  Food gives robots energy as usual.  Robots have to use energy
to exert force and accelerate.  Robots with energy of zero cannot accelerate themselves.

When two robots reproduce at time t, the offspring is added at time t + 1, and the robot that dies
is taken out at time t + 1 (maintaining constant population size).  The robot with the lowest energy
at time t + 1 will be dead at time t + 1.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import evolution_sim_useful_functions_v1 as useful

# ---------------------------------------------------------------------------------------------------------------------

# World set-up parameters:
amountOfTime = 1000  # Number of iterations
dimBox = 40  # y dimension = x dimension of the box which holds the robots
wallBounceProportion = 1  # When a circle hits a wall, it is accelerated in either a north, east, south, or west
# direction.  The wallBounceProportion is between 0 and 1 and designates what percentage of the velocity in the
# direction of acceleration is absorbed by the wall (reducing the velocity of the robot in that direction)
wallBounceMultiplier = 0.5 + (0.5 * wallBounceProportion)  # showing this here for transparency.
frictionConst = 0.08  # Friction constant to increase or decrease the effect of friction. Note that right now the
# friction force is depending on a particular motor's velocity, that is, increases linearly with the velocity of motor

# ---------------------------------------------------------------------------------------------------------------------

# Phenotype bounds (lower, upper)
radiusBounds = (1, 1)
massBounds = (1, 1)  # **not** fraction of radius^2
motorDistBounds = (1, 1)  # fraction of the radius
eyeAngleBounds = (np.pi / 2, np.pi / 2)
eyeDistBounds = (0, 1)  # fraction of the radius
redBounds = (0, 1)  # between 0 and 1 to avoid error
greenBounds = (0, 1)
blueBounds = (0, 1)
phenBounds = np.array((radiusBounds, massBounds, motorDistBounds, eyeAngleBounds, eyeDistBounds, redBounds, greenBounds,
                    blueBounds))
# ---------------------------------------------------------------------------------------------------------------------

# Robot miscellaneous parameters
vMagInit = 0  # initial speed of center of the robots (meters/second)
genesPerSupergene = 2   # number of bits in one supergene, where each supergene controls a phenotype trait
initNumberOfRobots = 20
minDistToEye = 0.05  # to prevent extremely high numbers for 1/(distToEye**2)
maxDistToActivate = 10  # half the edge of a square centered at the robot k, where robots within the square will add to
# eye activation for robot k, while robots outside of the square will not
propelAdjust = 0.01  # to adjust the propelling force that the robots' motors exert
waitTimeRep = 10  # time between possible reproduction for each indiv. robot #*****************************************
initRobotEnergy = 30  # the energy that each robot will have when spawned at time = 0
searchStartBoxEdge = radiusBounds[1] * 5  # distance away to start looking for a spot for offspring
phenotypeToAnalyze = 4  # this is the phenotype which will be displayed on graphs at the end

# ---------------------------------------------------------------------------------------------------------------------
# Food parameters
foodRadius = 0.3
foodEnergy = 10
foodFrequency = 0.2  # number between 0 and 1.  this is the number of food spawned per second

# ---------------------------------------------------------------------------------------------------------------------


# The following arrays are current, meaning they run on the alive_index, meaning that they report data about robots that
# are alive at a given time.

bookkeep_c = np.zeros((amountOfTime, initNumberOfRobots, 2), np.uint16)
# Third dimension info:
# 0: tells the bookkeeping_index of each of the initNumberOfRobots that are alive at a given time.
# 1: tells the age of the robot

pos_c = np.zeros((amountOfTime, initNumberOfRobots, 7), np.float16)  # Reports position:
# 0: x pos of center
# 1: y pos of center
# 2: direction
# 3: x pos left eye
# 4: y pos left eye
# 5: x pos right eye
# 6: y pos right eye

vel_c = np.zeros((amountOfTime, initNumberOfRobots, 3), np.float16)  # Reports velocity:
# 0: x comp. veloc.
# 1: y comp. veloc.
# 2: angular veloc.

accel_c = np.zeros((amountOfTime, initNumberOfRobots, 3), np.float16)  # Reports acceleration:
# 0: x comp. accel.
# 1: y comp. accel.
# 2: angular accel.

collision_c = np.full((amountOfTime, initNumberOfRobots, 2), -1, dtype=np.int16)  # The third dimension of collision_c
# is for whether we are talking about running into the wall (0) or another robot (1).  The values in this array for the
# wall collisions are either -1 for no collision or 1 for collision.  The values for the robot collisions are the
# alive_index of the robot colliding with, or else -1 for no collision.

rep_c = np.full((amountOfTime, initNumberOfRobots), -1, dtype=np.int16)  #
# This array tells the alive_index of the robot that the a certain robot reproduced with, or else -1 for no one,
# at a particular time

can_rep = np.full((amountOfTime, initNumberOfRobots), False, dtype=bool)

gen_c = np.zeros((amountOfTime, initNumberOfRobots, 8, genesPerSupergene, 2), np.uint16)  # Tells the genotype.  The
# third dimension designates supergene and the fourth dimension designates individual genes in the supergene and the
# fifth dimension designates "maternal" or "paternal" (the point is that the robots are diploid)

phen_c = np.zeros((amountOfTime, initNumberOfRobots, 8), np.float16)  # Reports the phenotype.  The third dimension is
# which supergene we are talking about

energy_c = np.zeros((amountOfTime, initNumberOfRobots), np.float16)

# ---------------------------------------------------------------------------------------------------------------------
# food source array

foodPos = np.zeros((0, 2))  # x and y coordinates of each food.

eaten = np.zeros((amountOfTime, 0), dtype=bool)  # Whether or not each food is in existence at each
# point in time

# ---------------------------------------------------------------------------------------------------------------------

# From now on, the letters k and j indicate alive indices, while the letter b indicates bookkeeping index

for k in range(initNumberOfRobots):  # Create a starting population of robots

    randomGenotype1 = useful.create_random_genotype(8, genesPerSupergene)
    randomGenotype2 = useful.create_random_genotype(8, genesPerSupergene)
    (gen_c[0, k], phen_c[0, k], pos_c[0, k], vel_c[0, k], energy_c[0, k]) = \
        useful.initialize_robot(genotype_1=randomGenotype1, genotype_2=randomGenotype2, phenotype_bounds=phenBounds,
                                time_t_pos_c=pos_c[0, :k], time_t_radii=phen_c[0, :k, 0], init_number_of_robots=k,
                                dim_box=dimBox, v_mag_init=vMagInit, new_robot_index=k,
                                init_robot_energy=initRobotEnergy)
    bookkeep_c[0, k, :] = (k, 0)  # The robots start at age = 0 seconds

nextBookkeepIndex = initNumberOfRobots  # note that if there are 3 robots initially, then their bookkeeping indices are
# 0, 1, and 2, so the next bookkeeping index (for the fourth robot) would be 3.

print("Done initializing")

# ---------------------------------------------------------------------------------------------------------------------

# Below this line is where the main calculation for the sim is made, iterating from time = 0 to time = amountOfTime - 1
for t in range(amountOfTime-1):  # start with t = 0, go through t = (amountOfTime - 2)

    if t % 100 == 0:  # If the time t is a multiple of 100,
        print("t = ", t)  # Print the time t.  This is so that I can monitor the progress of the calculation.

    (foodPos, eaten) = useful.add_food(food_pos=foodPos, eaten=eaten, time=t, food_frequency=foodFrequency,
                                       dim_box=dimBox, amount_of_time=amountOfTime)

    for k in range(initNumberOfRobots):  # This loop will calculate the acceleration due to collisions and motors for
        # each robot k that is alive

        (pos_c[t], vel_c[t], accel_c[t], collision_c[t]) = \
            useful.evaluate_collisions(alive_index=k,
                                       time_t_phen_c=phen_c[t], time_t_pos_c=pos_c[t], time_t_vel_c=vel_c[t],
                                       time_t_accel_c=accel_c[t], time_t_collision_c=collision_c[t], dim_box=dimBox,
                                       wall_bounce_multiplier=wallBounceMultiplier,
                                       init_number_of_robots=initNumberOfRobots)
        # the above function calculates collisions with the wall or another robot and adjusts arrays of possibly both
        # robots in a collision, in addition to calculating the accelerations due to collisions

        (eaten[t + 1], deltaFoodEnergy) = useful.eat_food(food_pos=foodPos, time_t_eaten=eaten[t],
                                                          time_t_plus_1_eaten=eaten[t + 1],
                                                          food_energy=foodEnergy, food_radius=foodRadius,
                                                          robot_center_xy=pos_c[t, k, 0:2],
                                                          robot_radius=phen_c[t, k, 0])
        # find overlaps between food and robot k, and then set those particles to "eaten[t + 1, f] = True" and compute a
        # deltaFoodEnergy value for the robot k

        (deltaAccel, deltaAccelEnergy) = useful.motor_acceleration(alive_index=k, time_t_pos_c=pos_c[t],
                                                                   time_t_vel_c=vel_c[t], time_t_phen_c=phen_c[t],
                                                                   min_dist_to_eye_squared=minDistToEye,
                                                                   max_dist_to_activate=maxDistToActivate,
                                                                   friction_const=frictionConst,
                                                                   init_number_of_robots=initNumberOfRobots,
                                                                   propel_adjust=propelAdjust,
                                                                   energy_of_k=energy_c[t, k], food_pos=foodPos,
                                                                   time_t_eaten=eaten[t])


        accel_c[t, k, :] = accel_c[t, k, :] + deltaAccel
        # change x and y and angular components of the acceleration for robot k (alive_index) at time t, due to
        # propelling forces from motors as well as friction forces

        energy_c[t + 1, k] = energy_c[t, k] + deltaFoodEnergy + deltaAccelEnergy

        if energy_c[t + 1, k] <= 0:
            energy_c[t + 1, k] = 0

    #  Note that the following code must be run in a ***new*** "k in range" loop.
    #  The idea for the following is to calculate the arrays for time t + 1 *as if* none of the robots are going to
    # reproduce at time t
    phen_c[t + 1] = phen_c[t]
    gen_c[t + 1] = gen_c[t]
    bookkeep_c[t + 1, :, 0] = bookkeep_c[t, :, 0]
    for k in range(initNumberOfRobots):  # Here is where to update the velocity and position for *all alive robots k*
        vel_c[t + 1, k, 0:3] = vel_c[t, k, 0:3] + accel_c[t, k, 0:3]  # Calculate the velocity at time t + 1
        pos_c[t + 1, k, 0:3] = pos_c[t, k, 0:3] + vel_c[t + 1, k, 0:3]  # Calculate the position at time t + 1
        pos_c[t + 1, k, 3:7] = useful.get_eyes_pos(robot_x=pos_c[t + 1, k, 0], robot_y=pos_c[t + 1, k, 1],
                                                   robot_angle=pos_c[t + 1, k, 2], eye_angle=phen_c[t + 1, k, 3],
                                                   eye_dist=phen_c[t + 1, k, 4])
        bookkeep_c[t + 1, k, 1] = bookkeep_c[t, k, 1] + 1  # the robot is one second older in time (t + 1)

    for k in range(initNumberOfRobots):
            kAge = bookkeep_c[t, k, 1]
            noRepArray = np.full(waitTimeRep, -1, np.int16)
            if np.array_equal((rep_c[(t + 1 - waitTimeRep):(t + 1), k]), noRepArray) == True \
                    and energy_c[t + 1, k] > (initRobotEnergy) and kAge >= waitTimeRep:  # ***************************
                can_rep[t, k] = True
                j = collision_c[t, k, 1]
                if j != -1:
                    jAge = bookkeep_c[t, j, 1]
                    if np.array_equal((rep_c[(t + 1 - waitTimeRep):(t + 1), j]), noRepArray) == True \
                            and jAge >= waitTimeRep and energy_c[t + 1, j] >= (initRobotEnergy):  # ******************
                        rep_c[t, k] = j
                        rep_c[t, j] = k
                        energy_c[t + 1, k] = energy_c[t + 1, k] - (initRobotEnergy / 2)
                        energy_c[t + 1, j] = energy_c[t + 1, j] - (initRobotEnergy / 2)
                        """d = useful.pick_robot_to_die(time_t_bookkeeping_indices=bookkeep_c[t, :, 0],
                                                     time_t_plus_1_bookkeeping_indices=bookkeep_c[(t + 1), :, 0],
                                                     init_number_of_robots=initNumberOfRobots)"""
                        d = useful.pick_lowest_energy_robot(time_t_energy_c=energy_c[t])  # Find the robot with the
                        # lowest energy at time t.  This robot will die and be replaced by a baby robot a time t+1
                        (gen_c[t + 1, d], phen_c[t + 1, d], pos_c[t + 1, d], vel_c[t + 1, d], energy_c[t + 1, d]) = \
                            useful.initialize_robot(new_robot_index=d, mother_index=k, genotype_1=gen_c[t, k],
                                                    genotype_2=gen_c[t, j],
                                                    phenotype_bounds=phenBounds,
                                                    dim_box=dimBox, v_mag_init=vMagInit,
                                                    init_number_of_robots=initNumberOfRobots, time_t_pos_c=pos_c[t],
                                                    time_t_radii=phen_c[t, :, 0], init_robot_energy=initRobotEnergy,
                                                    search_start_box_edge=searchStartBoxEdge)  # Note that the mother
                        # index is so that the new robot can be placed near the mother
                        bookkeep_c[t + 1, d, :] = (nextBookkeepIndex, 0)  # The robots start at age = 0 seconds
                        # Create new robot in the dead robot's place with bookkeeping_index = nextBookkeepIndex
                        nextBookkeepIndex += 1



print("total number of robots ever alive = ", nextBookkeepIndex)

# ---------------------------------------------------------------------------------------------------------------------
# animation

fig = plt.figure(figsize=(6, 6))  # create a figure instance fig which has dimensions 5 in by 5 in
ax = plt.axes(xlim =(0, dimBox), ylim=(0, dimBox))  # create an axes instance \
# ax which has an x axis and y axis and belongs to fig
fig.patch.set(facecolor=(0.5, 0.5, 0.5))
ax.patch.set(facecolor=(0.9, 0.9, 0.9))
time_display = plt.text(x=dimBox - (dimBox / 6), y=dimBox - (dimBox / 20),
                        s="t = 0", color=(0.25, 0.25, 0.25, 0.5))  # create a text artist to display the time t
ax.add_artist(time_display)  # add the text to the axes ax

robotsList = []  # this will become a list of the matplotlib patches circles
leftEyesList = []
rightEyesList = []
foodList = []
centerList = []

for k in range(initNumberOfRobots):  # Put all the circle patches for the body, left eye, right eye, into the lists
    robotsList.append(plt.Circle(xy=(0, 0), radius=0,
                                 linewidth=1, edgecolor="none", facecolor=(0, 0, 0)))
    # the inside of each circle is colored a certain color, determined by the matrix P characters 5, 6, and 7
    centerList.append(plt.Circle(xy=(0, 0), radius=0, linewidth=1, edgecolor="none",
                                         facecolor="white"))
    leftEyesList.append(plt.Circle(xy=(0, 0), radius=0, linewidth=1, edgecolor="none",
                                   facecolor="white"))
    rightEyesList.append(plt.Circle(xy=(0, 0), radius=0, linewidth=1, edgecolor="none",
                                    facecolor="white"))


numberOfFood = foodPos.shape[0]
print("total number of food = ", numberOfFood)
for f in range(numberOfFood):
    foodList.append(plt.Circle(xy=(foodPos[f]), radius=foodRadius, linewidth=1, edgecolor="none", facecolor="green"))

for robot in robotsList:
    ax.add_patch(robot)  # add all of the circles to the axes ax
for center in centerList:
    ax.add_patch(center)
for left in leftEyesList:
    ax.add_patch(left)
for right in rightEyesList:
    ax.add_patch(right)

for food in foodList:
    ax.add_patch(food)


def init():  # Not really sure what this does
    return(robotsList, leftEyesList, rightEyesList, centerList, foodList)


def animate(t):  # This function will be repeatedly called for each t
    time_display.set_text(("t = " + str(t)))  # update the text of time_display to the current t
    for k in range(initNumberOfRobots):
        energyColor = math.atan(energy_c[t, k] / initRobotEnergy) / (np.pi / 2)
        centerList[k].set_facecolor((energyColor, energyColor, energyColor))
        if bookkeep_c[t, k, 1] == 0:
            robotsList[k].set_radius(phen_c[t, k, 0])
            centerList[k].set_radius(phen_c[t, k, 0] / 2)
            leftEyesList[k].set_radius(phen_c[t, k, 0] / 5)
            rightEyesList[k].set_radius(phen_c[t, k, 0] / 5)
            robotsList[k].set_facecolor((phen_c[t, k, 5], phen_c[t, k, 6], phen_c[t, k, 7]))
        robotsList[k].center = pos_c[t, k, 0:2]  # set the new locations of the circles at time t
        centerList[k].center = pos_c[t, k, 0:2]
        leftEyesList[k].center = pos_c[t, k, 3:5]
        rightEyesList[k].center = pos_c[t, k, 5:7]
        if can_rep[t, k] == True:
            robotsList[k].set_edgecolor("black")
        else:
            robotsList[k].set_edgecolor("none")
        if energy_c[t, k] < 0:
            print("error for robot ", k, " at time ", t, ".  energy at time ", t-1, " = ", energy_c[t - 1, k], ". energy at"
                                                                                                               "time ", t, " = ", energy_c[t, k])



    for f in range(numberOfFood):
        if eaten[t, f] == True:
            foodList[f].set_facecolor("none")
        else:
            foodList[f].set_facecolor("green")

    return(robotsList, leftEyesList, rightEyesList, foodList, time_display)
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

fig2, ax2 = plt.subplots(1, 2, sharey=True, tight_layout=True)
plt.style.use('ggplot')

numberBins = genesPerSupergene*2 + 1
if p == 4 or p == 2:
    binDist = radiusBounds[1] * (phenBounds[p, 1] - phenBounds[p, 0])/(genesPerSupergene*2)  # **************************** there may be problems with bins if radius varies
    xlimits = radiusBounds[0]*(phenBounds[p, 0] - 0.5*binDist), radiusBounds[1]*(phenBounds[p, 1] + 0.5*binDist)  # ***********************
else:
    binDist = (phenBounds[p, 1] - phenBounds[p, 0]) / (genesPerSupergene * 2)
    xlimits = ((phenBounds[p, 0] - 0.5 * binDist), (phenBounds[p, 1] + 0.5 * binDist))
binEdges = np.linspace(xlimits[0], xlimits[1], num=numberBins+1)

ax2[0].hist(phen_c[0, :, p], bins=binEdges, color="black")
ax2[0].set_xlabel("phenotype " + str(p))
ax2[0].set_xlim(xlimits)
ax2[0].set_ylabel("number of living robots at beginning")

ax2[1].hist(phen_c[amountOfTime-1, :, p], bins=binEdges, color="black")
ax2[1].set_xlabel("phenotype " + str(p))
ax2[1].set_xlim(xlimits)
ax2[1].set_ylabel("number of living robots at end")

fig3 = plt.figure()
ax3 = plt.axes()
if p == 4 or p == 2:
    ax3.set_ylim(radiusBounds[0] * phenBounds[p, 0], radiusBounds[1] * phenBounds[p, 1])
else:
    ax3.set_ylim(phenBounds[p, 0], phenBounds[p, 1])

averagePhenotypeToAnalyze = np.zeros(amountOfTime, dtype=np.float16)
for t in range(amountOfTime):
    averagePhenotypeToAnalyze[t] = np.average(phen_c[t, :, p])
ax3.plot(np.linspace(0, amountOfTime - 1, amountOfTime), averagePhenotypeToAnalyze)
plt.show()