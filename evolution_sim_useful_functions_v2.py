import numpy as np
import math as math


def dist(point_a, point_b):  # inputs are x,y coordinates of two points a and b, and output is the distance in the plane
    return (math.sqrt(math.pow((point_a[0] - point_b[0]), 2) + math.pow((point_a[1] - point_b[1]), 2)))


def dist_squared(point_a, point_b):
    return (math.pow((point_a[0] - point_b[0]), 2) + math.pow((point_a[1] - point_b[1]), 2))


def polar_to_rect(radius, angle):
    return ((radius * np.cos(angle), radius * np.sin(angle)))


def bouncing_off_wall(R_x, R_y, radius, dim_of_box, v_x, v_y, wall_bounce_multiplier):
    touchingWall = -1
    a_x = 0
    a_y = 0
    if (R_x - radius < 0) or (R_x + radius > dim_of_box):  # if the x comp. of position is less than 0
        touchingWall = 1
        a_x = -2 * v_x * wall_bounce_multiplier  # make the new x component of acceleration the negative of the x comp. of velocity
        if R_x - radius < 0:
            R_x = radius
        if R_x + radius > dim_of_box:
            R_x = dim_of_box - radius
    if (R_y - radius < 0) or (R_y + radius > dim_of_box):
        touchingWall = 1
        a_y = -2 * v_y * wall_bounce_multiplier
        if R_y - radius < 0:
            R_y = radius
        if R_y + radius > dim_of_box:
            R_y = dim_of_box - radius
    return ((a_x, a_y, touchingWall, R_x, R_y))


def AABB_collision_test(R1_x, R1_y, radius1, R2_x, R2_y, radius2):
    """tests if bounding boxes of two circles are overlapping in the plane"""
    colliding = False
    if ((R1_x + radius1) > (R2_x - radius2)) \
            and ((R2_x + radius2) > (R1_x - radius1)) \
            and ((R1_y + radius1) > (R2_y - radius2)) \
            and ((R2_y + radius2) > (R1_y - radius1)):
        colliding = True
    return (colliding)


def one_dimensional_elastic(v_1, m_1, v_2, m_2):
    """given the initial one dimensional velocities and masses, this function outputs the acceleration for each object
    in one dimension"""
    a_1 = ((v_1 * (m_1 - m_2) + 2 * m_2 * v_2) / (m_1 + m_2) - v_1)
    a_2 = ((v_2 * (m_2 - m_1) + 2 * m_1 * v_1) / (m_2 + m_1) - v_2)
    return ((a_1, a_2))


"""
#this is a program to test the 1D elastic collision formula
massA = 5
massB = 10
vinitA = 5
vinitB = -5
aA, aB = one_dimensional_elastic(vinitA, massA, vinitB, massB)
vfinalA = vinitA + aA
vfinalB = vinitB + aB
print("mass of A = ", massA)
print("mass of B = ", massB)
print("init velocity of A = ", vinitA)
print("init velocity of B = ", vinitB)
print("final velocity of A = ", vfinalA)
print("final velocity of B = ", vfinalB)
"""


def get_eyes_pos(robot_x, robot_y, robot_angle, eye_angle, eye_dist):
    """This inputs the position and angle of direction of the robot, as well as the angle between the front of the robot
    and each of the eyes and the distance from center to eye.  The output is a four-tuple giving:
    1   the x pos. of the left eye
    2   the y pos. of the left eye
    3   the x pos. of the right eye
    4   the y pos. of the right eye"""
    left_eye_global_angle = robot_angle + eye_angle  # This computes the angle that the line from center to left eye makes
    # with the x axis of the box
    right_eye_global_angle = robot_angle - eye_angle
    left_delta_x = eye_dist * np.cos(left_eye_global_angle)
    left_delta_y = eye_dist * np.sin(left_eye_global_angle)
    right_delta_x = eye_dist * np.cos(right_eye_global_angle)
    right_delta_y = eye_dist * np.sin(right_eye_global_angle)
    return (robot_x + left_delta_x, robot_y + left_delta_y, robot_x + right_delta_x, robot_y + right_delta_y)


def get_motor_acceleration(robot_angle, robot_mass, left_friction_x, left_friction_y, right_friction_x,
                           right_friction_y,
                           left_propel_force, right_propel_force, motor_dist):
    """Inputs are robot angle of direction, robot mass, x and y components of the friction forces acting on the left
    and right motors, forward propelling force of left and right motors, and distance from center to motor.
    Very important note: Exactly (1/2) of the robot_mass is considered to be held in the left motor, and half in the right.
    Output is the linear acceleration of the robot and the angular acceleration of the robot, due to friction forces on
    either of the motors and due to propelling force applied by the robot"""
    rotate_dir_left = (np.cos(robot_angle + np.pi), np.sin(robot_angle + np.pi))
    rotate_dir_right = (np.cos(robot_angle), np.sin(robot_angle))
    force_x = left_friction_x + right_friction_x + \
              left_propel_force * np.cos(robot_angle) + \
              right_propel_force * np.cos(robot_angle)
    force_y = left_friction_y + right_friction_y + \
              left_propel_force * np.sin(robot_angle) + \
              right_propel_force * np.sin(robot_angle)
    accel_x = force_x / robot_mass
    accel_y = force_y / robot_mass

    F_left = np.dot(rotate_dir_left, (left_friction_x, left_friction_y)) - left_propel_force
    F_right = np.dot(rotate_dir_right, (right_friction_x, right_friction_y)) + right_propel_force
    accel_angular = (F_left + F_right) / (motor_dist * robot_mass)
    return (accel_x, accel_y, accel_angular)


def get_friction_forces(v_x, v_y, v_angular, motor_dist, friction_constant, robot_angle):
    motor_linear_speed_local = v_angular * motor_dist  # The speed of the motors if only rotation is considered
    v_right_local = (motor_linear_speed_local * np.cos(robot_angle), motor_linear_speed_local * np.sin(robot_angle))
    # The velocity vector of the right motor considering only rotation
    v_left_local = (-v_right_local[0], -v_right_local[1])
    v_right_global = (v_right_local[0] + v_x, v_right_local[1] + v_y)
    v_left_global = (v_left_local[0] + v_x, v_left_local[1] + v_y)
    left_friction_x = -friction_constant * v_left_global[0]  # Friction force is linearly related to velocity with
    # respect to the habitat
    left_friction_y = -friction_constant * v_left_global[1]
    right_friction_x = -friction_constant * v_right_global[0]
    right_friction_y = -friction_constant * v_right_global[1]
    return (left_friction_x, left_friction_y, right_friction_x, right_friction_y)


def get_offspring_genotype(father_genotype, mother_genotype):
    """The inputs are numpy arrays which are slices of the G genotype array.  If the father is robot m and the mother
    is robot n, then the father_genotype is G[m, :, :, :] and the mother_genotype is G[n, :, :, :].  The output is
    the offspring's genotype, as calculated by mitosis and independent assortment of all genes.  Note: the shape of
    both parental arrays must be the same."""
    offspring_genotype = np.zeros((father_genotype.shape[0], father_genotype.shape[1], 2), bool)
    for s in range(father_genotype.shape[0]):  # iterate over the supergenes
        for g in range(father_genotype.shape[1]):  # iterate over genes
            first_random_number = np.random.randint(0, 2)  # Generate either a zero or a one
            second_random_number = np.random.randint(0, 2)
            offspring_genotype[s, g, 0] = father_genotype[s, g, first_random_number]  # From the diploid father, the
            # offspring gets either the paternal or maternal chromosome
            offspring_genotype[s, g, 1] = mother_genotype[s, g, second_random_number]
    return (offspring_genotype)


def create_random_genotype(number_of_supergenes, number_of_genes_per_supergene):
    random_genotype = np.random.choice(a=(0, 1), size=(number_of_supergenes, number_of_genes_per_supergene, 2))
    return (random_genotype)


def find_random_space(dim_box, new_robot_index, robot_positions, new_radius, robot_radii, alives):
    """

    :param dim_box:
    :param new_robot_index: the index of the robot that a new spot needs to be found for
    :param robot_positions: a slice of pos for a particular time. It should have a dimension for robot index and a
    dimension for center x, y, etc.
    :param new_radius: the radius of the new robot that a place is being found for
    :param robot_radii: all the other robot radii for that particular time.  Should have one dimension for robot index
    :param alives: all the indices of the robots that are alive at that particular time
    :return: new spot is a numpy array with two entries, an x and y
    """
    # TODO: add an error message in case we cant find a spot
    k = new_robot_index
    empty_space = 2  # When I set empty_space = 2 that means we don't know whether the chosen position for circle k is
    # overlapping with another circle
    new_spot = np.zeros(2, dtype=np.float)
    while empty_space != 1:  # loop until we know that the chosen spot is good
        new_spot[0] = (np.random.rand()) * (dim_box - (2 * new_radius)) + new_radius
        # Sets the initial x component of the position vector for each circle for time = 0, making sure not to place
        # it overlapping with the walls of the box.
        new_spot[1] = (np.random.rand()) * (dim_box - (2 * new_radius)) + new_radius
        empty_space = 2
        for j in alives:  # look at all the other circles to test for overlap
            if k != j and AABB_collision_test(R1_x=robot_positions[j, 0], R1_y=robot_positions[j, 1],
                                              radius1=robot_radii[j], R2_x=new_spot[0], R2_y=new_spot[1],
                                              radius2=new_radius):
                if dist((robot_positions[j, 0], robot_positions[j, 1]),
                        (new_spot[0], new_spot[1])) <= new_radius + robot_radii[j]:  # if circle k
                    # is overlapping circle j
                    empty_space = 0  # 0 means that the chosen new_spot is not possible due to overlap
                    break
        if empty_space == 2:
            empty_space = 1  # if we got through all the previous circles and didn't find overlap, we know that the
            # chosen spot for k is good
    return (new_spot)


def find_space_close_by(dim_box, new_robot_index, new_radius, robot_positions, robot_radii,
                        search_start_box_edge, alives, pos_to_be_born_near):
    k = new_robot_index
    empty_space = 1  # When I set empty_space = 2 that means we don't know whether the chosen position for circle k is
    # overlapping with another circle
    s_box = search_start_box_edge

    while s_box < 2 * dim_box:
        lower_bound_x = max(pos_to_be_born_near[0] - (s_box / 2), 0) + new_radius  # this sets upper and lower bounds for
        # where the robot position should be, based on that the circle of the new robot should be within the habitat
        # and also within the search box (around the mother robot)
        upper_bound_x = min(pos_to_be_born_near[0] + (s_box / 2), dim_box) - new_radius
        lower_bound_y = max(pos_to_be_born_near[1] - (s_box / 2), 0) + new_radius
        upper_bound_y = min(pos_to_be_born_near[1] + (s_box / 2), dim_box) - new_radius

        tries = 0
        new_spot = np.zeros(2, dtype=np.float16)
        while tries < 10:  # loop and try many spots in a given box
            new_spot[0] = (np.random.rand()) * (upper_bound_x - lower_bound_x) + lower_bound_x
            new_spot[1] = (np.random.rand()) * (upper_bound_y - lower_bound_y) + lower_bound_y
            empty_space = 1
            for j in alives:
                if k != j and AABB_collision_test(new_spot[0], new_spot[1], new_radius,
                                                  robot_positions[j, 0], robot_positions[j, 1], robot_radii[j]) == True:
                    if dist((robot_positions[j, 0], robot_positions[j, 1]),
                            (new_spot[0], new_spot[1])) <= new_radius + robot_radii[j]:  # if circle k is
                        # overlapping circle j
                        empty_space = 0  # 0 means that the chosen position for k is not possible due to overlap
                        break
            if empty_space == 1:
                # if we got through all the previous circles and didn't find overlap, we know that the
                # chosen spot for k is good
                break
            tries += 1
        if empty_space == 1:
            break
        s_box += 1
    if s_box >= 2 * dim_box and empty_space == 0:
        print("error: could not find spot for new robot ", k)
        return ("error")
    else:
        print("found spot ", new_spot, "Took ", 10 * (s_box - search_start_box_edge) + tries,
              "tries")
    return (new_spot)


def genotype_to_phenotype(genotype, phenotype_bounds):
    """

    :param genotype: must be a three dimensional array, with dimensions for supergene, gene, and maternal/paternal
    :param phenotype_bounds:
    :return: a one dimensional array for the phenotype of each supergene
    """
    pb = phenotype_bounds
    averaged_genotype = np.zeros((8), float)
    for s in range(genotype.shape[0]):
        averaged_genotype[s] = np.average(genotype[s])  # a vector of numbers corresponding to the supergenes
    phenotype = np.zeros((8), float)
    phenotype[0] = (averaged_genotype[0] * (pb[0, 1] - pb[0, 0])) + pb[0, 0]  # Set the radius.
    phenotype[1] = (averaged_genotype[1] * (pb[1, 1] - pb[1, 0])) + pb[1, 0]  # Set the mass.
    phenotype[2] = ((averaged_genotype[2] * (pb[2, 1] - pb[2, 0])) + pb[2, 0]) * phenotype[0]  # Set motorDist =
    # (number ((between 0 and 1) * bounds_adjustment + bounds_adjustment)*radius
    phenotype[3] = (averaged_genotype[3] * (pb[3, 1] - pb[3, 0])) + pb[3, 0]  # Set eyeAngle.
    phenotype[4] = ((averaged_genotype[4] * (pb[4, 1] - pb[4, 0])) + pb[4, 0]) * phenotype[
        0]  # eyeDist = (number between 0 and 1) * radius
    for i in range(5, 8, 1):  # This sets a random rgb tuple for the color.  Color is given by supergenes 5, 6, and 7
        phenotype[i] = averaged_genotype[i] * (pb[i, 1] - pb[i, 0]) + pb[i, 0]
    return (phenotype)


"""
def robot_born(time, robot_number, father_number, father_genotype, mother_genotype, genotype_array, phenotype_array,
               alive_array, position_array, velocity_array, acceleration_array, collision_array, amount_of_time,
               dim_box, v_mag_init, phenotype_bounds, reproduce_array, search_start_box_edge):
    k = robot_number
    F = father_genotype
    M = mother_genotype
    G = genotype_array
    P = phenotype_array
    alive = alive_array
    R = position_array
    v = velocity_array
    a = acceleration_array
    collision = collision_array

    offspring_genotype = get_offspring_genotype(F, M)
    G = np.insert(arr=G, obj=k, values=offspring_genotype, axis=0)
    resultant_phenotype = genotype_to_phenotype(offspring_genotype, phenotype_bounds)  # Use the genotype of robot k to
    # calculate the phenotype of each supergene characteristic.
    P = np.insert(arr=P, obj=k, values=resultant_phenotype, axis=0)  # Add the robot k phenotype to the
    # array of phenotypes for all robots.

    alive = np.insert(arr=alive, obj=k, values=np.full((amount_of_time), True, dtype=bool), axis=1)
    alive[:time, k] = False  # the offspring is not alive up to and NOT including the current time

    reproduce_array = np.insert(arr=reproduce_array, obj=k, values=np.full((amount_of_time), False, dtype=bool), axis=1)

    R = np.insert(R, obj=k, values=np.full((amount_of_time, 7), 0, dtype=np.float16), axis=1)

    if father_number == "none":  # this is if the robot is being born at time t = 0, so the newborn robot is placed in
        # a random place on the habitat
        R[time, k, 0:2] = find_empty_space(dim_box=dim_box, radii=P[:, 0], alive_robot=alive[time, :],
                                           position_of_robots=R[time, :, 0:2], current_robot_index=k)
    else:
        R[time, k, 0:2] = find_space_close_by(dim_box=dim_box, radii=P[:, 0], alive_robot=alive[time, :],
                                              position_of_robots=R[time, :, 0:2], current_robot_index=k,
                                              parent_robot_index=father_number,
                                              search_start_box_edge=search_start_box_edge)

    R[time, k, 2] = np.random.rand() * 2 * math.pi  # Find a random starting direction for robot k
    R[time, k, 3:7] = get_eyes_pos(robot_x=R[0, k, 0], robot_y=R[0, k, 1], robot_angle=R[0, k, 2],
                                   eye_angle=P[k, 3], eye_dist=P[k, 4])  # Calculate the position of the eyes
    # for robot k at current time

    v = np.insert(v, obj=k, values=np.full((amount_of_time, 3), 0, dtype=np.float16), axis=1)

    # v = np.resize(v, (amountOfTime, np.ma.size(v, axis=1) + 1, 3))

    initVelAngle = np.random.rand() * 2 * np.pi  # This will be the direction of the initial velocity vector.
    v[time, k, 0:2] = polar_to_rect(v_mag_init, initVelAngle)  # Set initial x and y component
    # of the velocity vector for robot k at time = 0
    v[time, k, 2] = 0  # Set the angular velocity to zero radians per second at time t = 0

    a = np.insert(a, obj=k, values=np.full((amount_of_time, 3), 0, dtype=np.float16), axis=1)

    collision = np.insert(collision, obj=k, values=np.full((amount_of_time, 2), -1, dtype=np.int16), axis=1)

    return (G, P, alive, R, v, a, collision, reproduce_array)
"""  # This piece of code is an artifact from when new robots were inserted into position, etc. arrays


def initialize_robot(new_robot_index, genotype_1, genotype_2, phenotype_bounds, robot_positions,
                     robot_radii, alives, dim_box, v_mag_init, init_robot_energy,
                     pos_to_be_born_near='none', search_start_box_edge='none'):
    d = new_robot_index
    offspring_genotype = get_offspring_genotype(genotype_1, genotype_2)
    resultant_phenotype = genotype_to_phenotype(offspring_genotype, phenotype_bounds)

    position = np.zeros(7, dtype=np.float16)

    position[0:2] = find_random_space(dim_box=dim_box, new_radius=resultant_phenotype[0],
                                      new_robot_index=d, robot_radii=robot_radii,
                                      robot_positions=robot_positions, alives=alives)
    """if pos_to_be_born_near == 'none':
        position[0:2] = find_random_space(dim_box=dim_box, new_radius=resultant_phenotype[0],
                                          new_robot_index=d, robot_radii=robot_radii,
                                          robot_positions=robot_positions, alives=alives)
    else:
        position[0:2] = find_space_close_by(dim_box=dim_box, new_radius=resultant_phenotype[0],
                                            search_start_box_edge=search_start_box_edge,
                                            new_robot_index=d, robot_positions=robot_positions,
                                            robot_radii=robot_radii, alives=alives,
                                            pos_to_be_born_near=pos_to_be_born_near)"""

    position[2] = np.random.rand() * 2 * math.pi
    position[3:7] = get_eyes_pos(robot_x=position[0], robot_y=position[1], robot_angle=position[2],
                                 eye_angle=resultant_phenotype[3], eye_dist=resultant_phenotype[4])

    velocity = np.zeros(3, dtype=np.float16)
    init_vel_angle = np.random.rand() * 2 * np.pi  # This will be the direction of the initial velocity vector.
    velocity[0:2] = polar_to_rect(radius=v_mag_init, angle=init_vel_angle)  # Set initial x and y component
    # of the velocity vector for robot k at time = 0
    velocity[2] = 0  # Set the angular velocity to zero radians per second at time t = 0

    return offspring_genotype, resultant_phenotype, position, velocity, init_robot_energy


def evaluate_collisions(k, alives, time_t_phen_c, time_t_pos_c, time_t_vel_c, time_t_accel_c,
                        time_t_collision_c, dim_box, wall_bounce_multiplier, init_number_of_robots):
    tphen_c = time_t_phen_c  # tphen_c represents phen_c at time t, the current phenotypes of alive robots at time t;
    # thus tphen_c has no time dimension
    tpos_c = time_t_pos_c
    tvel_c = time_t_vel_c
    taccel_c = time_t_accel_c
    tcollision_c = time_t_collision_c

    wall = bouncing_off_wall(tpos_c[k, 0], tpos_c[k, 1], tphen_c[k, 0], dim_box, tvel_c[k, 0], tvel_c[k, 1],
                             wall_bounce_multiplier)
    # bouncing_off_wall returns a tuple: return ((accel_x, accel_y, touchingWall, pos_x, pos_y))
    # The purpose of bouncing_off_wall is to adjust acceleration if a robot is touching a wall.  The function also
    # adjusts position to prevent the robots from ever being overlapping with the wall
    taccel_c[k, :] = (0, 0, 0)
    taccel_c[k, 0:2] = wall[0:2]
    tcollision_c[k, 0] = wall[2]  # If touching wall, then collision_c[k, 0] will be 1, otherwise -1.
    tpos_c[k, 0:2] = wall[3:5]  # adjust the  position if the circle is bumping wall

    if tcollision_c[k, 1] == -1:  # if the circle a is not (so far as we know so far) colliding with another circle # TODO: remove this if statement, maybe not necessary?
        for j in alives:  # j is the "other" circle to test for collision with the current circle
            if (k != j):  # if the current circle is not the same as the other circle
                if AABB_collision_test(tpos_c[k, 0], tpos_c[k, 1], tphen_c[k, 0], tpos_c[j, 0], tpos_c[j, 1],
                                       tphen_c[j, 0]) == True:
                    # AABB collision test to see if the bounding boxes of a and j are overlapping.
                    # https://gamedevelopment.tutsplus.com/tutorials/when-worlds-collide-simulating-circle-circle-collisions--gamedev-769

                    center_distance = dist((tpos_c[k, 0], tpos_c[k, 1]),
                                           (tpos_c[j, 0], tpos_c[j, 1]))  # distance between centers of
                    # circles k and j
                    overlap = tphen_c[k, 0] + tphen_c[j, 0] - center_distance  # radii of both circles - centerDistance
                    if overlap >= 0:  # if circle k is overlapping circle j

                        # print("first accel robot ", k, "=", taccel_c[k])
                        # print("first accel robot ", j, "=", taccel_c[j])

                        tcollision_c[k, 1] = j  # this step encodes which robot that robot a is colliding with
                        tcollision_c[j, 1] = k
                        unit_normal = ((tpos_c[j, 0] - tpos_c[k, 0]) / center_distance, (tpos_c[j, 1] - tpos_c[k, 1])
                                       / center_distance)
                        # unit normal vector in direction from center of a to center of j; magnitude of unitNormal = 1
                        tpos_c[k, 0] = tpos_c[k, 0] - unit_normal[0] * 0.5 * overlap  # change x position for robot a
                        tpos_c[k, 1] = tpos_c[k, 1] - unit_normal[1] * 0.5 * overlap
                        tpos_c[j, 0] = tpos_c[j, 0] + unit_normal[0] * 0.5 * overlap
                        tpos_c[j, 1] = tpos_c[j, 1] + unit_normal[1] * 0.5 * overlap
                        # The code above manually changes the positions of circles a and j so that they are not
                        # overlapping.  Credit to https://www.youtube.com/watch?v=LPzyNOHY3A4

                        normal_accel = one_dimensional_elastic(v_1=np.dot(tvel_c[k, 0:2], unit_normal),
                                                               m_1=tphen_c[k, 1],
                                                               v_2=np.dot(tvel_c[j, 0:2], unit_normal),
                                                               m_2=tphen_c[j, 1])
                        # First find the mag. of projection of the velocity vector onto the normal vector for each
                        # circle, then use the function one_dimensional_elastic to compute the acceleration in the
                        # normal direction for each circle.

                        taccel_c[k, 0] = taccel_c[k, 0] + (unit_normal[0] * normal_accel[0])
                        taccel_c[k, 1] = taccel_c[k, 1] + (unit_normal[1] * normal_accel[0])
                        taccel_c[j, 0] = taccel_c[j, 0] + (unit_normal[0] * normal_accel[1])
                        taccel_c[j, 1] = taccel_c[j, 1] + (unit_normal[1] * normal_accel[1])
                        # normal_accel[0] is the magnitude of acceleration for a in the normal direction
                        # normal_accel[1] is the magnitude of acceleration for j in the normal direction
                        # unit_normal[0] is the x component of the normal unit vector
                        # unit_normal[1] is the y component of the normal unit vector

                        # print("last accel robot ", k, "=", taccel_c[k])
                        # print("last accel robot ", j, "=", taccel_c[j])

                        break  # break out of the for loop which iterates over the circles j

    return (tpos_c, tvel_c, taccel_c, tcollision_c)


def motor_acceleration(k, alives, time_t_pos_c, time_t_vel_c, robot_energy, init_robot_energy,
                       time_t_phen_c, min_dist_to_eye_squared, max_dist_to_activate, friction_const,
                       init_number_of_robots, time_t_food_pos, time_t_food_alive, propel_adjust=1):
    tpos_c = time_t_pos_c
    tvel_c = time_t_vel_c
    tphen_c = time_t_phen_c

    l_eye_activ = 0
    r_eye_activ = 0
    l_propel = 0
    r_propel = 0

    energy_color = math.atan(robot_energy / init_robot_energy) / (np.pi / 2)  # A sigmoid type function

    for j in alives:
        if j != k:
            if AABB_collision_test(tpos_c[k, 0], tpos_c[k, 1], max_dist_to_activate,
                                    tpos_c[j, 0], tpos_c[j, 1],
                                    0) == True:  # if robot k and robot j are close enough
                # to interact, that is, you would have to go less than max_dist_to_activate left/right and less than
                # max_dist_to_activate up/down to get from the center of k to the center of j.  Note that this
                # "activation window" is a square.
                l_eye_dist_squared = dist_squared(tpos_c[k, 3:5], tpos_c[j, 0:2])
                r_eye_dist_squared = dist_squared(tpos_c[k, 5:7], tpos_c[j, 0:2])
                if l_eye_dist_squared < min_dist_to_eye_squared:
                    l_eye_dist_squared = min_dist_to_eye_squared
                if r_eye_dist_squared < min_dist_to_eye_squared:
                    r_eye_dist_squared = min_dist_to_eye_squared
                l_eye_activ += energy_color / (l_eye_dist_squared) # when the energy is high, the robot will be more
                # attracted to other robots, but when the energy is low, the robot will be more attracted to food
                r_eye_activ += energy_color / (r_eye_dist_squared)
                # color_diff = ((phenotype_array[k, 5] - phenotype_array[j, 5]))**2  # between 0 and 1

        # Note: because l_eye_activ is a sum, a higher population will lead to much higher activation

    for f in np.array(np.where(time_t_food_alive == True))[0, :]:
        if AABB_collision_test(tpos_c[k, 0], tpos_c[k, 1], max_dist_to_activate,
                               time_t_food_pos[f, 0], time_t_food_pos[f, 1],
                               0) == True:
                l_eye_dist_squared = dist_squared(tpos_c[k, 3:5], time_t_food_pos[f])
                r_eye_dist_squared = dist_squared(tpos_c[k, 5:7], time_t_food_pos[f])
                if l_eye_dist_squared < min_dist_to_eye_squared:
                    l_eye_dist_squared = min_dist_to_eye_squared
                if r_eye_dist_squared < min_dist_to_eye_squared:
                    r_eye_dist_squared = min_dist_to_eye_squared
                l_eye_activ += (1-energy_color) / (l_eye_dist_squared)
                r_eye_activ += (1-energy_color) / (r_eye_dist_squared)

    l_propel = r_eye_activ * propel_adjust
    r_propel = l_eye_activ * propel_adjust

    delta_energy = -(l_propel + r_propel)  # TODO: should improve the physics of using energy.  Should delta_energy
    #                                           also depend on propel_adjust, since that is how actual acceleration is
    #                                           calculated?
    friction = get_friction_forces(v_x=tvel_c[k, 0], v_y=tvel_c[k, 1], v_angular=tvel_c[k, 2],
                                   friction_constant=friction_const, motor_dist=tphen_c[k, 2], robot_angle=tpos_c[k, 2])
    motor_accel = get_motor_acceleration(robot_angle=tpos_c[k, 2],
                                         robot_mass=tphen_c[k, 1],
                                         left_friction_x=friction[0],
                                         left_friction_y=friction[1],
                                         right_friction_x=friction[2],
                                         right_friction_y=friction[3],
                                         left_propel_force=l_propel,
                                         right_propel_force=r_propel,
                                         motor_dist=tphen_c[k, 2])
    return (motor_accel, delta_energy)


def pick_robot_to_die(time_t_bookkeeping_indices, time_t_plus_1_bookkeeping_indices, init_number_of_robots):
    found = False
    while found == False:
        d = np.random.randint(0, init_number_of_robots - 1)
        if time_t_bookkeeping_indices[d] == time_t_plus_1_bookkeeping_indices[d]:  # if the bookkeeping_index of
            # robot d (alive_index) at time t is the same as at time t + 1 (meaning that robot d did not die and be
            # replaced with new robot d
            found = True
    return (d)


def pick_lowest_energy_robot(time_t_energy_c):
    d = np.argmin(time_t_energy_c)
    return (d)


def add_food(food_pos, eaten, time, food_frequency, dim_box, amount_of_time):
    if (np.random.rand() < food_frequency):
        for f in range(1):
            food_pos = np.insert(food_pos, 0, np.full((2), 0), axis=0)
            food_pos[0, :] = (np.random.rand() * dim_box, np.random.rand() * dim_box)
            eaten = np.insert(eaten, 0, np.full((amount_of_time), False), axis=1)
            eaten[:(time), 0] = True  # food begins to be present at time t
    return (food_pos, eaten)


def eat_food(time, food_pos, food_alive, food_energy, food_radius, robot_center_xy, robot_radius):

    delta_food_energy = 0

    for f in np.array(np.where(food_alive[time + 1] == True))[0, :]:
        if AABB_collision_test(R1_x=robot_center_xy[0], R1_y=robot_center_xy[1], R2_x=food_pos[time, f, 0],
                                       R2_y=food_pos[time, f, 1], radius1=robot_radius, radius2=food_radius):
            if dist(robot_center_xy, food_pos[time, f]) < food_radius + robot_radius:
                        food_alive[(time + 1):, f] = False
                        delta_food_energy += food_energy

    return (food_alive, delta_food_energy)