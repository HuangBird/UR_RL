import numpy as np


# def tanh(x):
#     return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def reward(s):
    """ The real time reward (r(t)) from the environment """

    lambda1 = 1000
    # lambda2 = 1
    lambda3 = 0.1
    lambda4 = 0.1
    lambda5 = 0.1
    hole_position = np.array([-0.6, 0, 0.01])  # The coordinates of the hole bottom
    # hole_orientation = np.array([0, 0, 0])
    peg_force = np.array([0, 0, 0.9616])  # Gravity compensation
    peg_torque = np.array([0, 0, 0])
    safety = s[-1]
    p_r = np.array([s[0]-hole_position[0], s[1]-hole_position[1], s[2]-hole_position[2]])
    p_r_xy = np.array([s[0]-hole_position[0], s[1]-hole_position[1]])
    # o_r = np.array([s[3]-hole_orientation[0], s[4]-hole_orientation[1], s[5]-hole_orientation[2]])
    f_r = np.array([s[12]-peg_force[0], s[13]-peg_force[1], s[14]-peg_force[2]])
    m_r = np.array([s[15]-peg_torque[0], s[16]-peg_torque[1], s[17]-peg_torque[2]])
    r = -(lambda1*(np.linalg.norm(p_r) + 10*np.linalg.norm(p_r_xy)) + lambda3*np.linalg.norm(f_r) +
          lambda4*np.linalg.norm(m_r) + lambda5*safety)

    if np.linalg.norm(p_r) <= 0.002:
        done = True
    else:
        done = False

    return r, done


def safetycheck(s):
    """ This function checks if the force and position extends safety value """

    force_safe_bound = 45  # 5kg*10N/kg=50N
    position_safe_bound = 0.2  # m
    hole_position = np.array([-0.6, 0, 0])  # Top surface center of the hole, need to handcraft
    d = np.array([s[0]-hole_position[0], s[1]-hole_position[1], s[2]-hole_position[2]])
    d_xyz = np.linalg.norm(d)
    d_z = d[2]
    f_x = s[12]
    f_y = s[13]
    f_z = s[14]
    if abs(f_x) >= force_safe_bound or abs(f_y) >= force_safe_bound or abs(f_z) >= force_safe_bound:
        return 1  # Unsafe
    elif d_xyz >= position_safe_bound or d_z < 0.01:
        return 1  # Unsafe
    else:
        return 0  # Safe

