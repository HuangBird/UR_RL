import numpy as np
import calculations as cal
import vrep
import time
from logger import Logger
import os


class ArmEnv(object):

    state_dim = 28  # State dimension
    action_dim = 3  # Action dimension
    action_bound = 0.001  # Action boundary: -0.001m~0.001m, -0.001rad~0.001rad

    def __init__(self):
        """
        V-REP init session:
        Use legacy remote API
        Prior to run the code, the simulator should be launched first !!!
        """

        print('Program started ...')
        vrep.simxFinish(-1)  # Just in case, close all opened connections
        # Connect to V-REP, get clientID
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Enter server here
        """ 
        The server ID should be consistent with the ID listed in remoteApiConnections.txt, which can be found under
        the V-REP installation folder. 
        """
        vrep.c_Synchronous(self.clientID, True)
        # Confirm connection
        if self.clientID != -1:
            print('Connected to remote API server!')
        else:
            exit('Failed connecting to remote API server!')
        # Start simulation
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

        # Init hole position
        self.hole_position = np.array([-0.6, 0, 0.01])
        self.hole_orientation = np.array([0, 0, 0])
        print("Init hole position:", self.hole_position, self.hole_orientation)

        # Init ur5 joint
        print('Init joint position:', self.get_joint_position())

        # Init force sensor
        print('Init force sensor: ', self.read_force_sensor())

        # Initialize the start position
        self.start_position = np.array([-0.6, 0, 0.05])
        self.start_orientation = np.array([0, 0, 0])
        self.position = self.start_position
        self.orientation = self.start_orientation

        # Initialize the safety factor: safety=1 if safe else safety=-1
        self.safety = np.array([0])  # Safe: safety=0; Unsafe: safety=1

    def get_joint_position(self):
        """ Get robot joint position: {q1, q2, q3, q4, q5, q6} """

        sim_ret, ur5_joint1_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_joint1', vrep.simx_opmode_blocking)
        sim_ret, ur5_joint1_position = vrep.simxGetJointPosition(self.clientID, ur5_joint1_handle,
                                                                 vrep.simx_opmode_blocking)
        sim_ret, ur5_joint2_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_joint2', vrep.simx_opmode_blocking)
        sim_ret, ur5_joint2_position = vrep.simxGetJointPosition(self.clientID, ur5_joint2_handle,
                                                                 vrep.simx_opmode_blocking)
        sim_ret, ur5_joint3_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_joint3', vrep.simx_opmode_blocking)
        sim_ret, ur5_joint3_position = vrep.simxGetJointPosition(self.clientID, ur5_joint3_handle,
                                                                 vrep.simx_opmode_blocking)
        sim_ret, ur5_joint4_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_joint4', vrep.simx_opmode_blocking)
        sim_ret, ur5_joint4_position = vrep.simxGetJointPosition(self.clientID, ur5_joint4_handle,
                                                                 vrep.simx_opmode_blocking)
        sim_ret, ur5_joint5_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_joint5', vrep.simx_opmode_blocking)
        sim_ret, ur5_joint5_position = vrep.simxGetJointPosition(self.clientID, ur5_joint5_handle,
                                                                 vrep.simx_opmode_blocking)
        sim_ret, ur5_joint6_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_joint6', vrep.simx_opmode_blocking)
        sim_ret, ur5_joint6_position = vrep.simxGetJointPosition(self.clientID, ur5_joint6_handle,
                                                                 vrep.simx_opmode_blocking)
        ur5_joint = [ur5_joint1_position, ur5_joint2_position, ur5_joint3_position, ur5_joint4_position,
                     ur5_joint5_position, ur5_joint6_position]

        return ur5_joint

    def read_force_sensor(self):
        """ Read robot force sensor: {Fx, Fy, Fz, Mx, My, Mz} """

        error_code, force_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_connection',
                                                                   vrep.simx_opmode_blocking)
        error_code, force_state, force_vector, torque_vector = \
            vrep.simxReadForceSensor(self.clientID, force_sensor_handle, vrep.simx_opmode_blocking)

        return force_vector + torque_vector

    def get_depth_image(self):
        """
        Get depth image from the robot vision sensor
        The element in buffer array is normalized between 0 to 1.
        """

        error_code, vision_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor_persp',
                                                                    vrep.simx_opmode_blocking)
        error_code, resolution, depth_buffer = \
            vrep.simxGetVisionSensorDepthBuffer(self.clientID, vision_sensor_handle, vrep.simx_opmode_blocking)
        depth_image = np.asarray(depth_buffer)
        depth_image.shape = resolution

        return depth_image

    def move_to(self, tool_position, tool_orientation):
        """ Point to point move """

        sim_ret, ur5_target_handle = vrep.simxGetObjectHandle(self.clientID, 'UR5_target', vrep.simx_opmode_blocking)
        sim_ret, ur5_target_position = vrep.simxGetObjectPosition(self.clientID, ur5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)

        position_direction = np.asarray(
            [tool_position[0] - ur5_target_position[0], tool_position[1] - ur5_target_position[1],
             tool_position[2] - ur5_target_position[2]])
        position_magnitude = np.linalg.norm(position_direction)
        if position_magnitude <= 0.002:
            min_position_step = 0.0001
        else:
            min_position_step = 0.001
        # min_position_step = 0.001
        position_step = min_position_step * (position_direction / position_magnitude)
        num_move_steps = int(np.floor(position_magnitude / min_position_step))

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.clientID, ur5_target_handle, -1, (
                                       ur5_target_position[0] + position_step[0], ur5_target_position[1] + position_step[1],
                                       ur5_target_position[2] + position_step[2]),
                                       vrep.simx_opmode_blocking)
            sim_ret, ur5_target_position = vrep.simxGetObjectPosition(self.clientID, ur5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)

        vrep.simxSetObjectPosition(self.clientID, ur5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.clientID, ur5_target_handle, -1,
                                      (tool_orientation[0], tool_orientation[1], tool_orientation[2]),
                                      vrep.simx_opmode_blocking)

    def step(self, action):
        """ One step """

        # Do action
        self.position[0] += action[0]
        self.position[1] += action[1]
        self.position[2] += action[2]
        # self.orientation[0] += action[3]
        # self.orientation[1] += action[4]
        # self.orientation[2] += action[5]

        self.move_to(self.position, self.orientation)
        # time.sleep(0.5)  # Wait for action to finish

        # State
        s = np.concatenate((self.position, self.orientation, self.get_joint_position(), self.read_force_sensor(),
                            self.hole_position, self.hole_orientation, self.position-self.hole_position, self.safety))

        # Done and reward
        r, done = cal.reward(s)

        # Safety check
        self.safety[0] = cal.safetycheck(s)

        # State
        s = np.concatenate((self.position, self.orientation, self.get_joint_position(), self.read_force_sensor(),
                            self.hole_position, self.hole_orientation, self.position-self.hole_position, self.safety))

        return s, r, done, self.safety[0]

    def start_simulation(self):
        """ Start the simulation engine """

        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)
        vrep.simxFinish(-1)  # Close all communications
        vrep.c_Synchronous(self.clientID, True)
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Restart communication to the server
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)  # Start simulation

    def reset(self):
        """ Reset the environment parameters """

        # Random place the hole
        # new_position = self.init_position.copy()
        # new_orientation = self.init_orientation.copy()
        # new_position[0] += (np.random.rand(1) - 0.5) * 0.002
        # new_position[1] += (np.random.rand(1) - 0.5) * 0.002
        # new_position[2] += 0
        # new_orientation[0] += (np.random.rand(1) - 0.5) * 0.04
        # new_orientation[1] += (np.random.rand(1) - 0.5) * 0.04
        # new_orientation[2] += (np.random.rand(1) - 0.5) * 0.04
        # vrep.simxSetObjectPosition(self.clientID, self.hole_handle, -1, new_position, vrep.simx_opmode_oneshot)
        # vrep.simxSetObjectOrientation(self.clientID, self.hole_handle, -1, new_orientation, vrep.simx_opmode_oneshot)

        # Move to the init position
        self.start_position = np.array([-0.6, 0, 0.05])
        self.start_orientation = np.array([0, 0, 0])
        self.move_to(self.start_position, self.start_orientation)
        
        # Random init position 
        self.position = np.random.uniform(self.start_position-0.008, self.start_position+0.008)
        self.orientation = self.start_orientation
        self.move_to(self.position, self.orientation)
        # time.sleep(0.5)

        self.safety = np.array([0])  # Safe: safety=0; Unsafe: safety=1

        # State
        s = np.concatenate((self.position, self.orientation, self.get_joint_position(), self.read_force_sensor(),
                            self.hole_position, self.hole_orientation, self.position-self.hole_position, self.safety))
        # print(s)
        return s


if __name__ == '__main__':
    env = ArmEnv()
    env.reset()
    # logging_directory = os.path.abspath('logs')
    # logger = Logger(logging_directory)
    # reward = 0.
    # reward_value = []
    for i in range(10):
        # print('Joint position: ', env.get_joint_position())
        # print('Force sensor: ', env.read_force_sensor())
        b = env.get_depth_image()
        print('vision sensor: ', b)
        # env.move_to(np.array([-0.6, -0.15, 0.038+0.01*i]), np.array([0, 0, 0]))
        # reward += 1
        # reward_value.append(reward)
    # logger.save_reward_value(reward_value)
    time.sleep(2)

