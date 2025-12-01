import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation as R

from util import display_image, detect_cubes
from rasrobot import RASRobot, TIME_STEP

class UR5e(RASRobot):
    def __init__(self):
        """
        This is your main robot class. It inherits from RASRobot for access
        to devices such as motors, position sensors, and camera.
        """
        super().__init__()
        
        # load the kinematic chain based on the robot's URDF file
        end_link = 'wrist_3_link'  # link used for forward and inverse kinematics
        URDF_FN = '../../resources/ur5e_2f85_camera.urdf'
        
        # Read URDF file as a string
        with open(URDF_FN, 'r') as urdf_file:
            urdf_data = urdf_file.read()
        
        self.chain = kp.build_serial_chain_from_urdf(urdf_data, end_link)
        
        # print chain on console
        print('kinematic chain:')
        print(self.chain)
        print(f'The end link of the chain is <{end_link}>.')
        print('All computations of forward and inverse kinematics apply to this link.')
        
        
    @property
    def home_pos(self):
        """ 
        this is the home configuration of the robot 
        """
        return [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]

    def joint_pos(self):
        """
        :return: ndarray, the current joint position of the robot
        """
        joint_pos = np.asarray([m.getPositionSensor().getValue() for m in self.motors])
        return joint_pos
        
    def move_to_joint_pos(self, target_joint_pos, timeout=5, velocity=0.8):
        """
        blocking behaviour, moves the robot to the desired joint position.
        :param target_joint_pos: list/ndarray with joint configuration
        :param timeout: float, timeout in seconds after which this function returns latest
        :param velocity: float, target joint velocity in radians/second
        :return: bool, True if robot reaches the target position
                  else will return False after timeout (in seconds)
        """
        if len(target_joint_pos) != len(self.motors):
            raise ValueError('target joint configuration has unexpected length')
            
        for pos, motor in zip(target_joint_pos, self.motors):
            motor.setPosition(pos)
            if velocity is not None:
                motor.setVelocity(velocity)
            
        # step through simulation until timeout or position reached
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()

            # check if the robot is close enough to the target position
            if all(abs(target_joint_pos - self.joint_pos()) < 0.001):
                return True
                
        print('Timeout. Robot has not reached the desired target position.')
        return False
        
    def forward_kinematics(self, joint_pos=None):
        """
        computes the pose of the chain's end link for given joint position.
        :param joint_pos: joint position for which to compute the end-effector pose
                          if None given, will use the robot's current joint position
        :return: kinpy.Transform object with pos and rot
        """
        if joint_pos is None:
            joint_pos = self.joint_pos()
            
        ee_pose = self.chain.forward_kinematics(joint_pos)
        return ee_pose
        
    def inverse_kinematics(self, target_pose):
        """
        Computes a joint configuration to reach the given target pose.
        Note that the resulting joint position might not actually reach the target
        if the target is e.g. too far away.
        :param target_pose: kinpy.Transform, pose of the end link of the chain
        :return: list/ndarray, joint position
        """
        ik_result = self.chain.inverse_kinematics(target_pose, self.joint_pos())
        return ik_result

def move_to_cube(robot, cube_position):
    """
    Move the robot to align with the cube.
    :param robot: UR5e robot instance
    :param cube_position: tuple (x, y, w, h) of the detected cube
    """
    target_x = 64  # Target x-coordinate (center of the image)
    target_y = 60  # Target y-coordinate (bottom center of the image)
    
    cube_x, cube_y, _, _ = cube_position
    
    # Calculate the offset
    offset_x = target_x - cube_x
    offset_y = target_y - cube_y
    
    # Move the robot in small steps based on the offset
    while abs(offset_x) > 5 or abs(offset_y) > 5:
        current_pos = robot.joint_pos()
        
        # Adjust the position based on the offset
        if abs(offset_x) > 5:
            current_pos[0] += np.sign(offset_x) * 0.01
        if abs(offset_y) > 5:
            current_pos[1] += np.sign(offset_y) * 0.01
        
        robot.move_to_joint_pos(current_pos)
        
        # Update the image and recalculate the offset
        img = robot.get_camera_image()
        cubes = detect_cubes(img)
        if cubes:
            cube_x, cube_y, _, _ = cubes[0]
            offset_x = target_x - cube_x
            offset_y = target_y - cube_y
        else:
            break

    print("Cube aligned with the bottom center of the image.")

def grasp_and_drop(robot, cube_position):
    """
    Grasp the cube and drop it into the tray.
    :param robot: UR5e robot instance
    :param cube_position: tuple (x, y, w, h) of the detected cube
    """
    # Move down in z to grasp the cube
    current_pos = robot.joint_pos()
    current_pos[2] -= 0.1
    robot.move_to_joint_pos(current_pos)
    
    # Close the gripper to grasp the cube
    robot.close_gripper()
    
    # Lift the cube
    current_pos[2] += 0.1
    robot.move_to_joint_pos(current_pos)
    
    # Define the tray position in Cartesian coordinates (adjust based on your setup)
    tray_pose = kp.Transform(
        pos=np.array([0.5, 0.5, 0.2]),  # x, y, z position
        rot=np.array([1, 0, 0, 0])  # w, x, y, z quaternion (no rotation)
    )
    
    # Compute the joint positions to reach the tray position
    tray_joint_pos = robot.inverse_kinematics(tray_pose)
    
    # Move to the tray position
    if tray_joint_pos is not None:
        robot.move_to_joint_pos(tray_joint_pos)
        # Open the gripper to drop the cube
        robot.open_gripper()
        print("Cube dropped into the tray.")
    else:
        print("Failed to compute inverse kinematics for the tray position.")

if __name__ == '__main__':
    # Initialize robot and move to home position
    robot = UR5e()
    robot.move_to_joint_pos(robot.home_pos)
    
    while True:
        # Display the camera image
        img = robot.get_camera_image()
        display_image(img, 'camera view', wait=False)
        
        # Detect cubes
        cubes = detect_cubes(img)
        if cubes:
            cube = cubes[0]  # Take the first detected cube
            
            # Move to the cube
            move_to_cube(robot, cube)
            
            # Grasp and drop the cube into the tray
            grasp_and_drop(robot, cube)
        
        # Add a break condition or loop condition as needed
        break
