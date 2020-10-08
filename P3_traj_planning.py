import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        t_switch = self.traj_controller.traj_times[-1] - self.t_before_switch
        if (t < t_switch):
            return self.traj_controller.compute_control(x, y, th, t)
        return self.pose_controller.compute_control(x, y, th, t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########

    t = np.zeros(len(path)) # nominal times that we reach each point
    for i in range(1, len(path)): # skip t[0] which we know is 0
        curr_pt = np.array(path[i])
        prev_pt = np.array(path[i-1])
        dist_traveled = np.linalg.norm(curr_pt - prev_pt) # straight line distance
        delta_t = dist_traveled / V_des # distance / velocity = time
        t[i] = t[i-1] + delta_t
    
    x = np.array(path)[:,0]
    y = np.array(path)[:,1]
    t_smoothed = np.arange(t[0], t[-1], dt)

    # interpolate x and y as functions of (nominal) time
    tck_x = scipy.interpolate.splrep(t, x, s=alpha)
    tck_y = scipy.interpolate.splrep(t, y, s=alpha)

    # create traj_smoothed array
    traj_smoothed = np.zeros([len(t_smoothed),7])
    traj_smoothed[:,0] = scipy.interpolate.splev(t_smoothed, tck_x)         # x
    traj_smoothed[:,1] = scipy.interpolate.splev(t_smoothed, tck_y)         # y
    traj_smoothed[:,3] = scipy.interpolate.splev(t_smoothed, tck_x, der=1)  # xd 
    traj_smoothed[:,4] = scipy.interpolate.splev(t_smoothed, tck_y, der=1)  # yd
    traj_smoothed[:,2] = np.arctan2(traj_smoothed[:,4], traj_smoothed[:,3]) # th = arctan(yd/xd)
    traj_smoothed[:,5] = scipy.interpolate.splev(t_smoothed, tck_x, der=2)  # xdd
    traj_smoothed[:,6] = scipy.interpolate.splev(t_smoothed, tck_y, der=2)  # ydd

    ########## Code ends here ##########

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    # compute initial controls V and om
    V, om = compute_controls(traj)
    # rescale V to meet control constraints
    V_tilde = rescale_V(V, om, V_max, om_max) 

    # compute arc length along the trajectory
    s = compute_arc_length(V, t)
    # compute the new time history
    tau = compute_tau(V_tilde, s)
    # rescale om to meet control contraints
    om_tilde = rescale_om(V, om, V_tilde)

    # final state
    s_f = State(x=traj[-1,0],y=traj[-1,1],V=V_tilde[-1],th=traj[-1,2])
    # interpolate the trajectory
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
