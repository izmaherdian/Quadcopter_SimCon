import numpy as np
from numpy import pi
import config

deg2rad = pi/180.0

def makeWaypoints():
    
    v_average = 1.6

    t_ini = 3
    t = np.array(5)
    
    wp_ini = np.array([0, 0, 0])
    wp = np.array([-5, -5, -5])

    yaw_ini = 0    
    yaw = np.array([45])

    t = np.hstack((t_ini, t)).astype(float)
    wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw_ini, yaw)).astype(float)*deg2rad

    return t, wp, yaw, v_average, wp_ini
