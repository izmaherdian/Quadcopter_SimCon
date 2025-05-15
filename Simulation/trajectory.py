import numpy as np
from numpy import pi
from waypoints import makeWaypoints

class Trajectory:

    def __init__(self, quad, ctrlType, trajSelect, obstacles, obstacle_radii):
        self.ctrlType = ctrlType
        self.xyzType, self.yawType, self.averVel = trajSelect

        t_wps, wps, y_wps, v_wp, wp_ini = makeWaypoints()
        self.t_wps = t_wps
        self.wps  = wps
        self.y_wps = y_wps
        self.v_wp  = v_wp

        self.end_reached = 0
        self.obstacles = obstacles
        self.obstacle_radii = obstacle_radii
        
        self.current_heading = quad.psi

        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.0
        self.sDes = np.hstack((
            self.desPos, self.desVel, self.desAcc,
            self.desThr, self.desEul, self.desPQR,
            self.desYawRate
        )).astype(float)

    def desiredState(self, t, Ts, quad):
        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.0

        def pos_waypoint_arrived_wait():
            dist_threshold = 0.2
            if t == 0:
                self.t_idx = 0
                self.t_arrived = 0
                self.arrived = True
                self.end_reached = 0
            elif not self.end_reached:
                d = np.linalg.norm(self.wps[self.t_idx] - quad.pos)
                if d < dist_threshold and not self.arrived:
                    self.t_arrived = t
                    self.arrived = True
                elif self.arrived and (t - self.t_arrived > self.t_wps[self.t_idx]):
                    self.t_idx += 1
                    self.arrived = False
                    if self.t_idx >= len(self.wps):
                        self.end_reached = 0
                        self.t_idx = 0
            self.desPos = self.wps[self.t_idx]

        def trajectory_pos_coba2(amplitude=1.0, wavelength=10.0, speed=0.5):
            if t == 0:
                self.t_idx = 0
                self.t_arrived = 0
                self.arrived = True
                self.end_reached = 0
            freq = 1.0 / wavelength
            z = amplitude * np.sin(2 * np.pi * freq * t)
            x = speed * t
            y = -3.0 * amplitude * np.sin(2 * np.pi * freq * t)
            self.desPos = np.array([x, y, z])

        def yaw_follow():
            if self.xyzType in (1, 2, 13):
                self.desEul[2] = 0.0 if t == 0 else np.arctan2(
                    self.desPos[1] - quad.pos[1],
                    self.desPos[0] - quad.pos[0]
                )
            elif self.xyzType == 12:
                if t == 0:
                    self.desEul[2] = 0.0
                    self.prevDesYaw = 0.0
                elif not self.arrived:
                    self.desEul[2] = np.arctan2(
                        self.desPos[1] - quad.pos[1],
                        self.desPos[0] - quad.pos[0]
                    )
                    self.prevDesYaw = self.desEul[2]
                else:
                    self.desEul[2] = self.prevDesYaw
            else:
                if t == 0 or t >= self.t_wps[-1]:
                    self.desEul[2] = self.y_wps[self.t_idx]
                else:
                    self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])

            if abs(self.desEul[2] - self.current_heading) >= 2*pi - 0.1:
                self.current_heading += np.sign(self.desEul[2]) * 2*pi
            delta_psi = self.desEul[2] - self.current_heading
            self.desYawRate = delta_psi / Ts
            self.current_heading = self.desEul[2]

        if self.ctrlType == "xyz_pos":
            if self.xyzType != 0:
                if self.xyzType == 13:
                    # pos_waypoint_arrived_wait()
                    trajectory_pos_coba2()
                if self.yawType == 3:
                    yaw_follow()
                self.sDes = np.hstack((
                    self.desPos, self.desVel, self.desAcc,
                    self.desThr, self.desEul, self.desPQR,
                    self.desYawRate
                )).astype(float)

        return self.sDes
