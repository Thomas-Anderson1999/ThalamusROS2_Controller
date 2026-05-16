"""
Python port of:
  ThalamusCommonController/KalmanFilter_Attitude/KalmanFilter_AHRS.cpp
  ThalamusCommonController/IMUProcess.cpp
"""
import numpy as np


class KalmanAHRS:
    INIT_CNT = 500  # IMUProcess.cpp: INIT_IMUCNT

    def __init__(self, dt=0.005, Q_val=0.01, R_val=10.0):
        # AHRSInit: InitKF(0.005, 0.01, 10.)
        self.dt = dt
        self.H  = np.eye(4)
        self.Q  = np.eye(4) * Q_val
        self.R  = np.eye(4) * R_val

        self.x  = np.array([1.0, 0.0, 0.0, 0.0])
        self.xp = self.x.copy()
        self.P  = np.eye(4)
        self.Pp = self.P.copy()

        self.Gx_bias = self.Gy_bias = self.Gz_bias = 0.0
        self.one_g   = 1.0

        self.roll = self.pitch = self.yaw = 0.0

        # bias collection
        self._sum_gx = self._sum_gy = self._sum_gz = 0.0
        self._sum_ax = self._sum_ay = self._sum_az = 0.0
        self._cnt_g  = self._cnt_a  = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Helper conversions  (KalmanFilter_AHRS.cpp lines 3-56)
    # ------------------------------------------------------------------
    @staticmethod
    def _euler_accel(ax, ay):
        ax = np.clip(ax, -1.0, 1.0)
        ay = np.clip(ay, -1.0, 1.0)
        theta = np.arcsin(ax)
        temp  = np.clip(-ay / np.cos(theta), -1.0, 1.0)
        phi   = np.arcsin(temp)
        return phi, theta

    @staticmethod
    def _euler_to_quat(phi, theta, psi):
        sp, cp = np.sin(phi   / 2), np.cos(phi   / 2)
        st, ct = np.sin(theta / 2), np.cos(theta / 2)
        ss, cs = np.sin(psi   / 2), np.cos(psi   / 2)
        return np.array([
            cp*ct*cs + sp*st*ss,
            sp*ct*cs - cp*st*ss,
            cp*st*cs + sp*ct*ss,
            cp*ct*ss - sp*st*cs,
        ])

    @staticmethod
    def _quat_to_euler(q):
        q0, q1, q2, q3 = q
        phi   = np.arctan2(2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2))
        theta = -np.arcsin(np.clip(2*(q1*q3 - q0*q2), -1.0, 1.0))
        psi   = np.arctan2(2*(q1*q2 + q0*q3), 1 - 2*(q2*q2 + q3*q3))
        return phi, theta, psi

    # ------------------------------------------------------------------
    # KF steps
    # ------------------------------------------------------------------
    def _predict(self, p, q, r):
        """KF_AHRS_Predict — gyro input (rad/s), bias already removed"""
        A = np.eye(4) + 0.5 * self.dt * np.array([
            [ 0, -p, -q, -r],
            [ p,  0,  r, -q],
            [ q, -r,  0,  p],
            [ r,  q, -p,  0],
        ])
        self.xp = A @ self.x
        self.Pp = A @ self.P @ A.T + self.Q
        self.x  = self.xp   # gyro dominates; update x immediately
        self.roll, self.pitch, self.yaw = self._quat_to_euler(self.x)

    def _measure(self, ax, ay, az):
        """KF_AHRS_Mesure — accel input (normalised to 1g)"""
        ax /= self.one_g
        ay /= self.one_g
        # az not used in EulerAccel

        phi_a, theta_a = self._euler_accel(ax, ay)
        z = self._euler_to_quat(phi_a, theta_a, self.yaw)

        K      = self.Pp @ self.H.T @ np.linalg.inv(self.H @ self.Pp @ self.H.T + self.R)
        self.x = self.xp + K @ (z - self.H @ self.xp)
        self.P = self.Pp - K @ self.H @ self.Pp
        self.roll, self.pitch, self.yaw = self._quat_to_euler(self.x)

    # ------------------------------------------------------------------
    # Public interface — axis mapping mirrors IMUProcess.cpp
    #   Predict: (imu.z, imu.x, imu.y)
    #   Measure: (imu.z, imu.x, imu.y)
    #   BiasInit gyro order: (avg_gz, avg_gx, avg_gy)
    #   BiasInit accel order: (avg_az, avg_ax, avg_ay)
    # ------------------------------------------------------------------
    def push_gyro(self, gx, gy, gz):
        if not self._initialized:
            if self._cnt_g < self.INIT_CNT:
                self._sum_gx += gx
                self._sum_gy += gy
                self._sum_gz += gz
                self._cnt_g  += 1
            return

        p = gz - self.Gx_bias
        q = gx - self.Gy_bias
        r = gy - self.Gz_bias
        self._predict(p, q, r)

    def push_accel(self, ax, ay, az):
        if not self._initialized:
            if self._cnt_a < self.INIT_CNT:
                self._sum_ax += ax
                self._sum_ay += ay
                self._sum_az += az
                self._cnt_a  += 1

            if self._cnt_g >= self.INIT_CNT and self._cnt_a >= self.INIT_CNT:
                # SensorBiasInitialize(Gz_avg, Gx_avg, Gy_avg, Az_avg, Ax_avg, Ay_avg)
                self.Gx_bias = self._sum_gz / self._cnt_g
                self.Gy_bias = self._sum_gx / self._cnt_g
                self.Gz_bias = self._sum_gy / self._cnt_g
                raw_ax = self._sum_ax / self._cnt_a
                raw_ay = self._sum_ay / self._cnt_a
                raw_az = self._sum_az / self._cnt_a
                # one_g: sqrt(Ax^2+Ay^2+Az^2) using bias-order args (Az, Ax, Ay)
                self.one_g = np.sqrt(raw_az**2 + raw_ax**2 + raw_ay**2)
                if self.one_g < 1e-6:
                    self.one_g = 9.8
                self._initialized = True
            return

        # axis mapping: (imu.z, imu.x, imu.y)
        self._measure(az, ax, ay)

    def get_attitude_deg(self):
        """Returns (roll_deg, pitch_deg, yaw_deg)"""
        return (
            np.degrees(self.roll),
            np.degrees(self.pitch),
            np.degrees(self.yaw),
        )

    @property
    def is_ready(self):
        return self._initialized
