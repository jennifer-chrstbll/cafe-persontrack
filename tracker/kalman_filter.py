import numpy as np
import scipy.linalg
from typing import Tuple

class KalmanFilter:
    """
    2D Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state vector:
        x = [cx, cy, aspect_ratio, height, vx, vy, va, vh]^T

    contains bounding box center position (cx, cy), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter transition matrix F
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix H
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard deviations for position & velocity
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray (4,)
            Bounding box coordinates (cx, cy, aspect_ratio, height).

        Returns
        -------
        (mean, covariance)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state mean and covariance for next frame.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state vector with new measurement.
        """
        projected_mean = np.dot(self._update_mat, mean)
        
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)) + np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))

        kalman_gain = np.linalg.multi_dot((
            covariance, self._update_mat.T, scipy.linalg.inv(innovation_cov)
        ))
        
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, self._update_mat, covariance
        ))
        return new_mean, new_covariance
