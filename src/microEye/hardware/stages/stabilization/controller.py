from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, Union

import numpy as np

from microEye.hardware.stages.stage import Axis


class RejectionMethod(Enum):
    NONE = 'None'
    STANDARD_DEVIATION = 'Standard Deviation'
    MEDIAN_ABSOLUTE_DEVIATION = 'Median Absolute Deviation'

    def __str__(self):
        return self.value


def reject_outliers_sd(
    data: np.ndarray, threshold: float = 2.0, min_valid: int = 3
) -> np.ndarray:
    '''
    Iteratively reject outliers based on standard deviation.
    Returns indices of valid data.
    '''
    valid_idx = np.ones(len(data), dtype=bool)
    last_n_valid = len(data)
    while True:
        shifts = np.nanmean(data[valid_idx], axis=0)
        stds = np.nanstd(data[valid_idx], axis=0)
        outliers = np.abs(data - shifts) / stds > threshold
        if data.ndim == 2:
            outliers = outliers.sum(axis=1, dtype=bool)
        valid_idx = ~outliers
        n_valid = valid_idx.sum()
        if n_valid < min_valid or n_valid == last_n_valid:
            break
        last_n_valid = n_valid
    return valid_idx


def reject_outliers_mad(
    data: np.ndarray, threshold: float = 1.5
) -> np.ndarray:
    '''
    Iteratively reject outliers based on Median Absolute Deviation (MAD).
    Returns filtered data.
    '''
    data_mean = np.nanmean(data, axis=0)
    adifs = np.abs(data - data_mean)
    mad = np.median(adifs, axis=0)
    # Use 0.6745 for consistency with standard deviation
    aMi = 0.6745 * adifs / (mad + 1e-8)
    outliers = aMi > threshold
    if data.ndim == 2:
        outliers = outliers.sum(axis=1, dtype=bool)
    valid_idx = ~outliers
    return valid_idx


def apply_rejection_method(
    data: np.ndarray, method: RejectionMethod, **kwargs
) -> np.ndarray:
    valid_idx = slice(None)
    if method == RejectionMethod.STANDARD_DEVIATION:
        valid_idx = reject_outliers_sd(
            data,
            threshold=kwargs.get('threshold', 2.0),
            min_valid=kwargs.get('min_valid', 3),
        )
    elif method == RejectionMethod.MEDIAN_ABSOLUTE_DEVIATION and len(data) >= 4:
        valid_idx = reject_outliers_mad(
            data,
            threshold=kwargs.get('threshold', 1.5),
        )

    return valid_idx if kwargs.get('return_indices', False) else data[valid_idx]


class BaseController(ABC):
    '''
    Abstract base class for controllers.
    Supports multi-axis error, time-based integration, and flexible parameter setting.
    '''

    def __init__(self, n_axes: int = 3):
        self._n_axes = 3

        self._Kp = np.ones((n_axes,))
        self._Ki = np.zeros((n_axes,))
        self._Kd = np.zeros((n_axes,))
        self._integral = np.zeros((n_axes,))
        self._last_error = np.zeros((n_axes,))
        self._last_time = np.zeros((n_axes,))

        self._rejection_method = RejectionMethod.NONE
        self._outlier_threshold = 2.0
        self._outlier_min_points = 4

    @property
    def rejection_method(self) -> RejectionMethod:
        return self._rejection_method

    @rejection_method.setter
    def rejection_method(self, method: RejectionMethod):
        self._rejection_method = method

    @property
    def outlier_threshold(self) -> float:
        return self._outlier_threshold

    @outlier_threshold.setter
    def outlier_threshold(self, value: float):
        self._outlier_threshold = value

    @property
    def outlier_min_points(self) -> int:
        return self._outlier_min_points

    @outlier_min_points.setter
    def outlier_min_points(self, value: int):
        self._outlier_min_points = value

    def outlier_rejection(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if kwargs.get('threshold') is None:
            kwargs['threshold'] = self._outlier_threshold
        if kwargs.get('min_valid') is None:
            kwargs['min_valid'] = self._outlier_min_points

        return apply_rejection_method(data, self._rejection_method, **kwargs)

    def get_Kp(self, axis: Axis) -> Union[float, np.ndarray]:
        '''
        Get the proportional gain for a specific axis.
        '''
        if axis is None:
            return self._Kp.copy()

        return self._Kp[axis.axis_index()]

    def set_Kp(self, value: float, axis: Axis):
        '''
        Set the proportional gain for a specific axis.
        '''
        self._Kp[axis.axis_index()] = value

    def get_Ki(self, axis: Axis) -> Union[float, np.ndarray]:
        '''
        Get the integral gain for a specific axis.
        '''
        if axis is None:
            return self._Ki.copy()

        return self._Ki[axis.axis_index()]

    def set_Ki(self, value: float, axis: Axis):
        '''
        Set the integral gain for a specific axis.
        '''
        self._Ki[axis.axis_index()] = value

    def get_Kd(self, axis: Axis) -> Union[float, np.ndarray]:
        '''
        Get the derivative gain for a specific axis.
        '''
        if axis is None:
            return self._Kd.copy()

        return self._Kd[axis.axis_index()]

    def set_Kd(self, value: float, axis: Axis):
        '''
        Set the derivative gain for a specific axis.
        '''
        self._Kd[axis.axis_index()] = value

    def set_gains(
        self,
        axis: Optional[Axis],
        Kp: Union[float, tuple] = 1.0,
        Ki: Union[float, tuple] = 0.0,
        Kd: Union[float, tuple] = 0.0,
    ):
        '''
        Set the controller gains for a specific axis.
        '''
        idx = slice(None) if axis is None else axis.axis_index()
        self._Kp[idx] = (
            np.asarray(Kp) if isinstance(Kp, (float, int)) else np.asarray(Kp)[idx]
        )
        self._Ki[idx] = (
            np.asarray(Ki) if isinstance(Ki, (float, int)) else np.asarray(Ki)[idx]
        )
        self._Kd[idx] = (
            np.asarray(Kd) if isinstance(Kd, (float, int)) else np.asarray(Kd)[idx]
        )

    def reset(self, axes: list[Axis] = None):
        '''
        Reset the controller state for a specific axis.
        '''
        idx = (
            [a.axis_index() for a in axes]
            if axes is not None and len(axes) > 0
            else slice(None)
        )
        self._integral[idx] = 0.0
        self._last_time[idx] = 0.0

    @abstractmethod
    def response(
        self,
        t: float,
        x_shifts: Union[float, np.ndarray],
        y_shifts: Union[float, np.ndarray],
        z_shifts: float,
        **kwargs,
    ) -> np.ndarray:
        '''
        Calculate the controller response for the given error at time t.
        Must be implemented by subclasses.
        '''
        pass

class PIDController(BaseController):
    '''
    Multi-axis PID controller with time-based integration and differentiation.
    '''

    def __init__(
        self,
        Kp: Union[float, tuple] = 1.0,
        Ki: Union[float, tuple] = 0.0,
        Kd: Union[float, tuple] = 0.0,
    ):
        super().__init__(n_axes=3)

        self.set_gains(None, Kp, Ki, Kd)

    def response(
        self,
        t: float,
        x_shifts: Union[float, np.ndarray] = 0.0,
        y_shifts: Union[float, np.ndarray] = 0.0,
        z_shift: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(x_shifts, np.ndarray):
            x_shifts = np.nanmean(self.outlier_rejection(x_shifts))
        if isinstance(y_shifts, np.ndarray):
            y_shifts = np.nanmean(self.outlier_rejection(y_shifts))

        delta = np.array([x_shifts, y_shifts, z_shift], dtype=np.float64)

        # Initialize last_time if first call
        self._last_time[self._last_time <= 0.0] = t
        delta_t = t - self._last_time
        delta_t = np.clip(delta_t, 1e-3, 1.0)  # limit max delta_t to avoid spikes

        # Integral term
        self._integral += delta * delta_t

        # Derivative term
        derivative = (delta - self._last_error) / delta_t

        # PID output
        output = (
            self._Kp * delta + self._Ki * self._integral + self._Kd * derivative
        )

        # Update buffers
        self._last_error = delta
        self._last_time[:] = t

        return -output
