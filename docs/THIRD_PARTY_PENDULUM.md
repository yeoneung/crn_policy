# Pendulum dependency and provenance

The new regulation variant uses the documented equations and reward of Gymnasium Pendulum.
Gymnasium 1.2.3 is a dependency, under the MIT license. See
https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/LICENSE
and https://gymnasium.farama.org/environments/classic_control/pendulum/ .
The native step parity audit records the installed source hash.
The variant changes initialization, torque disturbance, actuator gain, damping and horizon;
it is not an unmodified Pendulum-v1 benchmark result.
No Gymnasium source tree or user-specific paths are redistributed here.
