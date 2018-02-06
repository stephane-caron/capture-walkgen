.. title:: Manual

######
Manual
######

An implementation for `pymanoid <https://github.com/stephane-caron/pymanoid>`_
of the walking controller described in [Caron18]_.

Capturability of the inverted pendulum
======================================

This framework applies to the *inverted pendulum model* (IPM), a reduced model
for 3D walking whose equation of motion is:

.. math::

    \ddot{\boldsymbol{c}} = \lambda (\boldsymbol{c} - \boldsymbol{r}) +
    \boldsymbol{g}

with :math:`\boldsymbol{c}` the position of the center of mass (CoM) and
:math:`\boldsymbol{g} = - g \boldsymbol{e}_z` the gravity vector. The two
control inputs of the IPM are the location of its center of pressure (CoP)
:math:`\boldsymbol{r}` and its stiffness :math:`\lambda`. Parameters of the IPM
are:

- :math:`g`: the gravitational constant
- :math:`\lambda_\text{min}` and :math:`\lambda_\text{max}`: lower and upper
  bound on the stiffness :math:`\lambda`

This model is implemented in the :class:`pymanoid.InvertedPendulum` class: 

.. code:: python

    from pymanoid.sim import gravity_const as g
    pendulum = InvertedPendulum(
        com, comd, contact=support_contact,
        lambda_min=0.1 * g, lambda_max=2 * g)
    sim.schedule(pendulum)  # integrate IPM dynamics

To make the robot's inverse kinematics track the reduce model, call:

.. code:: python

    robot.setup_ik_for_walking(pendulum.com)
    sim.schedule(robot.ik)  # enable robot IK

where ``robot`` is a :class:`pymanoid.Humanoid` robot model. IPM states (CoM
position ``com`` and velocity ``comd``) will then be sent to the
:class:`pymanoid.IKSolver` inverse kinematics of the robot.

Capture problem
---------------

The gist of capturability analysis is to solve *capture problems* that quantify
the ability to bring the robot to a stop at a desired 3D target.
Mathematically, a capture problem is formalized as:

.. math::

    \text{minimize}_{\boldsymbol{\varphi} \in \mathbb{R}^n}\ 
    & \sum_{j=1}^{n-1} \left[ 
    \frac{\varphi_{j+1}-\varphi_j}{\delta_j} - \frac{\varphi_j - \varphi_{j-1}}
    {\delta_{j-1}} \right]^2
    \\
    \text{subject to}\ 
    &
    \sum_{j=0}^{n-1} \frac{\delta_j}{\sqrt{\varphi_{j+1}} + \sqrt{\varphi_j}} 
    - \frac{\bar{z}_\text{i} \sqrt{\varphi_n} + \dot{\bar{z}}_\text{i}}{g} = 0 \\
    &
    \omega_{\text{i},\text{min}}^2 \leq \varphi_n \leq \omega_{\text{i},\text{max}}^2 \\
    &
    \forall j < n,\ \lambda_\text{min} \delta_j \leq \varphi_{j+1} - \varphi_j \leq
    \lambda_\text{max} \delta_j \\
    &
    \varphi_1 = \delta_0 g / \bar{z}_\text{f}

where the following notations are used:

- :math:`n` is the number of discretization steps
- :math:`\delta_1, \ldots, \delta_n` are spatial discretization steps

As these quantities don't vary between capture problems during walking, they
are set in the constructor of the :class:`capture_walking.CaptureProblem`
class:

.. autoclass:: capture_walking.CaptureProblem

The remaining notations in the capture problem above are:

- :math:`\bar{z}_\text{i}` is the instantaneous CoM height
- :math:`\bar{z}_\text{f}` is the desired CoM height at the end of the capture
  trajectory
- :math:`\omega_\text{min}` and :math:`\omega_\text{max}` are the lower and
  upper bound on IPM damping (representing notably the limits of the CoP area)

These quantities are state-dependent, and can be set via the following setters:

.. automethod:: capture_walking.CaptureProblem.set_init_omega_lim
.. automethod:: capture_walking.CaptureProblem.set_init_zbar
.. automethod:: capture_walking.CaptureProblem.set_init_zbar_deriv
.. automethod:: capture_walking.CaptureProblem.set_target_height

Once a capture problem is fully constructed, you can solve it by calling:

.. automethod:: capture_walking.CaptureProblem.solve

By default, :class:`capture_walking.CaptureProblem` is a thin wrapper used to
call `CPS <https://github.com/jrl-umi3218/CaptureProblemSolver>`_, a tailored
SQP optimization for this precise problem. You can also call the generic solver
`IPOPT <https://projects.coin-or.org/Ipopt>`_ with the function above (requires
`CasADi <https://github.com/casadi/casadi/wiki>`_).

Capture solution
----------------

Solutions found by the solver are stored in a:

.. autoclass:: capture_walking.CaptureSolution

Capture solutions are lazily computed: by default, only the instantaneous IPM
inputs :math:`\lambda_\text{i}`, :math:`\boldsymbol{r}_\text{i}` and
:math:`\omega_\text{i}` are computed. The complete solution (all values of
:math:`\lambda(t)` as well as its switch times :math:`t_j`) is completed by
calling:

.. automethod:: capture_walking.CaptureSolution.compute_lambda
.. automethod:: capture_walking.CaptureSolution.compute_switch_times

From there, all spatial mappings :math:`\lambda(s), \omega(s), t(s)` and time
mappings :math:`\lambda(t), \omega(t), s(t)` can be accessed via:

.. automethod:: capture_walking.CaptureSolution.lambda_from_s
.. automethod:: capture_walking.CaptureSolution.lambda_from_t
.. automethod:: capture_walking.CaptureSolution.omega_from_s
.. automethod:: capture_walking.CaptureSolution.omega_from_t
.. automethod:: capture_walking.CaptureSolution.s_from_t
.. automethod:: capture_walking.CaptureSolution.t_from_s

Walking controller
==================

The ability to solve capture problems is turned into a full-fledged walking
controller by the ``WalkingController`` class:

.. autoclass:: capture_walking.WalkingController

This class is a `pymanoid <https://github.com/stephane-caron/pymanoid>`_
process that you can readily schedule to your simulation:

.. code:: python

    controller = WalkingController(
        robot, pendulum, contact_feed, nb_steps=10, target_com_height=0.8)
    sim.schedule(controller)

The controller follows a Finite State Machine (FSM) with two states: `Zero-step
capture <#zero-step-capture>`_, where the robot balances on its support leg
while swinging for the next footstep, and `One-step capture
<#one-step-capture>`_, where the robot pushes on its support leg toward the
next footstep. See [Caron18]_ for details. When walking is finished, a simple
`Double-support capture <#double-support-capture>`_ strategy is applied to
bring the center of mass (CoM) to a mid-foot location.

Zero-step capture
-----------------

Zero-step capturability is handled by the `ZeroStepController` class:

.. autoclass:: capture_walking.ZeroStepController

The target contact is set independently by calling:

.. automethod:: capture_walking.ZeroStepController.set_contact

The ``pendulum`` reference to the inverted pendulum model is used to update the
CoM state (position and velocity) when computing control inputs:

.. automethod:: capture_walking.ZeroStepController.compute_controls

These two inputs can then be sent to the IPM for zero-step capture.

One-step capture
----------------

One-step capturability is handled by the `OneStepController` class:

.. autoclass:: capture_walking.OneStepController

The support and target contacts are set independently by calling:

.. automethod:: capture_walking.OneStepController.set_contacts

The ``pendulum`` reference to the inverted pendulum model is used to update the
CoM state (position and velocity) when computing control inputs:

.. automethod:: capture_walking.OneStepController.compute_controls

These two inputs can then be sent to the IPM for one-step capture.

Double-support capture
----------------------

At the end of an acyclic contact sequence, a simple double-support strategy is
applied by the `DoubleSupportController` class:

.. autoclass:: capture_walking.DoubleSupportController

Like the two preceding classes, the ``pendulum`` reference to the inverted
pendulum model is used to update the CoM state (position and velocity) when
computing control inputs:

.. automethod:: capture_walking.OneStepController.compute_controls

These two inputs can then be sent to the robot's CoM task directly.

References
==========

.. [Caron18] `Capturability-based Analysis, Optimization and Control of 3D Bipedal Walking <https://hal.archives-ouvertes.fr/hal-01689331/document>`_, S. Caron, A. Escande, L. Lanari and B. Mallein, submitted, January 2018.
