#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017-2018 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of capture-walking
# <https://github.com/stephane-caron/capture-walking>.
#
# capture-walking is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# capture-walking is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with capture-walking. If not, see <http://www.gnu.org/licenses/>.

import pymanoid

from .double_support import DoubleSupportController
from .one_step import OneStepController
from .swing_foot import SwingFootTracker
from .zero_step import ZeroStepController


class State(object):

    """
    State labels for the finite state machine.
    """

    OneStep = 0
    ZeroStep = 1


class WalkingController(pymanoid.Process):

    """
    Main walking controller.

    Parameters
    ----------
    robot : pymanoid.Robot
        Robot model.
    pendulum : pymanoid.InvertedPendulum
        Inverted pendulum model.
    contact_feed : pymanoid.ContactFeed
        Footstep sequence of the walking scenario.
    nb_steps : int
        Number of spatial discretization steps in capture problems.
    target_height : scalar
        CoM height above target contacts in asymptotic static equilibrium.
    """

    def __init__(self, robot, pendulum, contact_feed, nb_steps, target_height):
        super(WalkingController, self).__init__()
        support_foot = robot.right_foot
        support_contact = contact_feed.pop()
        target_contact = contact_feed.pop()
        one_step = OneStepController(pendulum, nb_steps, target_height)
        one_step.set_contacts(support_contact, target_contact)
        zero_step = ZeroStepController(
            pendulum, nb_steps, target_height, cop_gain=2.)
        zero_step.set_contact(target_contact)
        swing_foot = SwingFootTracker(
            contact_feed.last, target_contact, robot.stance.left_foot)
        self.contact_feed = contact_feed
        self.double_support = True
        self.double_support_brake = None
        self.initial_double_support = True
        self.one_step = one_step
        self.pendulum = pendulum
        self.robot = robot
        self.state = State.OneStep
        self.support_contact = support_contact
        self.support_foot = support_foot
        self.swing_foot = swing_foot
        self.target_contact = target_contact
        self.target_height = target_height
        self.zero_step = zero_step

    def precompute(self):
        """
        Pre-compute QR decompositions for one-step capture problems.
        """
        self.one_step.capture_pb.precompute()

    def compute_zero_step_controls(self):
        """
        Run the zero-step capture state of the FSM.
        """
        zero_step_controls = self.zero_step.compute_controls()
        try:
            one_step_controls = self.one_step.compute_controls()
            if self.one_step.solution.var_cost < 1.:
                self.state = State.OneStep
                self.zero_step.set_contact(self.target_contact)
                return one_step_controls
        except Exception:
            pass  # no transition if no solution
        return zero_step_controls

    def compute_one_step_controls(self):
        """
        Run the one-step capture state of the FSM.
        """
        one_step_controls = self.one_step.compute_controls(
            self.swing_foot.time_to_heel_strike)
        try:
            zero_step_controls = self.zero_step.compute_controls()
            self.zero_step.solution.compute_lambda()
            if self.zero_step.solution.var_cost < 1.:
                self.state = State.ZeroStep
                self.switch_to_next_step()
                return zero_step_controls
        except Exception:
            pass  # no transition if no solution
        return one_step_controls

    def switch_to_next_step(self):
        """
        Switch to next footstep after a successful one-step to zero-step
        transition of the FSM.
        """
        prev_contact = self.support_contact
        support_contact = self.target_contact
        target_contact = self.contact_feed.pop()
        if target_contact is None:  # end of contact sequence
            self.double_support = True
            self.double_support_brake = DoubleSupportController(
                self.pendulum, self.robot.stance, self.target_height)
            self.pendulum.check_cop = False
            return
        if 'ight' in self.swing_foot.foot_target.name:
            swing_foot_target = self.robot.stance.left_foot
            swing_foot_task = self.robot.ik.tasks['LeftFootCenter']
        else:  # current foot target is left foot
            swing_foot_target = self.robot.stance.right_foot
            swing_foot_task = self.robot.ik.tasks['RightFootCenter']
        contact_weight = self.robot.ik.DEFAULT_WEIGHTS['CONTACT']
        self.robot.ik.tasks['LeftFootCenter'].weight = contact_weight
        self.robot.ik.tasks['RightFootCenter'].weight = contact_weight
        self.pendulum.set_contact(support_contact)
        self.double_support = False
        self.one_step.set_contacts(support_contact, target_contact)
        self.support_contact = support_contact
        self.swing_foot.reset(prev_contact, target_contact, swing_foot_target)
        self.swing_foot_task = swing_foot_task
        self.target_contact = target_contact

    def on_tick(self, sim):
        """
        Main function called at each control cycle.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        """
        if not self.double_support:
            self.swing_foot.on_tick(sim)
            # relax foot task weight during swinging
            s = self.swing_foot.progression
            y = (4 * s * (1 - s)) ** 2
            contact_weight = self.robot.ik.DEFAULT_WEIGHTS['CONTACT']
            lowest_weight = contact_weight / 1000
            self.swing_foot_task.weight = (
                y * lowest_weight + (1 - y) * contact_weight)
        if self.double_support_brake is not None:  # end of contact sequence
            cop, lambda_ = self.double_support_brake.compute_controls()
        elif self.state == State.ZeroStep:
            cop, lambda_ = self.compute_zero_step_controls()
        else:  # self.state == State.OneStep
            cop, lambda_ = self.compute_one_step_controls()
        # feed back geometric CoM to the pendulum model
        cutoff_freq = 20  # [Hz]
        x = cutoff_freq * sim.dt
        self.pendulum.com.set_pos(
            x * self.robot.com + (1 - x) * self.pendulum.com.p)
        self.pendulum.set_cop(cop)
        self.pendulum.set_lambda(lambda_)
