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

import IPython
import os
import sys

try:  # use local pymanoid submodule
    script_path = os.path.realpath(__file__)
    sys.path = [os.path.dirname(script_path) + '/pymanoid'] + sys.path
    import pymanoid
except:  # avoid warning E402 from Pylint :p
    pass

from numpy import dot, sqrt
from pymanoid import PointMass, Stance
from pymanoid.contact import ContactFeed
from pymanoid.gui import draw_line, draw_point
from pymanoid.gui import TrajectoryDrawer
from pymanoid.models import InvertedPendulum
from pymanoid.sim import gravity_const

from capture_walking import DoubleSupportController
from capture_walking import OneStepController
from capture_walking import SwingFootTracker
from capture_walking import ZeroStepController


# Walking control parameters
LAMBDA_MAX = 2.0 * gravity_const
LAMBDA_MIN = 0.1 * gravity_const
NB_MPC_STEPS = 10
TARGET_COM_HEIGHT = 0.8

# Inverse kinematics parameters
CONTACT_IK_WEIGHT = 1.
SWING_FOOT_IK_WEIGHT = 1e-3

# COMANOID environment model (not distributed outside of the project)
COMANOID_MODEL = 'comanoid/model/ComanoidStructure.xml'
COMANOID_MODEL_FOUND = os.path.isfile(COMANOID_MODEL)


class WalkingController(pymanoid.Process):

    class State(object):

        OneStep = 0
        ZeroStep = 1

    def __init__(self, pendulum, contact_feed):
        super(WalkingController, self).__init__()
        support_foot = robot.right_foot
        support_contact = contact_feed.pop()
        target_contact = contact_feed.pop()
        one_step = OneStepController(
            pendulum, NB_MPC_STEPS, TARGET_COM_HEIGHT)
        one_step.set_contacts(support_contact, target_contact)
        zero_step = ZeroStepController(
            pendulum, NB_MPC_STEPS, TARGET_COM_HEIGHT, cop_gain=2.)
        zero_step.set_contact(target_contact)
        swing_foot = SwingFootTracker(
            contact_feed.last, target_contact, stance.left_foot)
        if "--ipopt" not in sys.argv and "--no-pre" not in sys.argv:
            one_step.capture_pb.cps_precompute()
        self.contact_feed = contact_feed
        self.double_support = True
        self.double_support_brake = None
        self.initial_double_support = True
        self.one_step = one_step
        self.state = self.State.OneStep
        self.support_contact = support_contact
        self.support_foot = support_foot
        self.swing_foot = swing_foot
        self.target_contact = target_contact
        self.verbose = False
        self.zero_step = zero_step

    def compute_zero_step_controls(self):
        zero_step_controls = self.zero_step.compute_controls()
        try:
            one_step_controls = self.one_step.compute_controls()
            if self.one_step.solution.var_cost < 1.:
                if self.verbose:
                    pymanoid.info(
                        "FSM: transition to 'One-step capture' with cost %f" %
                        self.one_step.solution.var_cost)
                self.state = self.State.OneStep
                self.zero_step.set_contact(self.target_contact)
                return one_step_controls
        except Exception as exn:
            if self.verbose:
                pymanoid.info(
                    "FSM: one-step transition not ready: %s" % str(exn))
        return zero_step_controls

    def switch_to_next_step(self):
        prev_contact = self.support_contact
        support_contact = self.target_contact
        target_contact = self.contact_feed.pop()
        if target_contact is None:  # end of contact sequence
            self.double_support = True
            self.double_support_brake = DoubleSupportController(
                pendulum, stance, self.target_height)
            pendulum.check_cop = False
            return
        if 'ight' in self.swing_foot.foot_target.name:
            swing_foot_target = stance.left_foot
            swing_foot_task = robot.ik.tasks['LeftFootCenter']
        else:  # current foot target is left foot
            swing_foot_target = stance.right_foot
            swing_foot_task = robot.ik.tasks['RightFootCenter']
        robot.ik.tasks['LeftFootCenter'].weight = CONTACT_IK_WEIGHT
        robot.ik.tasks['RightFootCenter'].weight = CONTACT_IK_WEIGHT
        pendulum.set_contact(support_contact)
        self.double_support = False
        self.one_step.set_contacts(support_contact, target_contact)
        self.support_contact = support_contact
        self.swing_foot.reset(prev_contact, target_contact, swing_foot_target)
        self.swing_foot_task = swing_foot_task
        self.target_contact = target_contact

    def compute_one_step_controls(self):
        one_step_controls = self.one_step.compute_controls(
            self.swing_foot.time_to_heel_strike)
        try:
            zero_step_controls = self.zero_step.compute_controls()
            self.zero_step.solution.compute_lambda()
            if self.zero_step.solution.var_cost < 1.:
                if self.verbose:
                    pymanoid.info(
                        "FSM: transition to 'Zero-step capture' with cost %f" %
                        self.zero_step.solution.var_cost)
                self.state = self.State.ZeroStep
                self.switch_to_next_step()
                return zero_step_controls
        except Exception as exn:
            if self.verbose:
                pymanoid.info(
                    "FSM: zero-step transition not ready: %s" % str(exn))
        return one_step_controls

    def on_tick(self, sim):
        if not self.double_support:
            self.swing_foot.on_tick(sim)
            # relax foot task weight during swinging
            s = self.swing_foot.progression
            y = (4 * s * (1 - s)) ** 2
            self.swing_foot_task.weight = \
                y * SWING_FOOT_IK_WEIGHT + (1 - y) * CONTACT_IK_WEIGHT
        if self.double_support_brake is not None:  # end of contact sequence
            cop, lambda_ = self.double_support_brake.compute_controls()
        elif self.state == self.State.ZeroStep:
            cop, lambda_ = self.compute_zero_step_controls()
        else:  # self.state == self.State.OneStep
            cop, lambda_ = self.compute_one_step_controls()
        # feed back geometric CoM to the pendulum model
        cutoff_freq = 20  # [Hz]
        x = cutoff_freq * sim.dt
        pendulum.com.set_pos(x * robot.com + (1 - x) * pendulum.com.p)
        pendulum.set_cop(cop)
        pendulum.set_lambda(lambda_)


class PreviewTrajectoryDrawer(pymanoid.Process):

    def __init__(self, controller):
        super(PreviewTrajectoryDrawer, self).__init__()
        self.handles = []
        self.one_step = controller.one_step
        self.zero_step = controller.zero_step

    def on_tick(self, sim):
        new_handles = []
        if self.one_step.solution is not None:
            new_handles.extend(self.one_step.draw_solution('r'))
        if "--record" not in sys.argv:
            if self.zero_step.solution is not None:
                new_handles.extend(self.zero_step.draw_solution('g'))
            new_handles.append(draw_line(
                pendulum.com.p, pendulum.com.p + 1.0 * pendulum.com.pd,
                linewidth=2, color='m'))
            target = self.one_step.target_contact.p \
                + [0., 0., TARGET_COM_HEIGHT]
            new_handles.append(draw_point(target, color='b', pointsize=0.01))
        self.handles = new_handles


def print_usage():
    print("Usage: %s [scenario] [--record]" % sys.argv[0])
    print("Scenarios:")
    if COMANOID_MODEL_FOUND:
        print("    --comanoid       COMANOID scenario")
    print("    --elliptic       Elliptic stairase scenario")
    print("    --flat           Flat floor scenario")
    print("    --regular        Regular stairase scenario")
    print("Options:")
    print("    --ipopt          Use IPOPT rather than the cps SQP solver")
    print("    --no-pre         Disable CPS precomputation (uses less memory)")
    print("    --record         Record simulation video")


def load_comanoid():
    assert COMANOID_MODEL_FOUND, "COMANOID model not found"
    sim.load_mesh(COMANOID_MODEL)
    contact_feed = ContactFeed('comanoid/contacts.json')
    for (i, contact) in enumerate(contact_feed.contacts):
        contact.link = robot.right_foot if i % 2 == 0 else robot.left_foot
        if i < 6:  # stair climbing
            contact.takeoff_clearance = 0.15 if i < 2 else 0.2
            contact.takeoff_tangent = (2 * contact.n - contact.t) / sqrt(5)
    sim.move_camera_to([
        [0.45414461, -0.37196997,  0.80956224, -7.18046379],
        [-0.89088689, -0.19832849,  0.40863965, -2.44471264],
        [0.00855758, -0.90680988, -0.42145298,  2.98527932],
        [0.,  0.,  0.,  1.]])
    return contact_feed


def load_elliptic_staircase():
    contact_feed = ContactFeed(
        path='scenarios/elliptic-staircase/contacts.json', cyclic=True)
    for (i, contact) in enumerate(contact_feed.contacts):
        contact.link = robot.right_foot if i % 2 == 0 else robot.left_foot
    sim.move_camera_to([
        [-0.85, -0.23,  0.46, -2.64],
        [-0.52,  0.41, -0.75,  4.49],
        [-0.02, -0.88, -0.48,  4.93],
        [0., 0., 0., 1.]])
    return contact_feed


def load_flat_floor_staircase():
    contact_feed = ContactFeed(
        path='scenarios/flat-floor/contacts.json', cyclic=False)
    for (i, contact) in enumerate(contact_feed.contacts):
        contact.link = robot.right_foot if i % 2 == 0 else robot.left_foot
    sim.move_camera_to([
        [-0.62,  0.09, -0.78, 2.76],
        [+0.79,  0.13, -0.61, 1.86],
        [+0.04, -0.99, -0.15, 1.23],
        [0., 0., 0., 1.]])
    return contact_feed


def load_regular_staircase():
    contact_feed = ContactFeed(
        path='scenarios/regular-staircase/contacts.json')
    for (i, contact) in enumerate(contact_feed.contacts):
        contact.link = robot.right_foot if i % 2 == 0 else robot.left_foot
        contact.takeoff_clearance = 0.15  # [m]
        contact.takeoff_tangent = (2 * contact.n - contact.t) / sqrt(5)
    contact_feed.staircase_handles = [pymanoid.Box(
        X=0.15, Y=0.4, Z=0.2, pos=[c.p[0], -0.15, c.p[2]], dZ=-0.2,
        color='r') for c in contact_feed.contacts]
    sim.move_camera_to([
        [-0.85, -0.23,  0.46, -1.92],
        [-0.52,  0.41, -0.75,  5.09],
        [-0.02, -0.88, -0.48,  4.67],
        [0.,  0.,  0.,  1.]])
    return contact_feed


def get_scenario():
    scenario = None
    if COMANOID_MODEL_FOUND and "--comanoid" in sys.argv:
        scenario = "comanoid"
    elif "--elliptic" in sys.argv:
        scenario = "elliptic"
    elif "--flat" in sys.argv:
        scenario = "flat"
    elif "--regular" in sys.argv:
        scenario = "regular"
    if scenario is None:
        print_usage()
    options = ["elliptic", "flat", "regular"]
    if COMANOID_MODEL_FOUND:
        options = ["comanoid"] + options
    while scenario not in options:
        scenario = raw_input("Which scenario in %s? " % str(options))
    return scenario


def terminate_in_double_support(contact_feed):
    """
    Duplicate last two contacts so that the robot terminates in double support
    at the end of a non-cylic contact feed.

    Parameters
    ----------
    contact_feed : pymanoid.ContactFeed
        Contact sequence of the walking scenario.
    """
    assert not contact_feed.cyclic, "contact sequence is cyclic"
    assert contact_feed.contacts[-1].link == robot.left_foot
    penultimate_contact = contact_feed.contacts[-2].copy()
    last_contact = contact_feed.contacts[-1].copy()
    delta = last_contact.p - penultimate_contact.p
    t, n = last_contact.t, last_contact.n
    p = penultimate_contact.p + dot(delta, t) * t + dot(delta, n) * n
    penultimate_contact.set_pos(p)
    penultimate_contact.set_color('b')
    contact_feed.contacts.append(penultimate_contact)
    contact_feed.contacts.append(last_contact)


def load_scenario():
    scenario = get_scenario()
    if scenario == "comanoid":
        contact_feed = load_comanoid()
    elif scenario == "regular":
        contact_feed = load_regular_staircase()
    elif scenario == "flat":
        contact_feed = load_flat_floor_staircase()
    else:  # scenario == "elliptic"
        contact_feed = load_elliptic_staircase()
    if not contact_feed.cyclic:
        terminate_in_double_support(contact_feed)
    return contact_feed


def init_robot_stance(contact_feed, robot):
    init_com = 0.5 * (contact_feed.contacts[0].p + contact_feed.contacts[1].p)
    init_com += [0., 0., robot.leg_length]
    stance = Stance(
        com=PointMass(pos=init_com, mass=robot.mass, visible=False),
        left_foot=contact_feed.contacts[1].copy(),
        right_foot=contact_feed.contacts[0].copy())
    stance.left_foot.set_name('LeftFootTarget')
    stance.right_foot.set_name('RightFootTarget')
    stance.left_foot.hide()
    stance.right_foot.hide()
    robot.set_pos([0., 0., 2.])  # start PG with the robot above contacts
    stance.bind(robot)
    robot.ik.solve(max_it=50)
    return stance


def customize_ik(robot):
    not_upper = set(robot.whole_body) - set(robot.upper_body)
    min_upper_vel = pymanoid.tasks.MinVelTask(
        robot, weight=5e-6, exclude_dofs=not_upper)
    robot.ik.add_task(min_upper_vel)


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print_usage()
        sys.exit()
    sim = pymanoid.Simulation(dt=3e-2)
    try:  # use HRP4 if available
        from hrp4_description import set_velocity_limits
        robot = pymanoid.robots.HRP4()
        set_velocity_limits(robot)
    except:  # otherwise use default model
        robot = pymanoid.robots.JVRC1()
    robot.set_transparency(0.2)
    sim.set_viewer()
    contact_feed = load_scenario()
    stance = init_robot_stance(contact_feed, robot)
    pendulum = InvertedPendulum(
        robot.com, robot.comd,
        contact=contact_feed.contacts[0],
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX)
    robot.setup_ik_for_walking(pendulum.com)
    customize_ik(robot)

    controller = WalkingController(pendulum, contact_feed)
    com_traj_drawer = TrajectoryDrawer(pendulum.com)
    preview_traj_drawer = PreviewTrajectoryDrawer(controller)
    sim.schedule(controller, log_comp_times=True)
    sim.schedule(pendulum)
    sim.schedule(robot.ik)
    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(preview_traj_drawer)
    sim.step()

    if "--ipopt" in sys.argv:
        controller.one_step.capture_pb.set_nlp_solver("ipopt")
        controller.zero_step.capture_pb.set_nlp_solver("ipopt")

    if IPython.get_ipython() is None:
        IPython.embed()