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
from pymanoid.gui import TrajectoryDrawer
from pymanoid.models import InvertedPendulum
from pymanoid.sim import gravity_const

from capture_walking import WalkingController

# Walking control parameters
LAMBDA_MAX = 2.0 * gravity_const
LAMBDA_MIN = 0.1 * gravity_const
NB_MPC_STEPS = 10
TARGET_COM_HEIGHT = 0.8


def print_usage():
    """
    Tell the user how to call this script.
    """
    print("Usage: %s [scenario] [solver]" % sys.argv[0])
    print("Scenarios:")
    print("    --elliptic       Elliptic stairase scenario")
    print("    --flat           Flat floor scenario")
    print("    --regular        Regular stairase scenario")
    print("Solvers:")
    print("    --cps            Use CaptureProblemSolver (default)")
    print("    --ipopt          Use IPOPT")


def load_elliptic_staircase():
    """
    Load the elliptic staircase scenario.

    Returns
    -------
    contact_feed : pymanoid.ContactFeed
        Contact feed corresponding to the desired scenario.
    """
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
    """
    Load the flat floor scenario.

    Returns
    -------
    contact_feed : pymanoid.ContactFeed
        Contact feed corresponding to the desired scenario.
    """
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
    """
    Load the regular staircase scenario.

    Returns
    -------
    contact_feed : pymanoid.ContactFeed
        Contact feed corresponding to the desired scenario.
    """
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


def tweak_acyclic_contact_feed(contact_feed):
    """
    Duplicate last two contacts so that the robot terminates in double support
    at the end of a non-cylic contact feed.

    Parameters
    ----------
    contact_feed : pymanoid.ContactFeed
        Footstep sequence of the walking scenario.
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
    """
    Load scenario from command-line arguments.

    Returns
    -------
    contact_feed : pymanoid.ContactFeed
        Contact feed corresponding to the desired scenario.
    """
    scenario = None
    available = ["elliptic", "flat", "regular"]
    if "--elliptic" in sys.argv:
        scenario = "elliptic"
    elif "--flat" in sys.argv:
        scenario = "flat"
    elif "--regular" in sys.argv:
        scenario = "regular"
    if scenario is None:
        print_usage()
    while scenario not in available:
        scenario = raw_input("Which scenario in %s? " % str(available))
    if scenario == "regular":
        contact_feed = load_regular_staircase()
    elif scenario == "flat":
        contact_feed = load_flat_floor_staircase()
    else:  # scenario == "elliptic"
        contact_feed = load_elliptic_staircase()
    if not contact_feed.cyclic:
        tweak_acyclic_contact_feed(contact_feed)
    return contact_feed


def init_robot_stance(contact_feed, robot):
    """
    Setup the initial robot stance from the desired footstep sequence.

    Parameters
    ----------
    contact_feed : pymanoid.ContactFeed
        Footstep sequence.
    robot : pymanoid.Robot
        Robot model.

    Returns
    -------
    stance : pymanoid.Stance
        Initial stance, i.e. set of contacts along with CoM position.
    """
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


def setup_robot_ik(robot, pendulum):
    """
    Initialize all tasks for inverse kinematics.

    Parameters
    ----------
    robot : pymanoid.Robot
        Robot model.
    pendulum : pymanoid.InvertedPendulum
        Inverted pendulum model.
    """
    robot.setup_ik_for_walking(pendulum.com)
    not_upper_body = set(robot.whole_body) - set(robot.upper_body)
    upper_body_task = pymanoid.tasks.MinVelTask(
        robot, weight=5e-6, exclude_dofs=not_upper_body)
    robot.ik.add(upper_body_task)


class CaptureTrajectoryDrawer(pymanoid.Process):

    """
    Optional simulation process to draw capture trajectories.
    """

    def __init__(self, controller):
        super(CaptureTrajectoryDrawer, self).__init__()
        self.handles = []
        self.one_step = controller.one_step
        self.zero_step = controller.zero_step

    def on_tick(self, sim):
        """
        Main function called at each control cycle.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        """
        new_handles = []
        if self.one_step.solution is not None:
            new_handles.extend(self.one_step.draw_solution('r'))
        self.handles = new_handles


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
        robot.com, robot.comd, contact=contact_feed.contacts[0],
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX)
    setup_robot_ik(robot, pendulum)
    controller = WalkingController(
        robot, pendulum, contact_feed, NB_MPC_STEPS, TARGET_COM_HEIGHT)
    if "--ipopt" not in sys.argv:
        controller.precompute()
    com_traj_drawer = TrajectoryDrawer(pendulum.com)
    preview_traj_drawer = CaptureTrajectoryDrawer(controller)
    sim.schedule(controller, log_comp_times=True)
    sim.schedule(pendulum)
    sim.schedule(robot.ik)
    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(preview_traj_drawer)
    sim.step()
    if "--ipopt" in sys.argv:
        controller.one_step.capture_pb.nlp_solver = "ipopt"
        controller.zero_step.capture_pb.nlp_solver = "ipopt"
    if IPython.get_ipython() is None:
        IPython.embed()
