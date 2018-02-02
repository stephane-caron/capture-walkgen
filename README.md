# Capturability-based Analysis, Optimization and Control of 3D Bipedal Walking

Source code for https://hal.archives-ouvertes.fr/hal-01689331

## Abstract

Capturability analysis of the linear inverted pendulum model (LIPM) enabled
walking over even terrains based on the *capture point*. We generalize this
analysis to the inverted pendulum model (IPM) and show how it enables 3D
walking over uneven terrains based on *capture inputs*. Thanks to a tailored
optimization scheme, we can compute these inputs fast enough for a real-time
control loop. We implement this approach as open-source software and
demonstrate it in simulations.

Authors:
[Stéphane Caron](https://scaron.info),
[Adrien Escande](https://sites.google.com/site/adrienescandehomepage/),
[Leonardo Lanari](http://www.diag.uniroma1.it/~lanari/) and
[Bastien Mallein](http://www.math.univ-paris13.fr/~mallein/)

## Installation

The following instructions were verified on Ubuntu 14.04:

- Install OpenRAVE: here are [instructions for Ubuntu 14.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html) as well as [for Ubuntu 16.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html)
- Install Python dependencies: ``sudo apt-get install cython python python-dev python-pip python-scipy``
- Install the QP solver module for Python: ``sudo pip install quadprog``

Finally, clone this repository and its submodules via:

```bash
git clone --recursive https://github.com/stephane-caron/capture-walking.git
```

### Optional

If you plan on trying out IPOPT (``--ipopt`` option), you will need to [install
CasADi](https://github.com/casadi/casadi/wiki/InstallationLinux). Pre-compiled
binaries are available, but I recommend you [build it from
source](https://github.com/casadi/casadi/wiki/InstallationLinux). When
installing IPOPT, make sure to install the MA27 linear solver
(``ThirdParty/HSL`` folder).
  
## Questions?

Feel free to post your questions or comments in the issue tracker.
