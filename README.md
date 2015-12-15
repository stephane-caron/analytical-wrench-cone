# Stability of Surface Contacts for Humanoid Robots

## Closed-Form Formulae of the Contact Wrench Cone for Rectangular Support Areas

<img src="https://raw.githubusercontent.com/stephane-caron/icra-2015/master/.illustration.png" height="250" />

Source code for http://arxiv.org/abs/1501.04719

## Abstract

Humanoid robots locomote by making and breaking contacts with their
environment. A crucial problem is therefore to find precise criteria for
a given contact to remain stable or to break. For rigid surface contacts, the
most general criterion is the Contact Wrench Condition (CWC). To check whether
a motion satisfies the CWC, existing approaches take into account a large
number of individual contact forces (for instance, one at each vertex of the
support polygon), which is computationally costly and prevents the use of
efficient inverse-dynamics methods. Here we argue that the CWC can be
explicitly computed without reference to individual contact forces, and give
closed-form formulae in the case of rectangular surfaces -- which is of
practical importance. It turns out that these formulae simply and naturally
express three conditions: (i) Coulomb friction on the resultant force, (ii) ZMP
inside the support area, and (iii) bounds on the yaw torque. Conditions (i) and
(ii) are already known, but condition (iii) is, to the best of our knowledge,
novel. It is also of particular interest for biped locomotion, where undesired
foot yaw rotations are a known issue. We also show that our formulae yield
simpler and faster computations than existing approaches for humanoid motions
in single support, and demonstrate their consistency in the OpenHRP simulator. 

Authors:
[Stéphane Caron](https://scaron.info),
[Quang-Cuong Pham](https://www.normalesup.org/~pham/) and
[Yoshihiko Nakamura](http://www.ynl.t.u-tokyo.ac.jp/)

## Robot model

You need to have your own model of HRP-4 in order to run the main script
`generate_motion.py`. The two model files that are not distributed in this
repository are: 

- `model.dae` (COLLADA model) 
- `model.py` (joint limits)

They are both located in `dmotions/hrp4`.

## Dependencies

- [CVXOPT](http://cvxopt.org/)
  - used for Quadratic Programming
  - tested with version 1.1.7
- [OpenRAVE](https://github.com/rdiankov/openrave)
  - used for forward kinematics and visualization
  - tested with commit `f68553cb7a4532e87f14cf9db20b2becedcda624` in branch
    `latest_stable`
  - you may need to [fix the Collision report issue](https://github.com/rdiankov/openrave/issues/333#issuecomment-72191884)
- [NumPy](http://www.numpy.org/)
  - used for scientific computing
  - tested with version 1.8.2
- [TOPP](https://github.com/quangounet/TOPP)
  - used for time-optimal retiming of generated motions
  - tested with commit `5370e95f635ac9e50538bb748fd93cee4758fcc9` in branch
    `master`

## Usage

There are two scripts in the repository. Most of the code is organized in the
`dmotions` module, which is an early version of
[pymanoid](https://github.com/stephane-caron/pymanoid).

### [generate\_motion.py](https://github.com/stephane-caron/icra-2015/blob/master/generate_motion.py)

Script generating the complete motion (including retiming) and writing it as
POS files into `openhrp/motions/`. These files can then be executed in OpenHRP.

### [check\_polyhedron.py](https://github.com/stephane-caron/icra-2015/blob/master/check_polyhedron.py)

Code used to double-check the validity of the analytical wrench cone by random
sampling. Writes down a success rate to the standard output, and fires up a 3D
plot displaying valid (green) and erroneous (red) points. See the paper for
details.
