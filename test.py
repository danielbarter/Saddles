from Saddles.Minima import *
from Saddles.TestSurfaces import *
import os
import sys
import subprocess


wolfe_schlegel_test_dir = './scratch/wolfe_schlegel'
muller_brown_test_dir = './scratch/muller_brown'

if os.path.isdir('./scratch'):
    subprocess.run(['rm', '-r', './scratch'])

subprocess.run(['mkdir', './scratch'])
subprocess.run(['mkdir', wolfe_schlegel_test_dir ])
subprocess.run(['mkdir', muller_brown_test_dir ])


def wolfe_schlegel_test():

    test_dir = wolfe_schlegel_test_dir

    minima_1 = find_minima(
        function=wolfe_schlegel,
        initial_point=jnp.array([-1.5, 1.5]),
        num_steps=10000,
        step_factor=0.0001)

    minima_2 = find_minima(
        function=wolfe_schlegel,
        initial_point=jnp.array([-1, -1.5]),
        num_steps=10000,
        step_factor=0.0001)

    minima_3 = find_minima(
        function=wolfe_schlegel,
        initial_point=jnp.array([1.0, -1.5]),
        num_steps=10000,
        step_factor=0.0001)

    special_point = jnp.array([0.1, 0.1])

    initial_points_1 = compute_initial_points(minima_1, special_point, 25)
    initial_points_2 = compute_initial_points(special_point, minima_3, 25)
    initial_points = jnp.vstack([initial_points_1, initial_points_2])


    paths = find_geodesic(
        function=wolfe_schlegel,
        initial_points=initial_points,
        start=minima_1,
        end=minima_3,
        num_steps=60000,
        step_factor=0.00001,
        distance_factor=100)

    contour_2d(
        function=wolfe_schlegel,
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
        levels=np.arange(-100,100,5),
        paths=paths,
        title="wolfe schlegel",
        contour_file = test_dir + '/contour_plot.gif'
               )


def muller_brown_test():

    test_dir = muller_brown_test_dir

    minima_1 = find_minima(
        function=muller_brown,
        initial_point=jnp.array([-0.7, 1.5]),
        num_steps=10000,
        step_factor=0.0001)

    minima_2 = find_minima(
        function=muller_brown,
        initial_point=jnp.array([0.0, 0.5]),
        num_steps=10000,
        step_factor=0.0001)

    minima_3 = find_minima(
        function=muller_brown,
        initial_point=jnp.array([0.5, 0.0]),
        num_steps=10000,
        step_factor=0.0001)

    initial_points = compute_initial_points(minima_1, minima_2, 30)

    paths=find_geodesic(
        function=muller_brown,
        initial_points=initial_points,
        start=minima_1,
        end=minima_2,
        num_steps=20000,
        step_factor=0.000001,
        distance_factor=100)

    contour_2d(
        function=muller_brown,
        x_min=-1.7,
        x_max=1.3,
        y_min=-0.5,
        y_max=2.2,
        levels=np.arange(-200,200,10),
        paths=paths,
        title="muller brown",
        contour_file = test_dir + '/contour_plot.gif')


wolfe_schlegel_test()
muller_brown_test()
