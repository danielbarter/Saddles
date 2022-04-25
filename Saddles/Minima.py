import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

### finding minima ###

@partial(jax.jit, static_argnums=[0])
def update_minima(function, point, step_factor):
    """
    returns the new point, and the val / grad norm at the old point.
    """
    grad = jax.grad(function)(point)
    new_point = point - step_factor * grad

    return new_point


def find_minima(
        function,
        initial_point,
        num_steps,
        step_factor
):
    """
    loop for finding minima
    """

    point = initial_point

    for step in range(num_steps):
        point = update_minima(function, point, step_factor)

    return point

### finding geodesics ###

@partial(jax.jit, static_argnums=[0])
def action(function, left_point, right_point, distance_factor):

    displacement = right_point - left_point
    squares = displacement * displacement
    graph_component = (function(right_point) - function(left_point)) ** 2
    return jnp.exp(distance_factor * squares.sum()) - 1.0 +  graph_component


@partial(jax.jit, static_argnums=[0])
def lagrangian(
        function,      # function defining graph
        points,        # n points
        start,         # start point. fixed
        end,           # end point. fixed
        distance_factor
):

    accumulator = action(function, start, points[0], distance_factor)

    accumulator += sum(jnp.array(
        [action(function, points[i], points[i+1], distance_factor)
         for i in range(0, points.shape[0] - 1)]))

    accumulator += action(function, points[-1], end, distance_factor)

    return accumulator


@partial(jax.jit, static_argnums=[0])
def update_geodesic(function, points, start, end, step_factor, distance_factor):

    new_points = points -  step_factor * jax.grad(lagrangian, argnums=1)(
        function,
        points,
        start,
        end,
        distance_factor)


    return new_points


def find_geodesic(
        function,
        initial_points,
        start,
        end,
        num_steps,
        step_factor,
        distance_factor,
        path_save_frequency=1000
):

    result = []
    points = initial_points
    result.append(points)

    for step in range(num_steps):
        points = update_geodesic(function, points, start, end, step_factor, distance_factor)
        if step % path_save_frequency == 0:
            result.append(points)
            print("step: ", step)

    result.append(points)
    return result


def compute_initial_points(start, end, number_of_points):
    ts = np.linspace(0.0, 1.0, number_of_points+1)[1:]
    points = [ start * ( 1 - t ) + end * t for t in ts ]
    return jnp.stack(points)


