import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

### finding minima ###


def training_logger(step,val):
    step_string = ("step: " + str(step)).ljust(15)
    val_string = "val: " + str(val)
    print(step_string, val_string)


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
        step_factor,
        log_frequency=1000
):
    """
    loop for finding minima
    """

    print("computing minima...")

    point = initial_point

    for step in range(num_steps):
        point = update_minima(function, point, step_factor)
        if step % log_frequency == 0:
            training_logger(step, function(point))


    print("\n\n\n")
    return point

### finding critical_path ###

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
def update_critical_path(function, points, start, end, step_factor, distance_factor):

    new_points = points -  step_factor * jax.grad(lagrangian, argnums=1)(
        function,
        points,
        start,
        end,
        distance_factor)


    return new_points


def find_critical_path(
        function,
        initial_points,
        start,
        end,
        num_steps,
        step_factor,
        distance_factor,
        log_frequency=1000
):

    print("computing critical_path...")
    result = []
    points = initial_points
    result.append(points)

    for step in range(num_steps):
        points = update_critical_path(function, points, start, end, step_factor, distance_factor)
        if step % log_frequency == 0:
            result.append(points)
            training_logger(step, lagrangian(function, points, start, end, distance_factor))

    result.append(points)

    print("\n\n\n")
    return result


def compute_initial_points(start, end, number_of_points):
    ts = np.linspace(0.0, 1.0, number_of_points+1)[1:]
    points = [ start * ( 1 - t ) + end * t for t in ts ]
    return jnp.stack(points)
