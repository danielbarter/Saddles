import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

### finding minima ###

@partial(jax.jit, static_argnums=[0])
def update_minima(function, point, factor):
    """
    returns the new point, and the val / grad norm at the old point.
    """


    val = function(point)
    grad = jax.grad(function)(point)
    grad_norm = jax.numpy.sqrt((grad * grad).sum())

    new_point = point - factor * grad


    return new_point, val, grad_norm


def find_minima(
        function,
        initial_point,
        num_steps,
        factor,
        minimization_report_file,
        log_frequency=1000
):
    """
    loop for finding minima
    """

    point = initial_point
    function_vals = np.zeros(num_steps)
    grad_norms = np.zeros(num_steps)

    for step in range(num_steps):
        point, val, grad_norm = update_minima(function, point, factor)
        function_vals[step] = val
        grad_norms[step] = grad_norm

        if step % log_frequency == 0:
            print("step:      ", step)
            print("function:  ", val)
            print("grad norm: ", grad_norm)

    fig, axs = plt.subplots(2, 1, figsize=(5,10), gridspec_kw={"height_ratios":[1,1]})
    axs[0].plot(function_vals)
    axs[0].set_title("function vals")
    axs[1].plot(grad_norms)
    axs[1].set_title("grad norms")
    fig.savefig(minimization_report_file)

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
def update_geodesic(function, points, start, end, factor, distance_factor):
    val = lagrangian(function, points, start, end, distance_factor)

    new_points = points -  factor * jax.grad(lagrangian, argnums=1)(
        function,
        points,
        start,
        end,
        distance_factor)


    return new_points, val



def compute_initial_points(start, end, number_of_points):
    ts = np.linspace(0.0, 1.0, number_of_points+1)[1:]
    points = [ start * ( 1 - t ) + end * t for t in ts ]
    return jnp.stack(points)



def find_geodesic(
        function,
        initial_points,
        start,
        end,
        num_steps,
        factor,
        distance_factor,
        geodesic_report_file,
        log_frequency=1000

):
    result = []
    points = initial_points
    result.append(points)
    lagrangian_vals = np.zeros(num_steps)


    for step in range(num_steps):
        points, val = update_geodesic(function, points, start, end, factor, distance_factor)
        lagrangian_vals[step] = val
        if step % 1000 == 0:
            result.append(points)
            print("step:      ", step)
            print("lagrangian:", val)


    result.append(points)
    function_vals = np.zeros(points.shape[0])
    grad_norms = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        grad = jax.grad(function)(points[i]).flatten()
        grad_norms[i] = jnp.sqrt(grad.dot(grad))
        function_vals[i] = function(points[i])

    fig, axs = plt.subplots(3, 1, figsize=(5,15), gridspec_kw={"height_ratios":[1,1,1]})
    axs[0].plot(lagrangian_vals)
    axs[0].set_title("lagrangian vals")

    axs[1].plot(grad_norms)
    axs[1].set_title("geodesic grads")

    axs[2].plot(function_vals)
    axs[2].set_title("function vals above geodesic")

    fig.savefig(geodesic_report_file)

    return result


