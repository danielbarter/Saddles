import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


@jax.jit
def wolfe_schlegel(point):
    x = point[0]
    y = point[1]
    return 10 * (x**4 + y**4 - 2 * x*x - 4 * y*y + x * y + 0.2 * x + 0.1 * y)



@jax.jit
def muller_brown(point):
    x = point[0]
    y = point[1]

    ai = [-200.0, -100.0, -170.0, 15.0]
    bi = [-1.0, -1.0, -6.5, 0.7]
    ci = [0.0, 0.0, 11.0, 0.6]
    di = [-10.0, -10.0, -6.5, 0.7]

    xi = [1.0, 0.0, -0.5, -1.0]
    yi = [0.0, 0.5, 1.5, 1.0]

    total = 0.0
    for i in range(4):
        total += ai[i] * jnp.exp(bi[i] * (x - xi[i]) * (x - xi[i]) +
                                 ci[i] * (x - xi[i]) * (y - yi[i]) +
                                 di[i] * (y - yi[i]) * (y - yi[i]))


    return  total


@partial(jax.jit, static_argnums=[0,1,2,3,4])
def contour_vals(function, x_min, x_max, y_min, y_max):
    x_vals = jnp.arange(x_min, x_max, 0.01)
    y_vals = jnp.arange(y_min, y_max, 0.01)
    l,r = jnp.meshgrid(x_vals, y_vals)
    args = jnp.stack([l,r],axis=2)
    return x_vals, y_vals, jnp.apply_along_axis(function, 2, args)


def contour_2d(
        function,
        x_min,
        x_max,
        y_min,
        y_max,
        levels,
        paths,
        title,
        contour_file):

    x_vals, y_vals, z_vals = contour_vals(function, x_min, x_max, y_min, y_max)

    ims = []
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.contour(x_vals, y_vals, z_vals, levels=levels)

    for path in paths:

        scatter = ax.scatter(path[:,0], path[:,1],color=['red'])
        ims.append([scatter])

    ani = animation.ArtistAnimation(fig, ims)
    ani.save(contour_file)
