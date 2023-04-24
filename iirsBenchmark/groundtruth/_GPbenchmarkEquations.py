
import jax.numpy as jnp

gpbenchmarkPyData = {
    'Korns-11' : {
        'string expression' : '6.87 + 11*np.cos( 7.23*(x**3)',
        'latex expression'  : r'6.87 + 11*cos( 7.23*(x^3)',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x, y, z, v, w: 6.87 + 11*jnp.cos( 7.23*(x**3) ))(*args),
    },
    'Korns-12' : {
        'string expression' : '2 - 2.1*np.cos(9.8*x)*np.sin(1.3*w)',
        'latex expression'  : r'2 - 2.1*cos(9.8*x)*sin(1.3*w)',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x, y, z, v, w: 2 - 2.1*jnp.cos(9.8*x)*jnp.sin(1.3*w))(*args),
    },
    'Vladislavleva-4' : {
        'string expression' : '10/(5+(x1-3)**2+(x2-3)**2+(x3-3)**2+(x4-3)**2+(x5-3)**2)',
        'latex expression'  : r'10/(5 + sum_{i=1}^{5}(x_i - 3)^2)',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x1, x2, x3, x4, x5: 10/(5 + sum([(x - 3)**2 for x in [x1, x2, x3, x4, x5]])))(*args),
    },
    'Pagie-1' : {
        'string expression' : '1/(1 + x**-4) + 1/(1 + y**-4)',
        'latex expression'  : r'1/(1 + x^{-4}) + 1/(1 + y^{-4})',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x, y: 1/(1 + x**-4) + 1/(1 + y**-4))(*args),
    }
}