from functools import partial

import jax
import jax.numpy as jnp
from jax import P
from jax.ad_checkpoint import checkpoint_name, checkpoint_policies
from jax._src.interpreters.remat import remat_transform

policy = checkpoint_policies.save_only_these_names('foo')

jax.config.update('jax_remat3', True)
jax.config.update('jax_traceback_filtering', 'off')
jax.config.update('jax_check_tracer_leaks', True)
jax.config.update('jax_num_cpu_devices', 2)

Auto = jax.sharding.AxisType.Auto
jax.set_mesh(jax.make_mesh((2,), ('i',), (Auto,)))

def f(x):
  y = checkpoint_name(jax.lax.sin(x), 'foo')
  z = jax.smap(lambda x: x * jax.lax.cos(y).sum(), in_axes=0, out_axes=0, axis_name='i')(x)
  return z.sum()

x = jax.device_put(jnp.arange(2.), P('i'))
y1, f_ = remat_transform(policy, f, x)
y2 = f_(x)
print(y1)
print(y2)

print(jax.jit(lambda x: remat_transform(None, f, x)[0]).trace(x).jaxpr)
print(jax.jit(f_).trace(x).jaxpr)

# jax.grad(jax.remat(f))(x)

