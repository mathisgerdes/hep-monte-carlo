from .gaussian import Gaussian
from .camel import Camel, UnconstrainedCamel
from .uniform import Uniform
from .banana import Banana
from .rambo import Rambo, RamboOnDiet

try:
    from .qcd import ee_qq
except ModuleNotFoundError:
    print("Can't find Sherpa installation, skipping ee_qq.")
    ee_qq = None

__all__ = ['Gaussian', 'Camel', 'UnconstrainedCamel', 'Uniform', 'Banana',
           'Rambo', 'RamboOnDiet', 'ee_qq']
