from .gaussian import Gaussian
from .camel import Camel, UnconstrainedCamel
from .uniform import Uniform
from .banana import Banana
from .rambo import Rambo, RamboOnDiet
from .sarge import Sarge

try:
    from .qcd import ee_qq, ee_qq_1g
except ModuleNotFoundError:
    print("Can't find Sherpa installation, skipping ee_qq.")
    ee_qq = None

__all__ = ['Gaussian', 'Camel', 'UnconstrainedCamel', 'Uniform', 'Banana',
           'Rambo', 'RamboOnDiet', 'Sarge', 'ee_qq', 'ee_qq_1g']
