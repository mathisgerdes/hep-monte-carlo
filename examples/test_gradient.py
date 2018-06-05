import numpy as np
import matplotlib.pyplot as plt
from hepmc.core.densities import UnconstrainedCamel

target = UnconstrainedCamel(1)
x = np.linspace(0, 1, 1000)
pdf = target.pdf(x)
print('t')
log_pdf = target.pot(x)
print('t2')
pdf_gradient = target.pdf_gradient(x)
log_pdf_gradient = target.pot_gradient(x)

plt.subplot(411)
plt.title('PDF')
plt.plot(x, pdf)

plt.subplot(412)
plt.title('Log PDF')
plt.plot(x, log_pdf)

plt.subplot(413)
plt.title('PDF Gradient')
plt.plot(x, pdf_gradient)

plt.subplot(414)
plt.title('Log PDF Gradient')
plt.plot(x, log_pdf_gradient)

plt.tight_layout()
plt.show()
