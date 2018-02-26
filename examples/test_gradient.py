import numpy as np
import matplotlib.pyplot as plt
from densities.camel import UnconstrainedCamel

target = UnconstrainedCamel()
x = np.linspace(-3, 4, 100)
pdf = [target.pdf(x) for x in x]
print('t')
log_pdf = [target.log_pdf(x) for x in x]
print('t2')
pdf_gradient = [target.pdf_gradient(x) for x in x]
log_pdf_gradient = [target.log_pdf_gradient(x) for x in x]

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
