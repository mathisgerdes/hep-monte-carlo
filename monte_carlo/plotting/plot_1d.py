import numpy as np
import matplotlib.pyplot as plt

def plot_1d(samples, target_pdf, mapping_pdf=None, target_log_pdf_gradient=None, surrogate_log_pdf_gradient=None):
    x = np.linspace(0, 1, 100)
    y = [target_pdf(x_i) for x_i in x]
    
    plt.subplot(211)
    plt.title('Chain')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(samples)
    
    if target_log_pdf_gradient is not None and surrogate_log_pdf_gradient is not None:
        plt.subplot(224)
        plt.title('Log PDF Gradient')
        plt.plot(x, [target_log_pdf_gradient(x) for x in x], label='True PDF')
        plt.plot(x, [surrogate_log_pdf_gradient(np.array([x])) for x in x], label='Surrogate')
        plt.legend()
        
        plt.subplot(223)
    else:
        plt.subplot(212)
        
    plt.title('Samples')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.hist(samples, bins=50, density=True, label='MC Samples')
    plt.plot(x, y, 'r-', label='True PDF')
    if mapping_pdf is not None:
        y_map = [mapping_pdf(x_i) for x_i in x]
        plt.plot(x, y_map, 'go', label='Mapping')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
