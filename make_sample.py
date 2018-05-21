import sys
import os
import json
import numpy as np
from monte_carlo import *


def run(config, dir_base):
    save_base = os.path.join(dir_base, config['name'])
    # make sure save directory exists
    if not os.path.exists(save_base):
        os.makedirs(save_base)

    # load sampler module
    sampler_module = getattr(interfaces, config['sampler'])
    if 'params_vary' in config:
        run_all(config, save_base, sampler_module)
    else:
        run_single(config, save_base, sampler_module)


def run_single(config, save_base, sampler_module):
    print(config['name'], flush=True)
    params = config['params']
    for param in params:
        params[param] = eval(params[param])

    if 'binning' in config:
        binning = eval(config['binning'])
    else:
        binning = None

    np.random.seed(42)

    target = eval(config['target'])(params['ndim'],
                                    **eval(config['target_args']))
    util.count_calls(target, 'pdf', 'pot_gradient')

    sampler, init = sampler_module.get_sampler(target, **params)
    sample = sampler.sample(config['size'], init)

    if binning is not None:
        sample._bin_wise_chi2 = util.bin_wise_chi2(sample, *binning)

    print(sample, flush=True)
    sample.save(os.path.join(save_base, 'sample'))


def run_all(config, save_base, sampler_module):
    print(config['name'], flush=True)
    params = config['params']
    for param in params:
        params[param] = eval(params[param])

    vary_names = config['params_vary'].keys()
    vary_values = [eval(v) for v in config['params_vary'].values()]

    if 'binning' in config:
        binning = eval(config['binning'])
    else:
        binning = None

    results = {
        'params_vary': dict(zip(vary_names,
                                [np.array(v).tolist() for v in vary_values])),
        'params': params,
        'sample_size': config['size'],
        'mean': [],
        'variance': [],
        'chi2': [],
        'chi2_p': [],
        'chi2_n': [],
        'eff_sample_size': [],
        'pdf_calls': [],
        'pot_gradient_calls': [],
        'accept_rate': []
    }

    index = 0
    for param_it in zip(*vary_values):
        np.random.seed(42)
        kwargs = dict(zip(vary_names, param_it))
        kwargs.update(params)

        target = eval(config['target'])(kwargs['ndim'], **eval(config['target_args']))
        util.count_calls(target, 'pdf', 'pot_gradient')

        sampler, init = sampler_module.get_sampler(target, **kwargs)
        print(config['name'] + ", run %d" % (index + 1), flush=True)
        sample = sampler.sample(config['size'], init)

        # should be evaluated before calling bin chi^2
        results['pdf_calls'].append(target.pdf.count)
        results['pot_gradient_calls'].append(target.pot_gradient.count)

        results['mean'].append(np.array(sample.mean).tolist())
        results['variance'].append(np.array(sample.variance).tolist())
        if binning is None:
            chi2, chi2_p, chi2_n = sample.bin_wise_chi2
        else:
            chi2, chi2_p, chi2_n = util.bin_wise_chi2(sample, *binning)
            sample._bin_wise_chi2 = (chi2, chi2_p, chi2_n)

        results['chi2'].append(chi2)
        results['chi2_p'].append(chi2_p)
        results['chi2_n'].append(chi2_n)
        results['eff_sample_size'].append(
            np.array(sample.effective_sample_size).tolist())
        results['accept_rate'].append(sample.accept_ratio)

        sample.save(os.path.join(save_base, 'run-%d' % index))
        index += 1

    with open(save_base + '.json', 'w') as out_file:
        json.dump(results, out_file, indent=2)


if __name__ == '__main__':
    config_file = sys.argv[1]
    base = os.path.split(config_file)[0]
    with open(config_file) as in_file:
        configs = json.load(in_file)
    for run_config in configs:
        run(run_config, base)
