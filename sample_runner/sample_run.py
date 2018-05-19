from inspect import getsourcefile
import importlib.util
import sys
import os
import json
import numpy as np


current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])
import monte_carlo as mc
sys.path.pop(0)


def run(config):
    # make sure save directory exists
    if not os.path.exists(config['name']):
        os.makedirs(config['name'])

    # load sampler module
    mod_path = config['sampler']
    mod_name = os.path.split(mod_path)[1].split('.', 1)[0]
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    sampler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler_module)

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
        'params': dict(zip(vary_names,
                           [np.array(v).tolist() for v in vary_values])),
        'mean': [],
        'variance': [],
        'chi2': [],
        'chi2_p': [],
        'chi2_n': [],
        'eff_sample_size': []
    }

    index = 0
    for param_it in zip(*vary_values):
        kwargs = dict(zip(vary_names, param_it))
        kwargs.update(params)
        sampler, init = sampler_module.get_sampler(**kwargs)

        print(config['name'] + ", run %d" % (index + 1))
        sample = sampler.sample(config['size'], init)
        sample.save(config['name'] + '/run-%d' % index)

        results['mean'].append(np.array(sample.mean).tolist())
        results['variance'].append(np.array(sample.variance).tolist())
        if binning is None:
            chi2, chi2_p, chi2_n = sample.bin_wise_chi2
        else:
            chi2, chi2_p, chi2_n = mc.util.bin_wise_chi2(sample, *binning)
        results['chi2'].append(chi2)
        results['chi2_p'].append(chi2_p)
        results['chi2_n'].append(chi2_n)
        results['eff_sample_size'].append(
            np.array(sample.effective_sample_size).tolist())

        index += 1

    with open(config['name'] + '.json', 'w') as out_file:
        json.dump(results, out_file, indent=2)


if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file) as in_file:
        configs = json.load(in_file)
    for run_config in configs:
        run(run_config)
