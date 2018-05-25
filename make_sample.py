import sys
import os
import json
from multiprocessing import Pool
import numpy as np
from monte_carlo import *

sampler_module = None
save_base = None


def run(config):
    # parse params
    for key in config['params']:
        config['params'][key] = eval(config['params'][key])
    if 'params_vary' in config:
        for key in config['params_vary']:
            config['params_vary'][key] = eval(config['params_vary'][key])

    if 'params_vary' in config:
        runner = RunIterator(config)
        runner.run()
    else:
        print('START SINGLE ' + config['name'], flush=True)
        run_single(config)


def run_single(config, params=None, name='sample'):
    if params is None:
        params = config['params']
    print("STARTING RUN " + str(params))

    target_class = eval(config['target'])
    target = target_class(params['ndim'], **eval(config['target_args']))
    util.count_calls(target, 'pdf', 'pot_gradient')

    sampler, init = sampler_module.get_sampler(target, **params)
    sample = sampler.sample(config['size'], init, log_every=-1)

    if 'binning' in config:
        binning = eval(config['binning'])
        sample._bin_wise_chi2 = util.bin_wise_chi2(sample, *binning)

    print("FINISHED " + str(params) + ':\n' + str(sample), flush=True)
    sample.save(os.path.join(save_base, name))

    info = {'mean': np.array(sample.mean).tolist(),
            'variance': np.array(sample.variance).tolist(),
            'pdf_calls': target.pdf.count,
            'pdf_gradient_calls': target.pdf.count,
            'eff_sample_size': np.array(sample.effective_sample_size).tolist(),
            'accept_rate': sample.accept_ratio,
            'chi2': sample.bin_wise_chi2[0],
            'chi2_p': sample.bin_wise_chi2[1],
            'chi2_n': sample.bin_wise_chi2[2]}
    return info


class RunIterator(object):
    def __init__(self, config):
        self.save_base = save_base
        self.config = config

        self.params = config['params']

        self.vary_names = list(config['params_vary'].keys())
        self.vary_values = list(config['params_vary'].values())

    def run_iter(self, index):
        vary_values = [val[index] for val in self.vary_values]
        params_iter = dict(zip(self.vary_names, vary_values))
        params_iter.update(self.params)
        return run_single(self.config, params_iter, 'run-%d' % index)

    def run(self):
        print('START ' + self.config['name'], flush=True)

        var_val_listed = [np.array(v).tolist() for v in self.vary_values]
        results = {'params_vary': dict(zip(self.vary_names, var_val_listed)),
                   'params': self.params,
                   'sample_size': self.config['size'],
                   'seed': self.config['seed']}

        indices = list(range(len(self.vary_values[0])))
        with Pool() as pool:
            infos = pool.map(self.run_iter, indices)

        for info in infos:
            for key in info:
                try:
                    results[key].append(info[key])
                except KeyError:
                    results[key] = [info[key]]

        with open(self.save_base + '.json', 'w') as out_file:
            json.dump(results, out_file, indent=2)

if __name__ == '__main__':
    config_file = sys.argv[1]
    dir_base = os.path.join(base, 'out')
    if not os.path.exists(dir_base):
        os.makedirs(dir_base)

    with open(config_file) as in_file:
        configs = json.load(in_file)
    for run_config in configs:
        # load sampler module
        sampler_module = getattr(interfaces, run_config['sampler'])

        # set, and make sure save directory exists
        save_base = os.path.join(dir_base, run_config['name'])
        if not os.path.exists(save_base):
            os.makedirs(save_base)

        # set random seed if given
        if 'seed' not in run_config:
            seed = np.random.randint(0, 2**32-1)
            run_config['seed'] = seed
        np.random.seed(run_config['seed'])

        run(run_config)
