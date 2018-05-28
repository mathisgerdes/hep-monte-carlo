import sys
import os
import json
from multiprocessing import Pool
import numpy as np
from monte_carlo import *


dir_base = None


def join_dir_safe(path, dir_name):
    new_dir = os.path.join(path, dir_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def _all_to_list(dictionary):
    for key in dictionary:
        val = dictionary[key]
        if isinstance(val, np.ndarray):
            dictionary[key] = val.tolist()
    return dictionary


def _run(config, meta=True, save=True):
    if 'params_vary' in config:
        runner = RunIterator(config)
        return runner.run(meta=meta)
    else:
        print('START SINGLE ' + config['name'], flush=True)
        return run_single(config, save=save)


def _run_averaging(config):
    results = dict()
    name = config['name']
    for it in range(config['repeats']):
        config['name'] = name + '-run-%d' % (it + 1)
        info = _run(config, meta=False, save=False)
        for key in info:
            val = info[key]
            try:
                results[key].append(val)
            except KeyError:
                results[key] = [val]
    config['name'] = name  # restore name
    for key in results:
        val = results[key]
        if isinstance(val[0], list):
            av_val = []
            for variation in range(len(results[key][0])):
                av_val.append(np.mean(
                    np.array(
                        [results[key][i][variation]
                         for i in range(len(val))],
                        dtype=np.float), axis=0).tolist())
            results[key] = av_val
        else:
            results[key] = np.mean(np.array(val, dtype=np.float), axis=0)

    for meta in ['params_vary', 'params', 'size']:
        try:
            results[meta] = config[meta]
        except KeyError:
            pass  # params vary might not be in config.

    save_base = os.path.join(dir_base, name)
    with open(save_base + '.json', 'w') as out_file:
        json.dump(_all_to_list(results), out_file, indent=2)


def run(config):
    # parse params
    for key in config['params']:
        config['params'][key] = eval(config['params'][key])
    if 'params_vary' in config:
        for key in config['params_vary']:
            config['params_vary'][key] = eval(config['params_vary'][key])

    if 'repeats' in config:
        _run_averaging(config)

    else:
        _run(config)
    print('CONFIG %s DONE' % config['name'], flush=True)


def run_single(config, params=None, name='sample', save=True):
    if params is None:
        params = config['params']
    print("STARTING RUN " + str(params), flush=True)
    np.random.seed()  # important for multiprocessing

    # load sampler module
    sampler_module = getattr(interfaces, config['sampler'])
    target_class = eval(config['target'])
    target = target_class(params['ndim'], **eval(config['target_args']))
    util.count_calls(target, 'pdf', 'pot_gradient')

    sampler, init = sampler_module.get_sampler(target, **params)
    sample = sampler.sample(config['size'], init, log_every=-1)

    if 'binning' in config:
        binning = eval(config['binning'])
        sample._bin_wise_chi2 = util.bin_wise_chi2(sample, *binning)

    print("FINISHED " + str(params) + ':\n' + str(sample), flush=True)
    if save:
        save_base = join_dir_safe(dir_base, config['name'])
        sample.save(os.path.join(save_base, name))

    info = {'mean': sample.mean,
            'variance': sample.variance,
            'pdf_calls': target.pdf.count,
            'pdf_gradient_calls': target.pdf.count,
            'eff_sample_size': sample.effective_sample_size,
            'accept_rate': sample.accept_ratio,
            'chi2': sample.bin_wise_chi2[0],
            'chi2_p': sample.bin_wise_chi2[1],
            'chi2_n': sample.bin_wise_chi2[2]}
    return _all_to_list(info)


class RunIterator(object):
    def __init__(self, config):
        self.save_base = os.path.join(dir_base, config['name'])
        self.config = config

        self.params = config['params']

        self.vary_names = list(config['params_vary'].keys())
        self.vary_values = list(config['params_vary'].values())

    def run_iter(self, index):
        vary_values = [val[index] for val in self.vary_values]
        params_iter = dict(zip(self.vary_names, vary_values))
        params_iter.update(self.params)
        save = 'save_all' in self.config and eval(self.config['save_all'])
        return run_single(self.config, params_iter, 'var-%d' % index, save=save)

    def run(self, meta=True):
        print('START ' + self.config['name'], flush=True)

        results = dict()
        vars_lists = [np.array(v).tolist() for v in self.vary_values]
        if meta:
            results = {'params_vary': dict(zip(self.vary_names, vars_lists)),
                       'params': self.params,
                       'sample_size': self.config['size']}

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

        return results


if __name__ == '__main__':
    config_file = sys.argv[1]
    base = os.path.split(config_file)[0]
    dir_base = join_dir_safe(base, 'out')

    with open(config_file) as in_file:
        configs = json.load(in_file)

    single_configs = [config for config in configs
                      if 'params_vary' not in config]
    vary_configs = [config for config in configs
                    if 'params_vary' in config]
    for run_config in vary_configs:
        run(run_config)

    with Pool() as pool:
        pool.map(run, single_configs)

    print("SCRIPT DONE")
