import sys
import os
import json
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from hepmc import *


dir_base = None


# allow encoding of numpy arrays via tolist
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def join_dir_safe(path, dir_name):
    new_dir = os.path.join(path, dir_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def update_meta(info, config, meta_vars=('params_vary', 'params', 'size')):
    for meta in meta_vars:
        try:
            info[meta] = config[meta]
        except KeyError:
            pass  # params vary might not be in config.


def _run(config, meta=True):
    save = 'save_all' in config and eval(config['save_all'])
    if 'params_vary' in config:
        runner = RunIterator(config, save=save)
        return runner.run(meta=meta)
    else:
        print('START SINGLE ' + config['name'], flush=True)
        return make_sample(config, save=save)


def _run_averaging(config):
    results = defaultdict(list)
    name = config['name']
    for it in range(config['repeats']):
        config['name'] = name + '-run-%d' % (it + 1)
        info = _run(config, meta=False)
        for key in info:
            val = info[key]
            results[key].append(val)
    config['name'] = name  # restore name

    averaged = dict()
    for key in results:
        val = results[key]
        if isinstance(val[0], list):
            av_val = []
            var_val = []
            for variation in range(len(results[key][0])):
                variations = np.array([results[key][i][variation]
                                       for i in range(len(val))], dtype=float)
                av_val.append(np.nanmean(variations, axis=0))
                var_val.append(np.nanvar(variations, axis=0))
            averaged[key] = av_val
            averaged[key + '_var'] = var_val
        else:
            averaged[key] = np.mean(np.array(val, dtype=np.float), axis=0)

    update_meta(averaged, config)

    save_base = os.path.join(dir_base, name)
    with open(save_base + '.json', 'w') as out_file:
        json.dump(averaged, out_file, indent=2, cls=NumpyEncoder)


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


def make_sample(config, params=None, name='sample', save=True):
    if params is None:
        params = config['params']
    print("START SAMPLING " + str(params), flush=True)
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

    print("FINISHED SAMPLING " + str(params) + ':\n' + str(sample), flush=True)
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
    return info


class RunIterator(object):
    def __init__(self, config, save=False):
        self.save_base = os.path.join(dir_base, config['name'])
        self.config = config
        self.save = save

        self.vary_names = list(config['params_vary'].keys())
        self.vary_values = list(config['params_vary'].values())

    def run_iter(self, index):
        vary_values = [val[index] for val in self.vary_values]
        params_iter = dict(zip(self.vary_names, vary_values))
        params_iter.update(self.config['params'])
        return make_sample(
            self.config, params_iter, 'var-%d' % index, save=self.save)

    def run(self, meta=True):
        print('START ' + self.config['name'], flush=True)

        results = defaultdict(list)
        if meta:
            update_meta(results, self.config)

        indices = list(range(len(self.vary_values[0])))
        with Pool() as param_pool:
            infos = param_pool.map(self.run_iter, indices)

        for info in infos:
            for key in info:
                results[key].append(info[key])

        with open(self.save_base + '.json', 'w') as out_file:
            json.dump(results, out_file, indent=2, cls=NumpyEncoder)

        return results


if __name__ == '__main__':
    config_file = sys.argv[1]
    base = os.path.split(config_file)[0]
    dir_base = join_dir_safe(base, 'out')

    with open(config_file) as in_file:
        configs = json.load(in_file)

    try:
        begin = int(sys.argv[2])
        try:
            end = int(sys.argv[3])
            configs = configs[begin:end]  # run a subset of configs
        except IndexError:
            configs = [configs[begin]]  # run only one config
    except IndexError:
        pass  # run all configs in config file

    single_configs = [config for config in configs
                      if 'params_vary' not in config]
    vary_configs = [config for config in configs
                    if 'params_vary' in config]

    for run_config in vary_configs:
        run(run_config)

    with Pool() as pool:
        pool.map(run, single_configs)

    print("SCRIPT DONE", flush=True)
