import re
import pandas as pd
import os

pd.options.display.width = 0

rx_dict = {
    'model_dir': re.compile(r'model_dir=\'(.*?)\''),
    'dataset': re.compile(r'dataset_name=\'(.*?)\''),
    'm': re.compile(r'rand_aug_m=([-+]?\d*)'),
    'n': re.compile(r'rand_aug_n=([-+]?\d*)'),
    'wd': re.compile("weight_decay=(.*?),"),
    'sc_weight': re.compile("sup_con_weight=(.*?),"),
    'split_idx': re.compile(r'split_idx=(\d)'),
    'epochs': re.compile(r'Train Epoch: (\d)'),
    'part_loss_mode': re.compile(r'part_loss_mode=(\d)'),
    'consistency_weight': re.compile(r'consistency_weight=(.*?),'),
    'lr': re.compile("lr=([-+]?\d*\.\d+|\d+)"),
    'Train Accs': re.compile("Train Accuracies: ([-+]?\d*\.\d+|\d+)")
}

save_root_dir = '/work/sagar/open_set_recognition/sweep_summary_files/ensemble_pkls'


def get_file(path):

    file = []
    with open(path, 'rt') as myfile:
        for myline in myfile:  # For each line, read to a string,
            file.append(myline)

    return file


def parse_out_file(path, rx_dict, root_dir=save_root_dir, save_name='test.pkl', save=True, verbose=True):

    file = get_file(path=path)
    for i, line in enumerate(file):
        if line.find('Namespace') != -1:

            model = {}
            s = rx_dict['model_dir'].search(line).group(1)
            exp_id = s[s.find("("):s.find(")") + 1]
            model['exp_id'] = exp_id
            model['dataset'] = rx_dict['dataset'].search(line).group(1)
            model['lr'] = rx_dict['lr'].search(line).group(1)

            break

    reverse_file = file[::-1]
    for i, line in enumerate(reverse_file):
        if line.find('Train Accuracies') != -1:

            model['Train Mean'] = re.findall("\d+\.\d+", line)[0]
            model['Train Old'] = re.findall("\d+\.\d+", line)[1]
            model['Train New'] = re.findall("\d+\.\d+", line)[2]

            for i_, line_ in enumerate(reverse_file[i:]):
                if 'Train Epoch' in line_:
                    model['Last Epoch'] = re.findall('Train Epoch: (\d+)', line_)[0]
                    break

            break

    for i, line in enumerate(reverse_file):
        if line.find('Best Train Accuracies') != -1:

            model['Best Train Mean'] = re.findall("\d+\.\d+", line)[0]
            model['Best Train Old'] = re.findall("\d+\.\d+", line)[1]
            model['Best Train New'] = re.findall("\d+\.\d+", line)[2]

            for i_, line_ in enumerate(reverse_file[i:]):
                if 'Train Epoch' in line_:
                    model['Best Epoch'] = re.findall('Train Epoch: (\d+)', line_)[0]
                    break

            break


    data = pd.DataFrame([model])

    if verbose:
        print(data)

    if save:

        save_path = os.path.join(root_dir, save_name)
        data.to_pickle(save_path)

    else:

        return data


def parse_multiple_files(all_paths, rx_dict, root_dir=save_root_dir, save_name='test.pkl', verbose=True, save=False):

    all_data = []
    for path in all_paths:

        data = parse_out_file(path, rx_dict, save=False, verbose=False)
        all_data.append(data)

    all_data = pd.concat(all_data)
    save_path = os.path.join(root_dir, save_name)

    if save:
        all_data.to_pickle(save_path)

    if verbose:
        print(all_data)

    return all_data


save_dir = '/work/sagar/open_set_recognition/sweep_summary_files/ensemble_pkls'
base_path = '/work/sagar/osr_novel_categories/slurm_outputs/myLog-{}.out'

all_paths = [base_path.format(i) for i in ['407814_{}'.format(j) for j in range(11)]]

data = parse_multiple_files(all_paths, rx_dict, verbose=True, save=False, save_name='test.pkl')