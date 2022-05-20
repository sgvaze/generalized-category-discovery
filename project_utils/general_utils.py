import os
import torch
import random
import numpy as np
import inspect

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from datetime import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def get_dino_head_weights(pretrain_path):

    """
    :param pretrain_path: Path to full DINO pretrained checkpoint as in https://github.com/facebookresearch/dino
     'full_ckpt'
    :return: weights only for the projection head
    """

    all_weights = torch.load(pretrain_path)

    head_state_dict = {}
    for k, v in all_weights['teacher'].items():
        if 'head' in k and 'last_layer' not in k:
            head_state_dict[k] = v

    head_state_dict = strip_state_dict(head_state_dict, strip_key='head.')

    # Deal with weight norm
    weight_norm_state_dict = {}
    for k, v in all_weights['teacher'].items():
        if 'last_layer' in k:
            weight_norm_state_dict[k.split('.')[2]] = v

    linear_shape = weight_norm_state_dict['weight'].shape
    dummy_linear = torch.nn.Linear(in_features=linear_shape[1], out_features=linear_shape[0], bias=False)
    dummy_linear.load_state_dict(weight_norm_state_dict)
    dummy_linear = torch.nn.utils.weight_norm(dummy_linear)

    for k, v in dummy_linear.state_dict().items():

        head_state_dict['last_layer.' + k] = v

    return head_state_dict

def transform_moco_state_dict(obj, num_classes):

    """
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        if k.startswith("fc.2"):
            continue

        if k.startswith("fc.0"):
            k = k.replace("0.", "")
            if "weight" in k:
                v = torch.randn((num_classes, v.size(1)))
            elif "bias" in k:
                v = torch.randn((num_classes,))

        newmodel[k] = v

    return newmodel


def init_experiment(args, runner_name=None, exp_id=None):

    args.cuda = torch.cuda.is_available()

    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        # Unique identifier for experiment
        now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)
        while os.path.exists(log_dir):
            now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                  datetime.now().strftime("%S.%f")[:-3] + ')'

            log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')

    args.writer = SummaryWriter(log_dir=args.log_dir)

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    print(runner_name)
    print(args)

    return args


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ClassificationPredSaver(object):

    def __init__(self, length, save_path=None):

        if save_path is not None:

            # Remove filetype from save_path
            save_path = save_path.split('.')[0]
            self.save_path = save_path

        self.length = length

        self.all_preds = None
        self.all_labels = None

        self.running_start_idx = 0

    def update(self, preds, labels=None):

        # Expect preds in shape B x C

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()

        b, c = preds.shape

        if self.all_preds is None:
            self.all_preds = np.zeros((self.length, c))

        self.all_preds[self.running_start_idx: self.running_start_idx + b] = preds

        if labels is not None:
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()

            if self.all_labels is None:
                self.all_labels = np.zeros((self.length,))

            self.all_labels[self.running_start_idx: self.running_start_idx + b] = labels

        # Maintain running index on dataset being evaluated
        self.running_start_idx += b

    def save(self):

        # Softmax over preds
        preds = torch.from_numpy(self.all_preds)
        preds = torch.nn.Softmax(dim=-1)(preds)
        self.all_preds = preds.numpy()

        pred_path = self.save_path + '.pth'
        print(f'Saving all predictions to {pred_path}')

        torch.save(self.all_preds, pred_path)

        if self.all_labels is not None:

            # Evaluate
            self.evaluate()
            torch.save(self.all_labels, self.save_path + '_labels.pth')

    def evaluate(self):

        topk = [1, 5, 10]
        topk = [k for k in topk if k < self.all_preds.shape[-1]]
        acc = accuracy(torch.from_numpy(self.all_preds), torch.from_numpy(self.all_labels), topk=topk)

        for k, a in zip(topk, acc):
            print(f'Top{k} Acc: {a.item()}')


def get_acc_auroc_curves(logdir):

    """
    :param logdir: Path to logs: E.g '/work/sagar/open_set_recognition/methods/ARPL/log/(12.03.2021_|_32.570)/'
    :return:
    """

    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Only gets scalars
    log_info = {}
    for tag in event_acc.Tags()['scalars']:

        log_info[tag] = np.array([[x.step, x.value] for x in event_acc.scalars._buckets[tag].items])

    return log_info


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


class IndicatePlateau(object):

    def __init__(self, threshold=5e-4, patience_epochs=5, mode='min', threshold_mode='rel'):

        self.patience = patience_epochs
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)

        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            print('Tracked metric has plateaud')
            self._reset()
            return True
        else:
            return False

    def is_better(self, a, best):

        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):

        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


if __name__ == '__main__':

    x = IndicatePlateau(threshold=0.0899)
    eps = np.arange(0, 2000, 1)
    y = np.exp(-0.01 * eps)

    print(y)
    for i, y_ in enumerate(y):

        z = x.step(y_)
        if z:
            print(f'Plateaud at epoch {i} with val {y_}')