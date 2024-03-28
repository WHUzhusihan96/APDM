from dataloader.dataloader import *
from models.net import *
from utils.config import *
import datetime
import argparse
import torch
import time
from utils.lib import *
from tqdm import tqdm
import numpy as np
from torch import optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import os
from scipy import io
cudnn.benchmark = True
cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')
args_config = parser.parse_args()
config_file = args_config.config

args, dataset, save_config = load_config_file(config_file)
# seed_everything()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


class TotalNet_new(nn.Module):
    def __init__(self, args):
        super(TotalNet_new, self).__init__()
        if 'res' in args.model:
            print("{} is used for training".format(args.model))
            self.G = ResNetFc(args.model)
        elif 'vgg' in args.model:
            print("currently denied")
        elif 'alex' in args.model:
            print("{} is used for training".format(args.model))
            self.G = AlexNetFc()
        num_classes = len(args.source_classes)
        self.C = CLS(self.G.output_num(), num_classes)
        self.D = AdversarialNetwork(self.C.output_num())
        # self.C.apply(weights_init)
        # self.D.apply(weights_init)


if __name__ == '__main__':
    # print(args)
    now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # load the source and target data
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = load_data(args, dataset)
    # load the network
    totalNet = TotalNet_new(args)
    G = totalNet.G.cuda().train()
    C = totalNet.C.cuda().train()
    # D = totalNet.D.cuda().train()
    # set the logdir
    root_dic = {0: 'AID', 1: 'CLRS', 2: 'MLRSN', 3: 'OPTIMAL-31'}
    task_dir = root_dic[args.data.dataset.source] + '_' + root_dic[args.data.dataset.target]

    # test phase
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    G.load_state_dict(data['G'])
    C.load_state_dict(data['C'])
    # D.load_state_dict(data['D'])

    # source_test_dl, target_test_dl
    st = time.time()
    with TrainingModeManager([G, C], train=False) as mgr, \
            Accumulator(['feature', 'bt_feature', 'predict_prob', 'label']) as target_accumulator, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
        # for i, (im, label) in enumerate(tqdm(source_test_dl, desc='testing ')):
            im = im.cuda()
            label = label.cuda()

            feature = G.forward(im)
            bt_feature, predict_prob = C.forward(feature)
            predict_prob = nn.Softmax(-1)(predict_prob)

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())
    et = time.time()
    print("训练时间为: ", et-st, "秒")
    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    if args.test.if_save_mat:
        mat_dir = args.test.feature_file + task_dir + '_source.mat'
        # mat_dir = args.test.feature_file + task_dir + '_target.mat'
        # mat_dir = args.test.feature_file + task_dir + '_source_current.mat'
        # mat_dir = args.test.feature_file + task_dir + '_target_current.mat'
        io.savemat(mat_dir, {'data': feature, 'bt_feature': bt_feature, 'predict_prob': predict_prob, 'label': label})

    counters = AccuracyCounterA(len(args.source_classes))

    for (each_predict_prob, each_label) in zip(predict_prob, label):
        counters.add_total(each_label)
        each_pred_id = np.argmax(each_predict_prob)
        if each_pred_id == each_label:
            counters.add_correct(each_label)

    # clear_output()
    print()
    print('---counters---')
    print(counters.Ncorrect)
    print(counters.Ntotal)
    print('each_accuracy')
    print(counters.each_accuracy())
    acc_test = counters.overall_accuracy()
    print('acc_test: ', acc_test)
    print(task_dir)
    exit(0)

