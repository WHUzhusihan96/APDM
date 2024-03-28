from dataloader.dataloader import *
from models.net import *
from utils.config import *
import datetime
import argparse
import torch
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
            print("{} is used for training".format(args.model))
            self.G = VGGFc()
        elif 'alex' in args.model:
            print("{} is used for training".format(args.model))
            self.G = AlexNetFc()
        elif 'dense' in args.model:
            print("{} is used for training".format(args.model))
            self.G = denseFc()
        num_classes = len(args.source_classes)
        self.C = CLS(self.G.output_num(), num_classes)
        # if 'cdan' in args.train.mode:
        #     self.D = AdversarialNetwork(num_classes * self.C.output_num())
        # else:
        #     self.D = AdversarialNetwork(self.C.output_num())
        # self.C.apply(weights_init)
        # self.D.apply(weights_init)


if __name__ == '__main__':
    now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # load the source and target data
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = load_data(args, dataset)
    # load the network
    totalNet = TotalNet_new(args)
    G = totalNet.G.cuda().train()
    C = totalNet.C.cuda().train()
    # set the logdir
    root_dic = {0: 'AID', 1: 'CLRS', 2: 'MLRSN', 3: 'OPTIMAL-31'}
    task_dir = root_dic[args.data.dataset.source] + '_' + root_dic[args.data.dataset.target]

    # test phase
    if args.test.test_only:
        assert os.path.exists(args.test.resume_file)
        data = torch.load(open(args.test.resume_file, 'rb'))
        G.load_state_dict(data['G'])
        C.load_state_dict(data['C'])

        # source_test_dl, target_test_dl
        with TrainingModeManager([G, C],
                                 train=False) as mgr, \
                Accumulator(['feature', 'predict_prob', 'label']) as target_accumulator, \
                torch.no_grad():
            for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                im = im.cuda()
                label = label.cuda()

                feature = G.forward(im)
                feature_bt, predict_prob = C.forward(feature)
                predict_prob = nn.Softmax(-1)(predict_prob)

                for name in target_accumulator.names:
                    globals()[name] = variable_to_numpy(globals()[name])

                target_accumulator.updateData(globals())

        for x in target_accumulator:
            globals()[x] = target_accumulator[x]

        if args.test.if_save_mat:
            mat_dir = args.test.feature_file + task_dir + '_source.mat'
            # mat_dir = args.test.feature_file + task_dir + '_target.mat'
            io.savemat(mat_dir, {'data': feature, 'predict_prob': predict_prob, 'label': label})

        counters = AccuracyCounterA(len(args.source_classes))

        for (each_predict_prob, each_label) in zip(predict_prob, label):
            counters.add_total(each_label)
            each_pred_id = np.argmax(each_predict_prob)
            if each_pred_id == each_label:
                counters.add_correct(each_label)

        clear_output()
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

    # train phase
    if not args.test.test_only:
        log_dir = f'{args.log.root_dir}/{task_dir}/{now}'
        logger = SummaryWriter(log_dir)
        with open(join(log_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(save_config))

        scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75,
                                                                  max_iter=10000)
        optimizer_G = OptimWithSheduler(
            optim.SGD(G.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay,
                      momentum=args.train.momentum, nesterov=True), scheduler)
        optimizer_C = OptimWithSheduler(
            optim.SGD(C.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
                      momentum=args.train.momentum, nesterov=True), scheduler)

        global_step = 0
        best_acc = 0
        total_steps = tqdm(range(args.train.min_step), desc='global step')
        epoch_id = 0

        critic = Inter_pcd(C).cuda()
        cls_criterion = nn.CrossEntropyLoss()

        while global_step < args.train.min_step:

            iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ',
                         total=min(len(source_train_dl), len(target_train_dl)))
            epoch_id += 1

            for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

                label_source = label_source.cuda()
                im_source = im_source.cuda()
                im_target = im_target.cuda()

                f_s, f_t = G(im_source), G(im_target)
                feature_s, output_s = C(f_s)
                feature_t, output_t = C(f_t)

                cls_loss = cls_criterion(output_s, label_source)

                loss_st = critic(f_s, f_t, label_source)
                rho = global_step / args.train.min_step
                loss_ss = Intra_pcd(output_s, label_source)
                nuc_tgt = - torch.norm(F.softmax(output_t, dim=1), 'nuc') / output_t.shape[0]
                nuc_src = torch.norm(F.softmax(output_s, dim=1), 'nuc') / output_s.shape[0]
                loss = cls_loss - args.train.pcd_weight * rho * (loss_st - loss_ss) + args.train.aux_weight * (nuc_tgt + nuc_src)
                with OptimizerManager([optimizer_G, optimizer_C]):
                    loss.backward()

                global_step += 1
                total_steps.update()

                if global_step % args.log.log_interval == 0:
                    counter = AccuracyCounter()
                    counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(args.source_classes))),
                                        variable_to_numpy(nn.Softmax(-1)(output_s)))
                    acc_train = torch.tensor([counter.reportAccuracy()]).cuda()
                    logger.add_scalar('cls_loss', cls_loss, global_step)
                    logger.add_scalar('acc_train', acc_train, global_step)

                if global_step % args.test.test_interval == 0:

                    with TrainingModeManager([G, C],
                                             train=False) as mgr, \
                            Accumulator(['feature', 'soft_prob', 'label']) as target_accumulator, \
                            torch.no_grad():
                        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                            im = im.cuda()
                            label = label.cuda()

                            feature = G.forward(im)
                            feature, output = C.forward(feature)
                            soft_prob = nn.Softmax(-1)(output)

                            for name in target_accumulator.names:
                                globals()[name] = variable_to_numpy(globals()[name])

                            target_accumulator.updateData(globals())

                    for x in target_accumulator:
                        globals()[x] = target_accumulator[x]

                    counters = AccuracyCounterA(len(args.source_classes))

                    for (each_predict_prob, each_label) in zip(soft_prob, label):
                        counters.add_total(each_label)
                        each_pred_id = np.argmax(each_predict_prob)
                        if each_pred_id == each_label:
                            counters.add_correct(each_label)

                    clear_output()
                    print()
                    print('---counters---')
                    print(counters.Ncorrect)
                    print(counters.Ntotal)
                    print('each_accuracy')
                    print(counters.each_accuracy())
                    acc_test = counters.overall_accuracy()
                    correct = np.sum(counters.Ncorrect)
                    total = np.sum(counters.Ntotal)
                    print('correct:{}, total:{}, acc_test:{} '.format(correct, total, acc_test))
                    logger.add_scalar('acc_test', acc_test, global_step)
                    print('best_acc: ', best_acc)

                    data = {'G': G.state_dict(), 'C': C.state_dict()}

                    if acc_test > best_acc:
                        best_acc = acc_test
                        with open(join(log_dir, 'best.pkl'), 'wb') as f:
                            torch.save(data, f)

                    with open(join(log_dir, 'current.pkl'), 'wb') as f:
                        torch.save(data, f)
        print(log_dir)
