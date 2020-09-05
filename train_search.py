import random

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--search_space', type=str, default='darts', help='nasp|darts')
args = parser.parse_args()

args.train_batch_size = args.batch_size
args.valid_batch_size = args.batch_size

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'), exec_script='search.sh')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

CIFAR_CLASSES = 10
if args.set=='cifar100':
    CIFAR_CLASSES = 100


def fix_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  cudnn.benchmark = False
  torch.manual_seed(seed)
  cudnn.enabled = True
  cudnn.deterministic = True
  torch.cuda.manual_seed(seed)


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  fix_seed(args.seed)

  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, space=args.search_space)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_queue, train_sampler, valid_queue = utils.get_train_validation_loader(args)

  test_queue = utils.get_test_loader(args)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  best_acc = 0
  total_train_time, total_valid_time, total_test_time = 0, 0, 0
  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    architect.alpha_forward = 0
    architect.alpha_backward = 0
    start_time = time.time()
    train_acc, train_obj, alphas_time, forward_time, backward_time = \
      train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)
    end_time = time.time()
    search_time = end_time - start_time
    total_train_time += search_time
    logging.info("train time %f", end_time - start_time)
    logging.info("alphas_time %f ", alphas_time)
    logging.info("forward_time %f", forward_time)
    logging.info("backward_time %f", backward_time)
    logging.info("alpha_forward %f", architect.alpha_forward)
    logging.info("alpha_backward %f", architect.alpha_backward)
    logging.info('train_acc %f', train_acc)

    # validation
    # if args.epochs-epoch<=1:
    #   valid_acc, valid_obj = infer(valid_queue, model, criterion)
    #   logging.info('valid_acc %f', valid_acc)

    # utils.save(model, os.path.join(args.save, 'weights.pt'))
    start_time2 = time.time()
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    end_time2 = time.time()
    valid_time = end_time2 - start_time2
    total_valid_time += valid_time
    logging.info("inference time %f", end_time2 - start_time2)
    logging.info('valid_acc %f', valid_acc)

    # test
    start = time.time()
    test_acc, test_obj = infer(test_queue, model, criterion)
    end = time.time()
    test_time = end - start
    total_test_time += test_time
    logging.info("inference time %f", end - start)
    logging.info('test_acc %f, test_obj %f', test_acc, test_obj)

    # update learning rate
    scheduler.step()

    is_best = valid_acc > best_acc
    best_acc = max(valid_acc, best_acc)
    if is_best:
      logging.info('best valid_acc: {} at epoch: {}, test_acc: {}'.format(
        best_acc, epoch, test_acc
      ))
      logging.info('Current best genotype = {}'.format(model.genotype()))
  return total_train_time, total_valid_time, total_test_time


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  alphas_time = 0
  forward_time = 0
  backward_time = 0
  begin = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    try:
      input_search, target_search = next(valid_queue_iter)
    except:
      valid_queue_iter = iter(valid_queue)
      input_search, target_search = next(valid_queue_iter)
    input_search = input_search.cuda(non_blocking=True)
    target_search = target_search.cuda(non_blocking=True)

    if epoch>=15:
      begin1 = time.time()
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
      end1 = time.time()
      alphas_time += end1 - begin1

    optimizer.zero_grad()
    begin.record()
    logits = model(input)
    loss = criterion(logits, target)
    end.record()
    forward_time += utils.get_elaspe_time(begin, end)

    begin.record()
    end.record()
    loss.backward()
    backward_time += utils.get_elaspe_time(begin, end)

    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg, alphas_time, forward_time, backward_time


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('%s %03d %e %f %f', valid_queue.name, step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  root = logging.getLogger()
  begin = time.time()
  total_train_time, total_valid_time, total_test_time = main()
  end = time.time()
  root.info('total search time: {} s'.format(end - begin))
  root.info('total train time: {} s'.format(total_train_time))
  root.info('total valid time: {} s'.format(total_valid_time))
  root.info('total test time: {} s'.format(total_test_time))


