import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
import mmcv
from mmcv import Config
from mmcv import digit_version as dv
from mmcv import load
from mmcv.cnn import fuse_conv_bn
from mmcv.engine import collect_results_cpu

from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from protogcn.datasets import build_dataloader, build_dataset
from protogcn.models import build_model
from protogcn.utils import cache_checkpoint, mc_off, mc_on, test_port

import pickle
import shutil
import tempfile
import time
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
from mmcv.runner import get_dist_info

import sys
import numpy as np


def multi_gpu_test(model: nn.Module,
                   data_loader: DataLoader,
                   tmpdir: Optional[str] = None,
                   gpu_collect: bool = False) -> Optional[list]:
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    # optional: save graphs for visualization
    # save_graph = []

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data) 
            # result, get_graph = model(return_loss=False, **data)
        results.extend(result)
        # save_graph.append(get_graph)
        
        if rank == 0:
            batch_size = len(result)
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        result_from_ranks = collect_results_gpu(results, len(dataset))
    else:
        result_from_ranks = collect_results_cpu(results, len(dataset), tmpdir)

    # graph_data = np.array(save_graph)
    # np.save('graph.npy', graph_data)
    
    return result_from_ranks


def parse_args():
    parser = argparse.ArgumentParser(
        description='protogcn test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['top_k_accuracy', 'mean_class_accuracy'],
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def inference_pytorch(args, cfg, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # build the model and load checkpoint
    model = build_model(cfg.model)
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    if args.checkpoint is None:
        work_dir = cfg.work_dir
        args.checkpoint = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint)

    args.checkpoint = cache_checkpoint(args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    return outputs


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    out = osp.join(cfg.work_dir, 'result.pkl') if args.out is None else args.out

    # Load eval_config from cfg
    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)
    if args.eval:
        eval_cfg['metrics'] = args.eval

    mmcv.mkdir_or_exist(osp.dirname(out))
    _, suffix = osp.splitext(out)
    assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    if rank == 0 and memcached:
        # mc_list is a list of pickle files you want to cache in memory.
        # Basically, each pickle file is a dictionary.
        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    dist.barrier()
    outputs = inference_pytorch(args, cfg, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        print(f'\nwriting results to {out}')
        dataset.dump_results(outputs, out=out)
        if eval_cfg:
            eval_res = dataset.evaluate(outputs, **eval_cfg)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')

    dist.barrier()
    if rank == 0 and memcached:
        mc_off()


if __name__ == '__main__':
    main()
