import math
import os
import os.path as osp

import gdown
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from src.nafnet.archs import NAFNet, NAFNetLocal, NAFSSR


class BaseModel:
    """Base model for inference."""

    def __init__(self, opt):
        self.opt = opt
        # Set device to CUDA if available and num_gpu is not 0, otherwise CPU
        self.device = torch.device('cuda' if opt.get('num_gpu', 0) != 0 and torch.cuda.is_available() else 'cpu')

    def feed_data(self, data):
        pass

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters',
                                                  False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)


def define_network(opt):
    network_type = opt.pop("type")
    if network_type == "NAFNet":
        net = NAFNet(**opt)
    elif network_type == "NAFNetLocal":
        net = NAFNetLocal(**opt)
    elif network_type == "NAFSSR":
        net = NAFSSR(**opt)
    else:
        raise ValueError(f'{network_type} is not found.')

    return net


class ImageRestorationModel(BaseModel):
    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.net_g.eval()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        gdrive_id = self.opt['path'].get('pretrain_network_g_gdrive_id', None)
        if load_path:
            model_dir = osp.dirname(load_path)
            os.makedirs(model_dir, exist_ok=True)

            if not osp.exists(load_path) and gdrive_id:
                print(f"Downloading model from Google Drive (ID: {gdrive_id}) to {load_path}...")
                gdown.download(id=gdrive_id, output=load_path, quiet=False)

            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        self.scale = int(opt['scale'])

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def test(self):
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = n
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
