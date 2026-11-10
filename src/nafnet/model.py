import os.path as osp

import torch
from collections import OrderedDict
from copy import deepcopy

from ..config import get_device
from ..model_store import ensure_model
from .archs import NAFNet, NAFNetLocal, NAFSSR


class BaseModel:
    """Base model for inference."""

    def __init__(self, opt, device=None):
        """
        Args:
            opt (dict): Options. `num_gpu: 0` in the option file (legacy) still forces CPU.
            device (str, optional): "cpu", "cuda" or None for automatic (argument > $IDCARD_DEVICE > auto).
        """
        self.opt = opt
        if device is None and opt.get('num_gpu', 1) == 0:
            device = 'cpu'
        self.device = torch.device(get_device(device))

    def feed_data(self, data):
        pass

    def model_to_device(self, net):
        """Move the network to the selected device (inference only: no multi-GPU wrappers)."""
        return net.to(self.device)

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
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage, weights_only=True)
        if param_key is not None:
            load_net = load_net[param_key]
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
    def __init__(self, opt, device=None):
        super(ImageRestorationModel, self).__init__(opt, device=device)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.net_g.eval()

        # load pretrained models: `pretrain_model` (kunci manifest, diunduh/diverifikasi oleh ensure_model)
        # atau `pretrain_network_g` (berkas lokal eksplisit, harus sudah ada)
        path_opt = self.opt['path']
        load_path = path_opt.get('pretrain_network_g', None)
        if not load_path and path_opt.get('pretrain_model'):
            load_path = str(ensure_model(path_opt['pretrain_model']))
        if load_path:
            if not osp.isfile(load_path):
                raise FileNotFoundError(f"Bobot NAFNet tidak ditemukan: {load_path}. "
                                        f"Gunakan `pretrain_model` (kunci manifest) agar diunduh otomatis.")
            self.load_network(self.net_g, load_path,
                              path_opt.get('strict_load_g', True),
                              param_key=path_opt.get('param_key', 'params'))

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
