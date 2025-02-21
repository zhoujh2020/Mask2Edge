import torch
#from torch.optim import lr_scheduler
from models.optimizer.lr_scheduler import LR_Scheduler
class Optimizer():
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, net):
        net_parameters_id = {}

        for pname, p in net.efficientnet_b7.named_parameters():
            if 'efficientnet.weight' not in net_parameters_id:
                net_parameters_id['efficientnet.weight'] = []
            if 'efficientnet.bias' not in net_parameters_id:
                net_parameters_id['efficientnet.bias'] = []
            if 'weight' in pname:
                net_parameters_id['efficientnet.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['efficientnet.bias'].append(p)
        for pname, p in net.mask2former.named_parameters():
            if 'mask2former.weight' not in net_parameters_id:
                net_parameters_id['mask2former.weight'] = []
            if 'mask2former.bias' not in net_parameters_id:
                net_parameters_id['mask2former.bias'] = []
            if 'weight' in pname:
                net_parameters_id['mask2former.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['mask2former.bias'].append(p)
        for pname, p in net.unetPlusPlus.named_parameters():
            if 'unetPlusPlus.weight' not in net_parameters_id:
                net_parameters_id['unetPlusPlus.weight'] = []
            if 'unetPlusPlus.bias' not in net_parameters_id:
                net_parameters_id['unetPlusPlus.bias'] = []
            if 'weight' in pname:
                net_parameters_id['unetPlusPlus.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['unetPlusPlus.bias'].append(p)
        for pname, p in net.named_parameters():

            if pname in ['msblock0.conv.weight', 'msblock1.conv.weight', 'msblock2.conv.weight', 'msblock3.conv.weight', 'msblock4.conv.weight',
                         'msblock0.conv1.weight', 'msblock1.conv1.weight', 'msblock2.conv1.weight', 'msblock3.conv1.weight',
                         'msblock4.conv1.weight',
                         'msblock0.conv2.weight', 'msblock1.conv2.weight', 'msblock2.conv2.weight', 'msblock3.conv2.weight',
                         'msblock4.conv2.weight',
                         'msblock0.conv3.weight', 'msblock1.conv3.weight', 'msblock2.conv3.weight', 'msblock3.conv3.weight',
                         'msblock4.conv3.weight',
                         ]:

                if 'msblock_0-4.weight' not in net_parameters_id:
                    net_parameters_id['msblock_0-4.weight'] = []
                net_parameters_id['msblock_0-4.weight'].append(p)
            elif pname in ['msblock0.conv.bias', 'msblock1.conv.bias', 'msblock2.conv.bias',
                             'msblock3.conv.bias', 'msblock4.conv.bias',
                             'msblock0.conv1.bias', 'msblock1.conv1.bias', 'msblock2.conv1.bias',
                             'msblock3.conv1.bias',
                             'msblock4.conv1.bias',
                             'msblock0.conv2.bias', 'msblock1.conv2.bias', 'msblock2.conv2.bias',
                             'msblock3.conv2.bias',
                             'msblock4.conv2.bias',
                             'msblock0.conv3.bias', 'msblock1.conv3.bias', 'msblock2.conv3.bias',
                             'msblock3.conv3.bias',
                             'msblock4.conv3.bias',]:
                if 'msblock_0-4.bias' not in net_parameters_id:
                    net_parameters_id['msblock_0-4.bias'] = []
                net_parameters_id['msblock_0-4.bias'].append(p)
            elif pname in ['score_dsn0.weight','score_dsn1.weight','score_dsn2.weight',
                           'score_dsn3.weight','score_dsn4.weight']:
                if 'score_dsn_0-4.weight' not in net_parameters_id:
                    net_parameters_id['score_dsn_0-4.weight'] = []
                net_parameters_id['score_dsn_0-4.weight'].append(p)
            elif pname in ['score_dsn0.bias','score_dsn1.bias','score_dsn2.bias',
                           'score_dsn3.bias','score_dsn4.bias']:
                if 'score_dsn_0-4.bias' not in net_parameters_id:
                    net_parameters_id['score_dsn_0-4.bias'] = []
                net_parameters_id['score_dsn_0-4.bias'].append(p)
            elif pname in ['conv1.0.weight','conv1.1.weight','conv2.0.weight',
                           'conv2.1.weight','conv3.0.weight','conv3.1.weight',
                           'conv4.0.weight','conv4.1.weight','conv5.0.weight',
                           'conv5.1.weight']:
                if 'conv.weight' not in net_parameters_id:
                    net_parameters_id['conv.weight'] = []
                net_parameters_id['conv.weight'].append(p)
            elif pname in ['conv1.0.bias', 'conv1.1.bias', 'conv2.0.bias',
                           'conv2.1.bias', 'conv3.0.bias', 'conv3.1.bias',
                           'conv4.0.bias','conv4.1.bias', 'conv5.0.bias', 'conv5.1.bias']:
                if 'conv.bias' not in net_parameters_id:
                    net_parameters_id['conv.bias'] = []
                net_parameters_id['conv.bias'].append(p)

        for pname, p in net.attention.named_parameters():

            if 'attn.weight' not in net_parameters_id:
                net_parameters_id['attn.weight'] = []
            if 'attn.bias' not in net_parameters_id:
                net_parameters_id['attn.bias'] = []
            if 'weight' in pname:
                net_parameters_id['attn.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['attn.bias'].append(p)
        
        optim = torch.optim.AdamW([
                {'params': net_parameters_id['efficientnet.weight']      , 'lr': self.cfg.lr*1, 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['efficientnet.bias']        , 'lr': self.cfg.lr*2, 'weight_decay': 0.},
                {'params': net_parameters_id['mask2former.weight'], 'lr': self.cfg.lr*1., 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['mask2former.bias'], 'lr': self.cfg.lr*2., 'weight_decay': 0.},
                {'params': net_parameters_id['unetPlusPlus.weight'], 'lr': self.cfg.lr * 1., 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['unetPlusPlus.bias'], 'lr': self.cfg.lr * 2., 'weight_decay': 0.},
                {'params': net_parameters_id['msblock_0-4.weight'], 'lr': self.cfg.lr*1 , 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['msblock_0-4.bias']  , 'lr': self.cfg.lr*2  , 'weight_decay': 0.},
                {'params': net_parameters_id['score_dsn_0-4.weight'], 'lr': self.cfg.lr*0.01 , 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['score_dsn_0-4.bias']  , 'lr': self.cfg.lr*0.02 , 'weight_decay': 0.},
                {'params': net_parameters_id['attn.weight']  , 'lr': self.cfg.lr*1, 'weight_decay': self.cfg.wd}, # 1
                {'params': net_parameters_id['attn.bias']    , 'lr': self.cfg.lr*2, 'weight_decay': 0.}, # 2
            ], lr=self.cfg.lr, weight_decay=self.cfg.wd)

        scheduler = LR_Scheduler('poly', self.cfg.lr, self.cfg.max_epoch, 7200)
        #scheduler = lr_scheduler.StepLR(optim, step_size=self.cfg.stepsize, gamma=self.cfg.gamma)

        return optim, scheduler
