#!/user/bin/python
# -*- encoding: utf-8 -*-

from os.path import join

class Config(object):
    def __init__(self):
        self.data = "bsds" # NYUD
        # ============== training
        # self.resume = "./pretrained/epoch-17-checkpoint.pth"
        self.resume = False
        self.msg_iter = 100
        self.gpu = '0'
        self.save_pth = join("./output", self.data)
        self.pretrained = "./pretrained/vgg16.pth"
        self.aug = True

        # ============== testing
        self.multi_aug = False # Produce the multi-scale results
        self.side_edge = False  # Output the side edges

        # ================ dataset
        self.dataset = "./data/{}".format(self.data)

        # =============== optimizer
        self.batch_size = 8
        self.lr = 1e-4 
        self.momentum = 0.9
        self.wd = 5e-2
        self.stepsize = 5
        self.gamma = 0.1
        self.max_epoch = 18
        self.itersize = 10

