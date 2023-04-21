"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.models.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate
from lib.models.basemodel import BaseModel
from lib.evaluate import recall_

##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, data):
        super(Ganomaly, self).__init__(opt, data)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.ids = torch.empty(size=(self.opt.batchsize, 2), dtype=torch.long, device=self.device)
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.opt.w_adv * self.l_adv(self.feat_fake, self.feat_real)
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)
        self.err_g_lat = self.opt.w_lat * self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()
    
    ##
    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            data ([type]): data for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.ids_total = torch.zeros(size=(len(self.data.valid.dataset), 2), dtype=torch.long, device=self.device)
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()

                self.ids_total[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.ids.reshape(
                    error.size(0), 2)
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.jpg' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.jpg' % (dst, i+1), normalize=True)
                    if self.opt.save_test_single:
                        dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'single_images')
                        if not os.path.isdir(dst): os.makedirs(dst)
                        for idx in range(real.size()[0]):
                            if self.opt.dataset == 'GAP':
                                p1 = self.ids[idx, 0, 1].item()
                                p2 = self.ids[idx, 0, 0].item()
                                file_idx = "{}_{}".format(p2, p1)
                            else:
                                with open(os.path.join(os.path.join(self.opt.dataroot, 'test'), 'test_files_key.pkl'), 'rb') as f:
                                    test_key = pickle.load(f)
                                    file_idx = test_key[self.ids[idx, 0, 0].item()][-1][:-4]
                            vutils.save_image(real[idx], '{}/REAL_{}.jpg'.format(dst, file_idx), normalize=True)
                            vutils.save_image(fake[idx], '{}/FAKE_{}.jpg'.format(dst, file_idx), normalize=True)
                            if self.opt.save_res:
                                dst2 = os.path.join(self.opt.outf, self.opt.name, 'test', 'res')
                                if not os.path.isdir(dst2): os.makedirs(dst2)
                                res = self.residuals(fake[idx], real[idx])
                                np.save('{}/RES_{}.npy'.format(dst2, file_idx), res, allow_pickle=True)
                                if self.opt.save_residual_img:
                                    dst3 = os.path.join(self.opt.outf, self.opt.name, 'test', 'res_images')
                                    if not os.path.isdir(dst3): os.makedirs(dst3)
                                    threshold = np.quantile(res, 0.75)
                                    res_mask = np.ma.masked_where(res > threshold, res)
                                    real_im = plt.imread('{}/REAL_{}.jpg'.format(dst, file_idx))
                                    plt.imshow(real_im)
                                    plt.imshow(res_mask, cmap="hsv", interpolation="none", clim=[0.8, 1.0], alpha=0.5)
                                    plt.axis('off')
                                    plt.savefig('{}/RES_{}.jpg'.format(dst3, file_idx), dpi=600)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            scores = OrderedDict([("ids", self.ids_total.cpu().detach().numpy()),
                                  ("gt_labels", self.gt_labels.cpu().detach().numpy()),
                                  ("an_scores", self.an_scores.cpu().detach().numpy())])

            # auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            rec, pre, acc, f1 = recall_(scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('Recall', rec), ('Precision', pre),
                                       ('Accuracy', acc), ('F1-score', f1)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance, scores
