
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from .Flows import *

class Flow_aff(nn.Module):
    def __init__(self, args):
        super(Flow_aff, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks
        self.affine = nn.ModuleList()


        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))

    def forward(self, future_enc, history_enc):
        B, F, K = history_enc.shape
        z_his = torch.randn(B, F, K).to(self.device)

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ =  log_det_a  # + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        # z_sample = history_enc
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample

class Flow_aff_conv(nn.Module):
    def __init__(self, args):
        super(Flow_aff_conv, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.convinv = nn.ModuleList()
        self.affine = nn.ModuleList()


        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.convinv.append(Invertible1x1Conv(feats_block))
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))

    def forward(self, future_enc, history_enc):
        B, F, K = history_enc.shape
        z_his = torch.randn(B, F, K).to(self.device)

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_c = self.convinv[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ = log_det_a   + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        # z_sample = history_enc
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.convinv[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample


class Flow_aff_act(nn.Module):
    def __init__(self, args):
        super(Flow_aff_act, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.affine = nn.ModuleList()
        self.actnorm = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))
            self.actnorm.append(ActNorm(feats_block))

    def forward(self, future_enc, history_enc):
        B, F, K = history_enc.shape
        z_his = torch.randn(B, F, K).to(self.device)

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_n = self.actnorm[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ = log_det_n / self.K + log_det_a  # + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        # z_sample = history_enc
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.actnorm[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample


class Flow_aff_ins(nn.Module):
    def __init__(self, args):
        super(Flow_aff_ins, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.instancenorm = nn.ModuleList()
        self.affine = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))
            self.instancenorm.append(GroupInstanceNorm(feats_block, group_len=4))

    def forward(self, future_enc, history_enc):
        B, F, K = history_enc.shape
        z_his = torch.randn(B, F, K).to(self.device)

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_n = self.instancenorm[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ = log_det_n / self.K + log_det_a  # + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        # z_sample = history_enc
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.instancenorm[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample



class Flow_aff_conv_act(nn.Module):
    def __init__(self, args):
        super(Flow_aff_conv_act, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.convinv = nn.ModuleList()
        self.affine = nn.ModuleList()
        self.actnorm = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.convinv.append(Invertible1x1Conv(feats_block))
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))
            self.actnorm.append(ActNorm(feats_block))

    def forward(self, future_enc, history_enc):
        B, F, K = history_enc.shape
        z_his = torch.randn(B, F, K).to(self.device)

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_n = self.actnorm[k](future_enc)
            future_enc, log_det_c = self.convinv[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ = log_det_n / self.K + log_det_a + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        # z_sample = history_enc
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.convinv[k].reverse(z_sample)
            z_sample = self.actnorm[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample

class Flow_aff_conv_ins(nn.Module):
    def __init__(self, args):
        super(Flow_aff_conv_ins, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.convinv = nn.ModuleList()
        self.instancenorm = nn.ModuleList()
        self.affine = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.convinv.append(Invertible1x1Conv(feats_block))
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))
            self.instancenorm.append(GroupInstanceNorm(feats_block, group_len=4))

    def forward(self, future_enc, history_enc):
        B, F, K = history_enc.shape
        z_his = torch.randn(B, F, K).to(self.device)

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_n = self.instancenorm[k](future_enc)
            future_enc, log_det_c = self.convinv[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ = log_det_n / self.K + log_det_a + log_det_c
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det

    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        # z_sample = history_enc
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.convinv[k].reverse(z_sample)
            z_sample = self.instancenorm[k].reverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample