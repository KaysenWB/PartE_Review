import torch
import torch.nn as nn
from math import log, pi
from .Flows import *



class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.K = args.K

    def glow(self, z, log_det):

        log_p = -0.5 * log(2*pi) - 0.5 * (z ** 2)
        log_p_sum = torch.sum(log_p)
        loss = - (log_p_sum  + log_det)
        loss = loss/(z.size(0)*z.size(1)*z.size(2))

        return loss

    def traj(self, pred_traj, fur):

        fur = fur.unsqueeze(2).repeat(1, 1, self.K, 1)
        traj_rmse = torch.sqrt(torch.sum((pred_traj - fur) ** 2, dim=-1) + 1e-8).sum(dim=1)
        best_idx = torch.argmin(traj_rmse, dim=1)
        loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()

        return loss_traj




class RealNVP_tra(nn.Module):
    def __init__(self, args):
        super(RealNVP_tra, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.K = args.K

        self.glow = RealNVP(args)

        self.fc_in_his = nn.Sequential(
            nn.Linear(self.feats_in*self.obs_length, self.feats_hidden),
            nn.ReLU())
        self.fc_in_fur = nn.Sequential(
            nn.Linear(self.feats_in*self.pred_length, self.feats_hidden),
            nn.ReLU())

        #self.decoder = nn.LSTM(self.feats_hidden, self.feats_hidden)
        self.pred_next = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.fc_out = nn.Linear(self.feats_hidden, self.feats_out)
        self.loss = Loss(args)


    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :self.feats_in].permute(1, 0, 2)
        his_enc = inputs[0][:self.obs_length, :, :self.feats_in]
        fut_enc = inputs[0][self.obs_length:, :, :self.feats_in]
        B = his_enc.shape[1]

        # emb
        his_enc = self.fc_in_his(his_enc.permute(1, 0, 2).contiguous().view(B, -1))
        fut_enc = self.fc_in_fur(fut_enc.permute(1, 0, 2).contiguous().view(B, -1))
        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)
        fut_enc = fut_enc.unsqueeze(2).repeat(1, 1, self.K)

        # train
        if iftrain:
            Z, log_det = self.glow(fut_enc, his_enc)
            loss_flow = self.loss.glow(Z, log_det)
        else:
            loss_flow = 0

        # predicting
        dec = self.glow.reverse(his_enc)
        traj = [dec.permute(0, 2, 1)]
        for i in range(self.pred_length):
            pred = self.pred_next(traj[-1])
            traj.append(pred)
        '''pred_traj = torch.stack(traj)[-self.pred_length:]
        dec = dec.permute(0, 2, 1).contiguous().view(-1, self.feats_hidden)
        dec = dec.unsqueeze(0).repeat(self.pred_length, 1, 1)
        pred_traj = self.decoder(dec)[0]
        pred_traj = pred_traj.view(self.pred_length, -1, self.K, self.feats_hidden).contiguous().permute(1, 0, 2, 3)
        pred_traj = self.fc_out(pred_traj)'''

        # cal loss
        loss_traj = self.loss.traj(pred_traj.permute(1,0,2,3), fut)
        loss_dict = loss_flow * 0.5 + loss_traj
        return pred_traj, loss_dict

