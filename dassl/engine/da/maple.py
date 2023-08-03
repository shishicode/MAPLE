import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU

from dassl.metrics import compute_accuracy
from dassl.modeling import build_head
from dassl.modeling.ops import ReverseGrad

from dassl.data.transforms import build_transform
from dassl.data import DataManager

from itertools import combinations

@TRAINER_REGISTRY.register()
class MAPLE(TrainerXU):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.ENTMIN.LMDA  
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        # to do semi_domain
        self.softmax = nn.Softmax()  # different 
        self.weight_u = cfg.TRAINER.FIXMATCH.WEIGHT_U
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE
        # split_batch
        self.split_batchsize = cfg.DATALOADER.TRAIN_U.BATCH_SIZE // cfg.DATALOADER.TRAIN_U.N_DOMAIN

        # combination of targets
        if cfg.DATALOADER.TRAIN_U.N_DOMAIN > 1:
            self.target_combinations = list(combinations(range(cfg.DATALOADER.TRAIN_U.N_DOMAIN), 2))
        else:
            self.target_combinations = []

        self.n_target_combinations = len(self.target_combinations)

        self.build_critic()

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS) > 0
    
    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.lab2cname = self.dm.lab2cname

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {
            "acc_thre": acc_thre,
            "acc_raw": acc_raw,
            "keep_rate": keep_rate
        }
        return output

    def build_critic(self):
        cfg = self.cfg

        print("Building critic network")
        # fdim = self.model.fdim
        # to do domain
        if isinstance(self.model, nn.DataParallel):
            fdim = self.model.module.fdim
        else:
            fdim = self.model.fdim

        self.critic = []        
        # source-target
        for _ in range(self.cfg.DATALOADER.TRAIN_U.N_DOMAIN):
            critic_body = build_head(
                "mlp",
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=[fdim, fdim],
                activation="leaky_relu"
            )
            self.critic.append(nn.Sequential(critic_body, nn.Linear(fdim, 1)))
        
        # target-target
        for _ in range(self.n_target_combinations):
            critic_body = build_head(
                "mlp",
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=[fdim, fdim],
                activation="leaky_relu"
            )
            self.critic.append(nn.Sequential(critic_body, nn.Linear(fdim, 1)))
        self.critic = nn.ModuleList(self.critic)

        print("# params: {:,}".format(count_num_param(self.critic)))
        self.critic.to(self.device)
        self.optim_c = build_optimizer(self.critic, cfg.OPTIM)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM)
        self.register_model("critic", self.critic, self.optim_c, self.sched_c)
        self.revgrad = ReverseGrad()


    def forward_backward(self, batch_x, batch_u, coef):
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2, label_u = parsed_data
        input_u = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)
        n_x = input_x.size(0)

        # Generate pseudo labels
        with torch.no_grad():
            output_u = self.softmax(self.model(input_u))
            max_prob, label_u_pred = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()

            # Evaluate pseudo labels' accuracy
            y_u_pred_stats = self.assess_y_pred_quality(
                label_u_pred[n_x:], label_u, mask_u[n_x:]
            )
        
        # Unsupervised loss
        output_u = self.model(input_u2)
        loss_u = F.cross_entropy(output_u, label_u_pred, reduction="none")
        loss_u = (loss_u * mask_u).mean()


        domain_x = torch.ones(self.split_batchsize, 1).to(self.device)
        domain_u = torch.zeros(self.split_batchsize, 1).to(self.device)

        # show progress
        global_step = self.batch_idx + self.epoch * self.num_batches
        progress = global_step / (self.max_epoch * self.num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1

        # lable data CrossEntropyLoss,Supervised loss
        logit_x, feat_x = self.model(input_x, return_feature=True)
        loss_x = self.ce(logit_x, label_x)

        _, feat_u = self.model(input_u, return_feature=True)
        # domain lable
        feat_x = self.revgrad(feat_x, grad_scaling=lmda)
        feat_u = self.revgrad(feat_u, grad_scaling=lmda)

        loss_d = 0.

        # source-target
        for i in range(self.cfg.DATALOADER.TRAIN_U.N_DOMAIN):
            st_critic = self.critic[i]
            feat_xi = feat_x[i*self.split_batchsize:(i+1)*self.split_batchsize]
            feat_ui = feat_u[i*self.split_batchsize:(i+1)*self.split_batchsize]
            output_xd = st_critic(feat_xi)  # 1
            output_ud = st_critic(feat_ui)  # 0
            loss_d += (self.bce(output_xd, domain_x) + self.bce(output_ud, domain_u))

        # target-target
        for i, (target_i, target_j) in enumerate(self.target_combinations):
            tt_critic = self.critic[self.cfg.DATALOADER.TRAIN_U.N_DOMAIN + i]
            feat_ti = feat_u[target_i*self.split_batchsize:(target_i+1)*self.split_batchsize]
            feat_tj = feat_u[target_j*self.split_batchsize:(target_j+1)*self.split_batchsize]
            output_ti = tt_critic(feat_ti)  # 1
            output_tj = tt_critic(feat_tj)  # 0
            loss_d += (self.bce(output_ti, domain_x) + self.bce(output_tj, domain_u))

        # loss = loss_x + loss_d 
        # to do semi_domain
        loss_d = loss_d / (self.cfg.DATALOADER.TRAIN_U.N_DOMAIN + self.n_target_combinations)
        loss = loss_x + loss_d * coef* 0.8 + loss_u * 1 
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss:" : loss.item(),
            "loss_x": loss_x.item(),
            "acc_x": compute_accuracy(logit_x, label_x)[0].item(),
            "loss_d": loss_d.item(),
            "loss_u": loss_u.item(),
            "y_u_pred_acc_raw": y_u_pred_stats["acc_raw"],
            "y_u_pred_acc_thre": y_u_pred_stats["acc_thre"],
            "y_u_pred_keep": y_u_pred_stats["keep_rate"]
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        input_x2 = batch_x["img2"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]
        input_u2 = batch_u["img2"]
        # label_u is used only for evaluating pseudo labels' accuracy
        label_u = batch_u["label"]

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, input_x2, label_x, input_u, input_u2, label_u
