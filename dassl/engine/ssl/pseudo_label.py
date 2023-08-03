import torch
import torch.nn as nn
import torch.nn.functional as F
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy

@TRAINER_REGISTRY.register()
class PseudoLabel(TrainerXU):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u, label_u = self.parse_batch_train(batch_x, batch_u)
        label_u[:] = -1
        target = torch.cat([label_x, label_u], 0)
        unlabeled_mask = (target == -1).float()
        
        # print("label_u:, len(label_u):", label_u, len(label_u))
        # print("target:, len(target):", target, len(target))
        # print("unlabeled_mask:, len(unlabeled_mask):", unlabeled_mask, len(unlabeled_mask))
        # exit(0)

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        else:
            self.model = self.model

        inputs = torch.cat([input_x, input_u], 0)
        outputs = self.model(inputs)
        loss_x = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()

        # self.model.update_batch_stats(False) # all error, if del the loss_u always 0   
        y_hat = self.model(inputs)        
        # self.model.update_batch_stats(True)
	
        loss_u = (F.mse_loss(y_hat.softmax(1), outputs.detach().softmax(1).detach(), reduction="none").mean(1) * unlabeled_mask).mean()
        loss = loss_x + loss_u
        self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "acc_x": compute_accuracy(outputs, target)[0].item(),
            "loss_u": loss_u.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]
        # label_u is used only for evaluating pseudo labels' accuracy
        label_u = batch_u["label"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, label_x, input_u, label_u

