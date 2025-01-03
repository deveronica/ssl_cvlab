import torch
import torch.nn.functional as F
import torch.distributed as dist

def ce_loss(logits, targets, use_hard_labels=True, reduction='mean'):
    if not use_hard_labels:
        raise ValueError("Soft labels are not supported yet")
    
    log_pred = F.log_softmax(logits, dim=-1)
    targets = targets.view(-1, 1)
    loss = -log_pred.gather(1, targets).view(-1)
    
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError("Invalid reduction type specified")

def consistency_loss(logits_s, logits_w, p_cutoff=0.98, use_hard_labels=True):
    pseudo_label = F.softmax(logits_w.detach(), dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = (max_probs >= p_cutoff).float()
    select = mask.long()
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, reduction='none') * mask
    else:
        raise ValueError('Hard labels must be used.')
    return masked_loss.mean(), mask.mean(), select, max_idx

def reduce_tensor(tensor, mean=True):
    # Assuming 'tensor' is the output of a loss function in a distributed setting.
    ts = tensor.clone()
    dist.all_reduce(ts, op=dist.ReduceOp.SUM)
    if mean:
        ts = ts / dist.get_world_size()
    return ts

def cal_topK(pred_s, pred_w, topk_range=(1, )):
    # Compute the accuracy over the k range and return the k value where the accuracy exceeds a threshold
    target_w = torch.argmax(pred_w, dim=-1)
    maxk = topk_range[1]
    batch_size = target_w.size(0)

    if maxk > pred_s.size(1):
        maxk = pred_s.size(1)
    
    _, pred = pred_s.topk(maxk, 1, True, True)
    correct = pred.eq(target_w.view(-1, 1).expand_as(pred))
    for k in range(topk_range[0], topk_range[1] + 1):
        correct_k = correct[:, :k].contiguous().view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        if acc.item() > 99.9:
            return k
    return maxk  # Default to the max if the threshold is not exceeded

def nl_em_loss(logits_s, logits_w, k, mask_pred, p_cutoff):
    softmax_pred_s = F.softmax(logits_s, dim=-1)
    softmax_pred_w = F.softmax(logits_w.detach(), dim=-1)
    topk_values, topk_indices = softmax_pred_w.topk(k, dim=-1, sorted=True)
    mask_k = torch.zeros_like(softmax_pred_w).scatter_(1, topk_indices, 1)
    mask_k_npl = torch.where((mask_k == 1) & (softmax_pred_s > p_cutoff**2), torch.zeros_like(mask_k), mask_k)
    loss_npl = (-torch.log(1 - softmax_pred_s + 1e-10) * mask_k_npl).sum(dim=1).mean()

    label = torch.argmax(softmax_pred_w, dim=-1)
    mask_k = mask_k.scatter(1, label.view(-1, 1), 1)

    yg = torch.masked_select(softmax_pred_s, mask_k.bool()).view(logits_w.size(0), -1).sum(dim=-1, keepdim=True)
    soft_ml = ((1 - yg + 1e-7) / (k - 1)).expand_as(logits_s)
    soft_ml = soft_ml.expand_as(logits_s)
    mask = 1 - mask_k
    mask = mask * mask_pred.view(-1, 1)
    mask = torch.where((mask == 1) & (softmax_pred_s > p_cutoff**2), torch.zeros_like(mask), mask)
    loss_em = -(soft_ml * torch.log(softmax_pred_s + 1e-10) + (1 - soft_ml) * torch.log(1 - softmax_pred_s + 1e-10))
    loss_em = (loss_em * mask).sum() / (mask.sum() + 1e-10)
    
    return loss_npl, loss_em

# The rest of the implementation will depend on the particular details of your learning model.
# You will need to add these functions to your model's training loop where appropriate.

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpOnVariablePlateau(_LRScheduler):
    def __init__(self, optimizer, max_epoch, warmup_steps, max_lr=0.1, min_lr=0.001, gamma=None, last_epoch=-1):
        """
        optimizer (Optimizer): Wrapped optimizer.
        max_epoch (int): Total number of epochs for training.
        warmup_steps (int): Warm-up period steps before cosine annealing starts.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        gamma (float): Learning rate decay factor after each cycle. Automatically calculated if None.
        last_epoch (int): The index of last epoch. Default: -1.
        """
        self.max_epoch = max_epoch
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.cycle_length = max(10, max_epoch // 10)  # cycle_length가 최소 10 이상
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.patience_trigger = False  # patience 초과 시 트리거
        self.patience = None  # patience는 로그 적분 방식으로 나중에 설정
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.best_metrics = 0
        self.current_patience = 0
        self.current_cycle = 0
        
        # 자동으로 gamma 설정
        if gamma is None:
            self.gamma = self._calculate_gamma()

        super(CosineAnnealingWarmUpOnVariablePlateau, self).__init__(optimizer, last_epoch)
        self._reset_patience()

    def _reset_patience(self):
        """Recalculate patience based on the current cycle length and log difference."""
        log_diff = math.log(self.max_lr / self.min_lr)
        self.patience = max(0.2 * self.cycle_length, min(log_diff * self.cycle_length * 0.8, 0.8 * self.cycle_length))

    def get_lr(self):
        """Compute the learning rate based on the current cycle"""
        if self.last_epoch < self.warmup_steps:
            # Warm-up 단계: 학습률 선형 증가
            warmup_factor = self.last_epoch / self.warmup_steps
            return [self.min_lr + (self.max_lr - self.min_lr) * warmup_factor for _ in self.base_lrs]
        else:
            # Cosine Annealing 단계
            cycle_progress = (self.last_epoch - self.warmup_steps) % self.cycle_length
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_progress / self.cycle_length))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_factor for _ in self.base_lrs]

    def step(self, metrics=None):
        """Adjust the learning rate and set the gamma trigger if necessary"""
        if metrics is not None:
            # 성능이 개선된 경우
            if metrics > self.best_metrics:
                self.best_metrics = metrics
                self.current_patience = 0
                self.patience_trigger = False  # 성능 개선되었으므로 trigger 비활성화
            else:
                self.current_patience += 1

            # patience 초과 시 트리거 설정
            if self.current_patience >= self.patience:
                self.patience_trigger = True

        # 사이클 종료 시 gamma 적용
        if self.last_epoch % self.cycle_length == 0 and self.last_epoch > 0:
            if self.patience_trigger:  # patience 초과 트리거가 활성화된 경우에만 gamma 적용
                self._reduce_lr_on_plateau()
            self.current_cycle += 1

        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _reduce_lr_on_plateau(self):
        """Reduce the learning rate when a plateau is detected"""
        self.max_lr *= self.gamma
        self.current_patience = 0
        self.patience_trigger = False  # gamma 적용 후 트리거 비활성화
        self._reset_patience()

    def _calculate_gamma(self):
        """Automatically calculate gamma based on max_lr and min_lr"""
        num_cycles = self.max_epoch // self.cycle_length
        return (self.min_lr * 100 / self.max_lr) ** (1 / num_cycles)
