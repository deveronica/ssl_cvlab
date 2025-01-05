import random

import torch
import torch.nn.functional as F
import torch.optim as optim
# from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm

import numpy as np

import mlflow
import mlflow.pytorch

from data.loader import get_dataloader

from .train_utils import CosineAnnealingWarmUpOnVariablePlateau
from models.comannet import ComanNet
from models.resnet18 import ResNet18


def get_model(cfg):
    if cfg.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=28,
                                        widen_factor=10,
                                        dropout=0,
                                        num_classes=cfg.num_classes)
    elif cfg.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=8,
                                    depth=29,
                                    width=64,
                                    num_classes=cfg.num_classes)
    elif cfg.arch == "comannet":
        model = ComanNet(para=((128 - 4) // 2) ** 2 * 32, cls=cfg.num_classes)
    elif cfg.arch == 'resnet18':
        model = ResNet18(cfg)
    return model


class Trainer():
    """Trainer for SSL like FixMatch, FullMatch"""
    def __init__(self, cfg):
        self.cfg = cfg
        mlflow.set_tracking_uri(uri=cfg.uri)
        self.train(self.cfg)

    def set_seed(self, cfg):
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        if cfg.n_gpu > 0:
            torch.cuda.manual_seed_all(cfg.SEED)

    def open(self, cfg):
        try:
            mlflow.set_experiment(cfg.experiment)
        except:
            mlflow.create_experiment(cfg.experiment)
            mlflow.set_experiment(cfg.experiment)
        mlflow.start_run()
        mlflow.log_params(vars(cfg))

    def close(self):
        mlflow.end_run()
        
    def train(self, cfg):
        self.open(cfg)

        global best_loss
        con = 0
        
        self.set_seed(cfg)  # 시드 고정

        model = get_model(cfg).to(f'cuda:{cfg.device}')

        # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        # scheduler = CosineAnnealingWarmUpOnVariablePlateau(
        #     optimizer,
        #     max_epoch=cfg.epochs,
        #     warmup_steps=30,
        #     max_lr=0.01,
        #     min_lr=1e-6,
        #     )  # 스케줄러 설정
        
        
        if cfg.optimizer == 'sgd':
            # optimizer SGD로 변경
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=5e-4
            )   

        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        labeled_trainloader, unlabeled_trainloader, testloader =  get_dataloader(cfg)

        # early stopping (loss 변경)
        best_loss = float('inf') #best_acc = 0
        early_stopping_counter = 0

        for epoch in tqdm(range(cfg.epochs), desc="Training"):
            model.train()
            total_loss = 0
            last_loss = 10e9
            test_acc = 0

            for idx, (labeled, unlabeled) in enumerate(zip(labeled_trainloader, unlabeled_trainloader)):
                optimizer.zero_grad()
                inputs_x, targets_x = labeled
                (inputs_u_w, inputs_u_s1), _ = unlabeled

                inputs_x, targets_x = inputs_x.to(f'cuda:{cfg.device}'), targets_x.to(f'cuda:{cfg.device}')
                inputs_u_w = inputs_u_w.to(f'cuda:{cfg.device}')
                inputs_u_s1 = inputs_u_s1.to(f'cuda:{cfg.device}')
                # inputs_u_s2 = inputs_u_s2.to(f'cuda:{cfg.device}')

                logits_x = model(inputs_x)
                logits_u_w = model(inputs_u_w)
                logits_u_s1 = model(inputs_u_s1)
                # logits_u_s2 = model(inputs_u_s2)

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(cfg.threshold).float()

                Lu = (F.cross_entropy(logits_u_s1, targets_u, reduction='none') * mask).mean()
                # Lu2 = (F.cross_entropy(logits_u_s2, targets_u, reduction='none') * mask).mean()

                # # AugMix
                # p_u_w = torch.clamp(F.softmax(logits_u_w, dim=1), 1e-7, 1)
                # p_u_s1 = torch.clamp(F.softmax(logits_u_s1, dim=1), 1e-7, 1)
                # p_u_s2 = torch.clamp(F.softmax(logits_u_s2, dim=1), 1e-7, 1)

                # # Clamp mixture distribution to avoid exploding KL divergence
                # p_mix = torch.clamp((p_u_w + p_u_s1 + p_u_s2) / 3., 1e-7, 1).log()
                # Ljsd = cfg.jsd * (F.kl_div(p_mix, p_u_w, reduction='batchmean')
                #              + F.kl_div(p_mix, p_u_s1, reduction='batchmean') 
                #              + F.kl_div(p_mix, p_u_s2, reduction='batchmean')) / 3
                # # -AugMix

                loss = Lx + Lu # + Ljsd

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # mlflow.log_metric("lr", scheduler.get_lr()[0], step=epoch)
            
            avg_loss = total_loss / len(labeled_trainloader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            test_acc = self.test(cfg, model, testloader)

            # is_best = test_acc > best_acc
            # best_acc = max(test_acc, best_acc)

            # scheduler.step(test_acc)  # 스케줄러 업데이트

            mlflow.log_metric("accuracy", test_acc, step=epoch)

            # 조기 종료 조건 추가
            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter > 120:
                    print("Early stopping(loss)...")
                    break
                
        mlflow.pytorch.log_model(model, "model")
        self.close()

    def test(self, cfg, model, testloader):
        model.eval()
        correct = 0
        total = 0
        loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(f'cuda:{cfg.device}'), targets.to(f'cuda:{cfg.device}')
                outputs = model(inputs)
                predicted = torch.max(outputs, 1)[1]
                correct += (predicted == targets).sum()
                loss += criterion(outputs, targets)
                total += inputs.size(0)

        accuracy = correct / total
        return accuracy