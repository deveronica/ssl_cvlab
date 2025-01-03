import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from torchvision import datasets, transforms

from tqdm import tqdm

import numpy as np

import mlflow
import mlflow.pytorch

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
        # mlflow.set_experiment(cfg.experiment)
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
            try:
                mlflow.create_experiment(cfg.experiment)
            except:
                from mlflow.tracking.client import MlflowClient
                client = MlflowClient(tracking_uri=cfg.uri)
                key = client.get_experiment_by_name(cfg.experiment).experiment_id
                client.restore_experiment(key)
            mlflow.set_experiment(cfg.experiment)
        mlflow.start_run()
        mlflow.log_params(vars(cfg))

    def close(self):
        mlflow.end_run()
        
    def train(self, cfg):
        self.open(cfg)

        global best_loss
        
        self.set_seed(cfg)  # 시드 고정

        model = get_model(cfg).to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 스케줄러 설정

        # cifar10_mean = (0.4914, 0.4822, 0.4465)
        # cifar10_std = (0.2471, 0.2435, 0.2616)

        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        # ])
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
        test_transform = preprocess

        trainset = datasets.CIFAR10(cfg.root, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(cfg.root, train=False, download=True, transform=test_transform)

        train_sampler = RandomSampler
        test_sampler = SequentialSampler
        
        labeled_trainloader = DataLoader(
            trainset,
            sampler=train_sampler(trainset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        testloader = DataLoader(
            testset,
            sampler=test_sampler(testset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        best_acc = 0
        early_stopping_counter = 0

        for epoch in tqdm(range(cfg.epochs), desc="Training"):
            model.train()
            total_loss = 0

            for idx, labeled in enumerate(labeled_trainloader):
                inputs_x, targets_x = labeled
                # (inputs_u_w, inputs_u_s), _ = unlabeled

                inputs_x, targets_x = inputs_x.to('cuda'), targets_x.to('cuda')
                optimizer.zero_grad()

                # inputs_u_s = torch.Tensor(inputs_u_s, requires_grad=True)
                # inputs_u_s.requires_grad_()

                logits_x = model(inputs_x)
                # inputs_u_w = inputs_u_w.to('cuda')
                # logits_u_w = model(inputs_u_w)
                # inputs_u_s = inputs_u_s.to('cuda')
                # logits_u_s = model(inputs_u_s)

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                # pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
                # max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                # For FGSM
                # cost = -torch.nn.CrossEntropyLoss()(logits_u_s, pseudo_label)
                # if inputs_u_s.grad is not None:
                    # inputs_u_s.grad.data.fill_(0)
                # cost.backward()
                # print(inputs_u_s.grad)
                # inputs_u_s = inputs_u_s - 0.03*torch.Tensor.sign_(inputs_u_s.grad)
                # model.zero_grad()
                # logits_u_s = model(inputs_u_s)

                # mask = max_probs.ge(cfg.threshold).float()

                # Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                loss = Lx #+ Lu
                loss.backward()
                optimizer.step()
                # scheduler.step()  # 스케줄러 업데이트

                total_loss += loss.item()

            mlflow.log_metric("loss", total_loss / len(labeled_trainloader), step=epoch)

            test_acc = self.test(model, testloader)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            mlflow.log_metric("accuracy", test_acc, step=epoch)

            # 조기 종료 조건 추가
            if is_best:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter > 100:
                    print("Early stopping...")
                    break
                
        mlflow.pytorch.log_model(model, "model")
        self.close()

    def test(self, model, testloader):
        model.eval()
        correct = 0
        total = 0
        loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                predicted = torch.max(outputs, 1)[1]
                correct += (predicted == targets).sum()
                loss += criterion(outputs, targets)
                total += inputs.size(0)

        accuracy = correct / total
        return accuracy