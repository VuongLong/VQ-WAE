import os
import argparse
from configs.defaults import get_cfgs_defaults
import torch

from trainer import VQWAETrainer
from util import set_seeds, get_loader
from networks.classifier import Classifier
from torch import nn


def arg_parse():
    parser = argparse.ArgumentParser(
            description="main.py")
    parser.add_argument(
        "-c", "--config_file", default="", help="config file")
    parser.add_argument(
        "-ts", "--timestamp", default="", help="saved path (random seed + date)")
    parser.add_argument(
        "-rs", "--resume", default="", help="saved path (random seed + date)")
    parser.add_argument(
        "--save", action="store_true", help="save trained model")
    parser.add_argument(
        "--dbg", action="store_true", help="print losses per epoch")
    parser.add_argument(
        "--gpu", default="0", help="index of gpu to be used")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed number for randomness")
    args = parser.parse_args()
    return args


def load_config(args):
    cfgs = get_cfgs_defaults()
    config_path = os.path.join(os.path.dirname(__file__), "configs", args.config_file)
    print(config_path)
    cfgs.merge_from_file(config_path)
    cfgs.train.seed = args.seed
    cfgs.flags.save = args.save
    cfgs.flags.noprint = not args.dbg
    cfgs.path_data = cfgs.path
    cfgs.path = os.path.join(cfgs.path, cfgs.path_specific)

    cfgs.freeze()
    flgs = cfgs.flags
    return cfgs, flgs

def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


if __name__ == "__main__":
    print("main.py")
    
    ## Experimental setup
    args = arg_parse()
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfgs, flgs = load_config(args)
    print("[Checkpoint path] "+cfgs.path)
    print(cfgs)
    
    ## Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)

    ## Data loader
    train_loader, val_loader, test_loader = get_loader(
        cfgs.dataset.name, cfgs.path_dataset, cfgs.train.bs, cfgs.nworker)
    print("Complete dataload")

    ## Trainer
    print("=== {} ===".format(cfgs.model.name.upper()))
    trainer = VQWAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    trainer.load(args.timestamp)

    Classifier = Classifier(4096, 10).cuda()

    optimizer = torch.optim.Adam(Classifier.parameters(), lr=1e-5, amsgrad=False)
    criterion = nn.CrossEntropyLoss()

    epoches = 100
    vals = []
    tests = []
    for epoch in range(epoches):
        print('Epoch: ', epoch)
        for batch_idx, (real_images, labels) in enumerate(train_loader):
            real_images, labels = real_images.cuda(), labels.cuda()
            z_from_encoder = trainer.model.encoder(real_images)
            z_quantized, _, _, _= trainer.model.quantizer._inference(z_from_encoder, trainer.model.codebook)
            logit = Classifier(z_quantized.reshape(real_images.shape[0], -1))
            loss = criterion(logit, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 300 == 0:
                print('Loss: ', loss)

        correct = 0
        for batch_idx, (real_images, labels) in enumerate(val_loader):
            real_images, labels = real_images.cuda(), labels.cuda()
            z_from_encoder = trainer.model.encoder(real_images)
            z_quantized, _, _, _= trainer.model.quantizer._inference(z_from_encoder, trainer.model.codebook)
            logit = Classifier(z_quantized.reshape(real_images.shape[0], -1))
            correct += calculate_correct(logit, labels)
            vals.append(correct/len(val_loader.dataset))

        print('Val: ', vals[-1])

        correct = 0
        for batch_idx, (real_images, labels) in enumerate(test_loader):
            real_images, labels = real_images.cuda(), labels.cuda()
            z_from_encoder = trainer.model.encoder(real_images)
            z_quantized, _, _, _= trainer.model.quantizer._inference(z_from_encoder, trainer.model.codebook)
            logit = Classifier(z_quantized.reshape(real_images.shape[0], -1))
            correct += calculate_correct(logit, labels)

        tests.append(correct/len(test_loader.dataset))
        print('Test: ', tests[-1])
        print('Best val:, ', vals)
        print('Best test:, ', tests)


