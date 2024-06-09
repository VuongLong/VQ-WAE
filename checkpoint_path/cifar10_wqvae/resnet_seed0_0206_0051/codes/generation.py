import os
import argparse
from configs.defaults import get_cfgs_defaults
import torch
import numpy as np
from torch import nn
from pixelcnn.models import GatedPixelCNN
from model import GaussianSQVAE, VmfSQVAE, WQVAE, VQVAE
from util import *

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
    if cfgs.model.name.lower() == "vmfsqvae":
        cfgs.quantization.dim_dict += 1
    cfgs.flags.var_q = not(cfgs.model.param_var_q=="gaussian_1" or
                                        cfgs.model.param_var_q=="vmf")
    cfgs.freeze()
    flgs = cfgs.flags
    return cfgs, flgs


def generate_samples(vq_model, model, size_dict, img_dim, bs, labels, latents):
    
    min_encoding_indices = model.generate(labels, shape=(img_dim, img_dim), batch_size=bs)
    #min_encoding_indices = latents
    min_encoding_indices =min_encoding_indices.view(-1).unsqueeze(1)
    min_encodings = torch.zeros(min_encoding_indices.shape[0], size_dict).cuda()
    min_encodings.scatter_(1, min_encoding_indices, 1)
    # get quantized latent vectors
    z_q = torch.matmul(min_encodings, vq_model.module.codebook)

    z_q = z_q.view(-1,img_dim, img_dim, 64)
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    x_hat = vq_model.module.decoder(z_q)
    return x_hat

def sample(vq_model, gen_model, save_path, labels, size_dict=512, bs=32, img_dim=8, is_label=0, latents=None):
    
    labels = labels.view(-1,bs)

    if is_label == 0:
        labels = torch.zeros(labels.shape).long().cuda()
    if is_label:
        pp = '/gen_label/'
    else:
        pp = '/gen/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + pp, exist_ok=True)

    print(labels.shape)
    
    for i in range(labels.shape[0]):
        print(i*50, labels.shape[1])
        x = generate_samples(vq_model, gen_model, size_dict, img_dim, labels.shape[1], labels[i], latents)
        for idx in range(x.shape[0]):
            save_image(tensor2im(x[idx]), save_path + pp + str(i*labels.shape[1]+idx)+'.png')



def arg_parse():
    parser = argparse.ArgumentParser(
            description="main.py")
    parser.add_argument(
        "-c", "--config_file", default="", help="config file")
    parser.add_argument(
        "-ts", "--timestamp", default="", help="saved path (random seed + date)")
    parser.add_argument(
        "--save", action="store_true", help="save trained model")
    parser.add_argument(
        "--dbg", action="store_true", help="print losses per epoch")
    parser.add_argument(
        "--gpu", default="0", help="index of gpu to be used")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed number for randomness")

    parser.add_argument("--is_label", type=int, default=0)

    args = parser.parse_args()
    return args

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

    
    # load vq_model
    vq_model = eval("nn.DataParallel({}(cfgs, flgs).cuda())".format(cfgs.model.name))
    vq_model_path = os.path.join(cfgs.path, args.timestamp)
    vq_model.load_state_dict(torch.load(os.path.join(vq_model_path, "best.pt")))

    # load latent data
    load_data = np.load(cfgs.generation.base_path+cfgs.generation.data, allow_pickle=True)
    data = torch.from_numpy(load_data['data']).cuda()
    labels = torch.from_numpy(load_data['label']).cuda()
    labels = labels[:labels.shape[0]-labels.shape[0] % cfgs.test.bs]

    # load generation_model
    gen_model = GatedPixelCNN(cfgs.quantization.size_dict, cfgs.generation.gen_model.img_dim**2, cfgs.generation.gen_model.n_layers).cuda()
    gen_model.load_state_dict(torch.load(cfgs.generation.base_path+cfgs.generation.gen_model.name.format(bool(args.is_label)),map_location='cuda:0'))
    gen_model.eval()
    save_path = cfgs.generation.base_path+cfgs.generation.gen_model.name[:-3].format(bool(args.is_label))

    print(cfgs.generation.base_path+cfgs.generation.gen_model.name.format(bool(args.is_label)))
    print(cfgs.generation.base_path+cfgs.generation.gen_model.name[:-3].format(bool(args.is_label)))
    print("Best models were loaded!!")

    sample(vq_model, gen_model, save_path, labels, 
        size_dict=cfgs.quantization.size_dict, 
        bs=cfgs.test.bs, 
        img_dim=cfgs.generation.gen_model.img_dim, 
        is_label=args.is_label,
        latents=data[:50,:,:,0])


