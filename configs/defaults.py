from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
_C.path = "/home/bta/Long/ICML_VQWAE/checkpoint_path" # To be set in advance
_C.path_dataset = "/home/bta/Long/ICML_VQWAE/data" # To be set in advance
_C.nworker = 2
_C.list_dir_for_copy = ['', 'networks/'] # []


_C.dataset = CN(new_allowed=True)

_C.model = CN(new_allowed=True)

_C.network = CN(new_allowed=True)

_C.train = CN(new_allowed=True)
_C.train.bs = 32
_C.train.lr = 0.001
_C.train.epoch_max = 100

_C.quantization = CN(new_allowed=True)
_C.quantization.temperature = CN(new_allowed=True)
_C.quantization.temperature.init = 0.5
_C.quantization.temperature.decay = 0.00001
_C.quantization.temperature.min = 0.0
_C.quantization.beta = 0.25

_C.generation = CN(new_allowed=True)
_C.generation.base_path = "/home/bta/Long/ICML_VQWAE/latent_results/"
_C.generation.data = "latent_block.npz"
_C.generation.save_path = "pixelcnn"

_C.generation.gen_model = CN(new_allowed=True)
_C.generation.gen_model.name = "pixelcnn.pt"
_C.generation.gen_model.img_dim = 8
_C.generation.gen_model.n_layers = 15


_C.test = CN(new_allowed=True)
_C.test.bs = 50

_C.flags = CN(new_allowed=True)
_C.flags.arelbo = True
_C.flags.decay = True
_C.flags.bn = True


def get_cfgs_defaults():
  return _C.clone()