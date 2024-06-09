import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quantizer import WSVectorQuantizer, DuelForm_WSVectorQuantizer, Global_DuelForm_WSVectorQuantizer, VectorQuantizer, GaussianVectorQuantizer
import networks.mnist as net_mnist
import networks.cifar10 as net_cifar10
import networks.svhn as net_svhn
import networks.celeba as net_celeba
#from third_party.ive import ive
from torch.distributions.normal import Normal


def init_gaussian_array(std, size_dict):
	center = size_dict // 2
	normal = Normal(center, std)
	weights = torch.ones(size_dict)
	for i in range(0,size_dict):
		weights[i] = normal.cdf(torch.tensor(i))-normal.cdf(torch.tensor(i-1))
	return weights

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm") != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


class EnsembleLinear(nn.Linear):
	def __init__(self, ensemble_size, in_features, out_features):
		nn.Module.__init__(self)
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
		self.bias = nn.Parameter(torch.Tensor(ensemble_size, 1, out_features))
		self.reset_parameters()

	def forward(self, x):
		x = torch.baddbmm(self.bias, x, self.weight)
		return x


class KantorovichNetwork(nn.Module):
	def __init__(self, ensemble, embeddings_size=64, output_size=1):
		super(KantorovichNetwork, self).__init__()
		if ensemble > 1:
			self.fc1 = EnsembleLinear(ensemble, embeddings_size, embeddings_size)
			self.fc2 = EnsembleLinear(ensemble, embeddings_size, output_size)
		else:
			self.fc1 = nn.Linear(embeddings_size, embeddings_size)
			self.fc2 = nn.Linear(embeddings_size, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x


class VQWAE(nn.Module):
	def __init__(self, cfgs, flgs):
		super(VQWAE, self).__init__()
		# Data space
		dataset = cfgs.dataset.name
		self.dim_x = cfgs.dataset.dim_x
		self.dataset = cfgs.dataset.name

		##############################################
		# Encoder/decoder
		##############################################
		self.encoder = eval("net_{}.EncoderVq_{}".format(dataset.lower(), cfgs.network.name))(
			cfgs.quantization.dim_dict, cfgs.network, flgs.bn)
		self.decoder = eval("net_{}.DecoderVq_{}".format(dataset.lower(), cfgs.network.name))(
			cfgs.quantization.dim_dict, cfgs.network, flgs.bn)

		#self.pre_quantization_conv_m = nn.Conv2d(128, 64, kernel_size=1, stride=1)
		self.apply(weights_init)

		##############################################
		# Codebook
		##############################################
		self.size_dict = cfgs.quantization.size_dict
		self.dim_dict = cfgs.quantization.dim_dict
		self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
		
		init_weights = None
		if cfgs.quantization.global_optimization:
			if cfgs.quantization.init_weight == 'gaussian':
				print("INIT GAUSSIAN")
				init_weights = torch.log(init_gaussian_array(100, self.size_dict))
				init_weights = init_weights.cpu().numpy() 
				self.codebook_weight = nn.Parameter(torch.tensor(init_weights))
			elif cfgs.quantization.init_weight == 'gaussian200':
				print("INIT GAUSSIAN-200")
				init_weights = torch.log(init_gaussian_array(200, self.size_dict))
				init_weights = init_weights.cpu().numpy() 
				self.codebook_weight = nn.Parameter(torch.tensor(init_weights))
			elif cfgs.quantization.init_weight == 'gaussian150':
				print("INIT GAUSSIAN-150")
				init_weights = torch.log(init_gaussian_array(150, self.size_dict))
				init_weights = init_weights.cpu().numpy() 
				self.codebook_weight = nn.Parameter(torch.tensor(init_weights))
			elif cfgs.quantization.init_weight == 'peaked':
				print("INIT PEAKED")
				init_weights = torch.ones(self.size_dict)/self.size_dict/2
				init_weights[206:306]+=0.005
				init_weights = torch.log(init_weights).cpu().numpy() 
				self.codebook_weight = nn.Parameter(torch.tensor(init_weights))
			else:
				print("INIT UNIFORM")
				self.codebook_weight = nn.Parameter(torch.ones(self.size_dict)/self.size_dict)
		else:
			print("INIT UNIFORM")
			self.codebook_weight = nn.Parameter(torch.zeros(64, self.size_dict)/self.size_dict)
			for i in range(64):
				self.codebook_weight.data[i, 8*i:8*(i+1)]=1.0



		##############################################
		# Quantizer
		##############################################
		if cfgs.quantization.name == "VQWAE":
			self.quantizer = WSVectorQuantizer(self.size_dict, self.dim_dict, cfgs, init_weights)
		elif cfgs.quantization.name == "FVQWAE":
			if cfgs.quantization.global_optimization:
				self.kan_net1 = KantorovichNetwork(1, self.dim_dict, 1)
				self.kan_net2 = KantorovichNetwork(1, self.dim_dict, 1)
				self.quantizer = Global_DuelForm_WSVectorQuantizer(self.size_dict, self.dim_dict, self.kan_net1, self.kan_net2, cfgs, init_weights)
			else:
				self.kan_net1 = KantorovichNetwork(64, self.dim_dict, 1)
				self.kan_net2 = KantorovichNetwork(64, self.dim_dict, 1)

				self.quantizer = DuelForm_WSVectorQuantizer(self.size_dict, self.dim_dict, self.kan_net1, self.kan_net2, cfgs, init_weights)
		else:
			self.quantizer = VectorQuantizer(self.size_dict, self.dim_dict, cfgs)
		

	def forward(self, real_images, flg_train=False, flg_quant_det=True):
		# Encoding
		if flg_train:
			##############################################
			# VQ-model
			##############################################
			z_from_encoder = self.encoder(real_images)

			z_quantized, loss_latent, perplexity = self.quantizer(z_from_encoder, self.codebook, self.codebook_weight, flg_train)
			latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)
			# Decoding
			x_reconst = self.decoder(z_quantized)

			# Loss
			loss = self._calc_loss(x_reconst, real_images, loss_latent, flg_train=flg_train)
			loss["perplexity"] = perplexity

			return x_reconst, latents, loss
		else:
			z_from_encoder = self.encoder(real_images)

			z_quantized, min_encodings, e_indices, perplexity = self.quantizer._inference(z_from_encoder, self.codebook)
			# Decoding
			x_reconst = self.decoder(z_quantized)
			# Loss
			loss = self._calc_loss(x_reconst, real_images, 0, flg_train=flg_train)
			loss["perplexity"] = perplexity
			return x_reconst, min_encodings, e_indices, loss
		return 0, 0, 0
		

	def _calc_loss(self, x_reconst, x, loss_latent, flg_train=False):  
		bs = x.shape[0]

		if flg_train: 
			mse = F.mse_loss(x_reconst, x)
			loss_all = mse + loss_latent 

		else:
			mse = torch.mean((x_reconst - x)**2)
			loss_all = mse

		loss = dict(all=loss_all, mse=mse)

		return loss



class SQVAE(nn.Module):
	def __init__(self, cfgs, flgs):
		super(SQVAE, self).__init__()
		# Data space
		dataset = cfgs.dataset.name
		self.dim_x = cfgs.dataset.dim_x
		self.dataset = cfgs.dataset.name

		# Encoder/decoder
		self.param_var_q = cfgs.model.param_var_q
		self.encoder = eval("net_{}.EncoderVq_{}".format(dataset.lower(), cfgs.network.name))(
			cfgs.quantization.dim_dict, cfgs.network, flgs.bn)
		self.decoder = eval("net_{}.DecoderVq_{}".format(dataset.lower(), cfgs.network.name))(
			cfgs.quantization.dim_dict, cfgs.network, flgs.bn)
		self.pre_quantization_conv_m = nn.Conv2d(128, 64, kernel_size=1, stride=1)

		self.apply(weights_init)

		# Codebook
		self.size_dict = cfgs.quantization.size_dict
		self.dim_dict = cfgs.quantization.dim_dict
		self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
		self.log_param_q_scalar = nn.Parameter(torch.tensor(cfgs.model.log_param_q_init))
		self.quantizer = GaussianVectorQuantizer(
			self.size_dict, self.dim_dict, cfgs.quantization.temperature.init, self.param_var_q)
		
	
	def forward(self, x, flg_train=False, flg_quant_det=True):
		z_from_encoder = self.encoder(x)
		z_from_encoder = self.pre_quantization_conv_m(z_from_encoder)
		log_var_q = torch.tensor([0.0], device="cuda")
		self.param_q = (log_var_q.exp() + self.log_param_q_scalar.exp())
		if flg_train:
			# Quantization
			z_quantized, loss_latent, perplexity = self.quantizer(
				z_from_encoder, self.param_q, self.codebook, flg_train, flg_quant_det)
			latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)
			# Decoding
			x_reconst = self.decoder(z_quantized)
			# Loss
			loss = self._calc_loss(x_reconst, x, loss_latent)
			loss["perplexity"] = perplexity
		else:
			z_quantized, min_encodings, e_indices, loss_latent = self.quantizer._inference(z_from_encoder, self.param_q, self.codebook)
			x_reconst = self.decoder(z_quantized)
			loss = self._calc_loss(x_reconst, x, loss_latent)
			return x_reconst, min_encodings, e_indices, loss
		
		return x_reconst, latents, loss
	
	def _calc_loss(self):
		raise NotImplementedError()
	

class GaussianSQVAE(SQVAE):
	def __init__(self, cfgs, flgs):
		super(GaussianSQVAE, self).__init__(cfgs, flgs)
		self.flg_arelbo = flgs.arelbo # Use MLE for optimization of decoder variance
		if not self.flg_arelbo:
			self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))
	
	def _calc_loss(self, x_reconst, x, loss_latent):
		bs = x.shape[0]
		# Reconstruction loss
		mse = F.mse_loss(x_reconst, x, reduction="sum") / bs
		if self.flg_arelbo:
			# "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
			# https://arxiv.org/abs/2102.08663
			loss_reconst = self.dim_x * torch.log(mse) / 2
		else:
			loss_reconst = mse / (2*self.logvar_x.exp()) + self.dim_x * self.logvar_x / 2
		# Entire loss

		loss_all = loss_reconst + loss_latent
		mse = torch.mean((x_reconst - x)**2)
		loss = dict(all=loss_all, mse=mse)

		return loss 

