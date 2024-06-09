import time

from trainer_base import TrainerBase
from util import *
from torch import nn


class VQWAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(VQWAETrainer, self).__init__(
            cfgs, flgs, train_loader, val_loader, test_loader)
        self.plots = {
            "loss_train": [], "mse_train": [], "perplexity_train": [],
            "loss_val": [], "mse_val": [], "perplexity_val": [],
            "loss_test": [], "mse_test": [], "perplexity_test": []
        }
        
    def _train(self, epoch):
        train_loss = []
        ms_error = []
        perplexity = []
        self.model.train()
        start_time = time.time()

        #import pdb; pdb.set_trace()

        if self.cfgs.quantization.global_optimization:
            cb_distribution = nn.Softmax()(self.model.codebook_weight)
        else:
            cb_distribution = nn.Softmax(dim=1)(self.model.codebook_weight).mean(0)

            # weight = nn.Softmax(dim=1)(self.model.codebook_weight)
            # torch.topk(weight[0],3)[0].sum()
            # torch.topk(weight[0],3)[1]
        entropy = torch.exp(-torch.sum(cb_distribution * torch.log(cb_distribution + 1e-10)))
        
        print('CB_distribution: ', entropy)
        #if entropy < 10:
        #    import pdb; pdb.set_trace()

        code_matrix = torch.sum(self.model.codebook ** 2, dim=1, keepdim=True) + torch.sum(self.model.codebook**2, dim=1) - 2 * torch.matmul(self.model.codebook, self.model.codebook.t())
        print(code_matrix.min())

        for batch_idx, (real_images, _) in enumerate(self.train_loader):
            if self.flgs.decay:
                if batch_idx == 0:
                    step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                    temperature_current = self._set_temperature(
                        step, self.cfgs.quantization.temperature)
                    self.model.quantizer.set_temperature(temperature_current)
            
            real_images = real_images.cuda()

            _, _, loss = self.model(real_images, flg_train=True, flg_quant_det=False)


            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].detach().cpu().item())
            ms_error.append(loss["mse"].detach().cpu().item())
            perplexity.append(loss["perplexity"].detach().cpu().item())
            #break
        
        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
                
        return result    
    
    def _test(self, mode="validation"):

        self.model.eval()
        #_ = self._test_sub(False, mode)
        result = self._test_sub(True, mode)
        self.scheduler.step(result["loss"])
        return result

    def _test_sub(self, flg_quant_det, mode="validation"):
        
        os.makedirs('image_results', exist_ok=True)
        os.makedirs('latent_results', exist_ok=True)
        if mode == "validation":
            data_loader = self.val_loader

        elif mode == "train":
            data_loader = self.train_loader
            
            save_path = 'image_results/train_{}_{}_Fixed_{}_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.fixed_weight, self.cfgs.quantization.global_optimization,
                self.cfgs.quantization.init_weight, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            
            save_latent_path = 'latent_results/train_{}_{}_Fixed_{}_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.fixed_weight, self.cfgs.quantization.global_optimization,
                self.cfgs.quantization.init_weight, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path + '/train/', exist_ok=True)
            os.makedirs(save_path + '/rec/', exist_ok=True)
        
        elif mode == "test":
            data_loader = self.test_loader
            save_path = 'image_results/seed_100_test_{}_{}_Fixed_{}_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.fixed_weight, self.cfgs.quantization.global_optimization,
                self.cfgs.quantization.init_weight, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            save_latent_path = 'latent_results/seed_100_test_{}_{}_Fixed_{}_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.fixed_weight, self.cfgs.quantization.global_optimization,
                self.cfgs.quantization.init_weight, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path + '/train/', exist_ok=True)
            os.makedirs(save_path + '/rec/', exist_ok=True)
        

        start_time = time.time()
        
        test_loss = []
        recon_loss = 0.0
        histogram = torch.zeros(64, self.cfgs.quantization.size_dict).cuda()
        len_data  = len(data_loader.dataset)
        save_data = None
        save_label = None
        with torch.no_grad():
            i = 0
            for x, y in data_loader:
                x = x.cuda()
                if len(y.shape) > 1:
                    y = y.sum(1)
                x_reconst, min_encodings, e_indices, loss = self.model(x)
                #import pdb; pdb.set_trace()
                
                histogram += min_encodings.reshape(x_reconst.shape[0], 64, self.cfgs.quantization.size_dict).sum(0)
                recon_loss += ((x_reconst - x)**2).mean(3).mean(2).mean(1).sum()

                test_loss.append(loss["all"].item())

                if mode == "test" or mode == "train":
                    for idx in range(x.shape[0]):
                        save_image(tensor2im(x[idx]), save_path + '/train/' + str(i*self.cfgs.test.bs+idx)+'.png')
                        save_image(tensor2im(x_reconst[idx]), save_path + '/rec/' + str(i*self.cfgs.test.bs+idx)+'.png')
                
                    latent_size = int(x_reconst.shape[-1] / 4)
                    indices_numpy = e_indices.view(x.shape[0],latent_size,latent_size, 1).cpu().numpy()
                    if save_data is None:
                        save_data = indices_numpy
                        save_label = y.numpy()
                    else:
                        save_data = np.concatenate([save_data, indices_numpy],0)
                        save_label = np.concatenate([save_label, y.numpy()],0)
                i+=1
            recon_loss /= len_data  
            e_mean = histogram.sum(0)/(len_data*x_reconst.shape[-1]*x_reconst.shape[-1]/16)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
            if mode == 'test':
                np.savez(save_latent_path, data=save_data, label=save_label, hist=histogram.cpu().numpy()) 
                np.savez(save_latent_path+'codebook_weight', weight=self.model.codebook_weight.cpu().numpy()) 
        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["mse"] = recon_loss
        
        result["perplexity"] = perplexity
        self.print_loss(result, mode, time.time()-start_time)
        
        return result
    

    def generate_reconstructions(self, filename, nrows=4, ncols=8):
        self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
    

    def print_loss(self, result, mode, time_interval):
        #import pdb; pdb.set_trace()
        myprint(mode.capitalize().ljust(16) +
            "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec"
            .format(
                result["loss"], result["mse"], result["perplexity"], time_interval
            ), self.flgs.noprint)


class GaussianSQVAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(GaussianSQVAETrainer, self).__init__(
            cfgs, flgs, train_loader, val_loader, test_loader)
        self.plots = {
            "loss_train": [], "mse_train": [], "perplexity_train": [],
            "loss_val": [], "mse_val": [], "perplexity_val": [],
            "loss_test": [], "mse_test": [], "perplexity_test": []
        }
        
    def _train(self, epoch):
        train_loss = []
        ms_error = []
        perplexity = []
        self.model.train()
        start_time = time.time()
        for batch_idx, (x, _) in enumerate(self.train_loader):
            if self.flgs.decay:
                step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                temperature_current = self._set_temperature(
                    step, self.cfgs.quantization.temperature)
                self.model.quantizer.set_temperature(temperature_current)
            x = x.cuda()
            _, _, loss = self.model(x, flg_train=True, flg_quant_det=False)
            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].detach().cpu().item())
            ms_error.append(loss["mse"].detach().cpu().item())
            perplexity.append(loss["perplexity"].detach().cpu().item())

        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
                
        return result    
    
    def _test(self, mode="validation"):
        self.model.eval()
        #_ = self._test_sub(False, mode)
        result = self._test_sub(True, mode)
        self.scheduler.step(result["loss"])
        return result

    def _test_sub(self, flg_quant_det, mode="validation"):
        os.makedirs('image_results', exist_ok=True)
        os.makedirs('latent_results', exist_ok=True)
        if mode == "validation":
            data_loader = self.val_loader
        elif mode == "train":
            data_loader = self.train_loader
            save_path = 'image_results/train_{}_{}_{}'.format(self.cfgs.dataset.name, 
                'sqvae', self.cfgs.quantization.size_dict)
            save_latent_path = 'latent_results/train_{}_{}_{}'.format(self.cfgs.dataset.name, 
                'sqvae', self.cfgs.quantization.size_dict)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path + '/train/', exist_ok=True)
            os.makedirs(save_path + '/rec/', exist_ok=True)
        

        elif mode == "test":
            data_loader = self.test_loader
            save_path = 'image_results/test_{}_{}_{}'.format(self.cfgs.dataset.name, 
                'sqvae', self.cfgs.quantization.size_dict)
            save_latent_path = 'latent_results/test_{}_{}_{}'.format(self.cfgs.dataset.name, 
                'sqvae', self.cfgs.quantization.size_dict)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path + '/train/', exist_ok=True)
            os.makedirs(save_path + '/rec/', exist_ok=True)

        test_loss = []
        ms_error = []
        perplexity = []
        histogram = torch.zeros(self.cfgs.quantization.size_dict).cuda()
        recon_loss = 0.0
        len_data  = len(data_loader.dataset)
        save_data = None
        save_label = None

        start_time = time.time()
        with torch.no_grad():
            i = 0
            for x, y in data_loader:
                x = x.cuda()
                if len(y.shape) > 1:
                    y = y.sum(1)
                x_reconst, min_encodings, e_indices, loss = self.model(x)
                
                histogram += min_encodings.sum(0)
                recon_loss += ((x_reconst - x)**2).mean(3).mean(2).mean(1).sum()
                
                test_loss.append(loss["all"].item())

                if mode == "test" or mode == "train":
                    #import pdb; pdb.set_trace()
                    for idx in range(x.shape[0]):
                        save_image(tensor2im(x[idx]), save_path + '/train/' + str(i*self.cfgs.test.bs+idx)+'.png')
                        save_image(tensor2im(x_reconst[idx]), save_path + '/rec/' + str(i*self.cfgs.test.bs+idx)+'.png')
                
                    latent_size = int(x_reconst.shape[-1] / 4)
                    indices_numpy = e_indices.view(x.shape[0],latent_size,latent_size, 1).cpu().numpy()
                    if save_data is None:
                        save_data = indices_numpy
                        save_label = y.numpy()
                    else:
                        save_data = np.concatenate([save_data, indices_numpy],0)
                        save_label = np.concatenate([save_label, y.numpy()],0)
                i+=1
            recon_loss /= len_data  
            e_mean = histogram/(len_data*x_reconst.shape[-1]*x_reconst.shape[-1]/16)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
            if mode == 'test':
                np.savez(save_latent_path, data=save_data, label=save_label, hist=histogram.cpu().numpy()) 

        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["mse"] = recon_loss
        
        result["perplexity"] = perplexity
        self.print_loss(result, mode, time.time()-start_time)
        
        return result
    
    def generate_reconstructions(self, filename, nrows=4, ncols=8):
        self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
    
    def print_loss(self, result, mode, time_interval):
        myprint(mode.capitalize().ljust(16) +
            "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec"
            .format(
                result["loss"], result["mse"], result["perplexity"], time_interval
            ), self.flgs.noprint)
