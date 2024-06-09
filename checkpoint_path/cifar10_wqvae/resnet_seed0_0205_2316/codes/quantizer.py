import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
import ot
from torch import nn

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device="cuda")
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size())
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (torch.sum(z_continuous_flat**2, dim=1, keepdim=True) 
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(z_continuous_flat, codebook.t()))

    return distances


class VectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, cfgs):
        super(VectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = cfgs.quantization.temperature.init
        self.beta = cfgs.quantization.beta
        print('Using VectorQuantizer')
    
    def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
        return self._quantize(z_from_encoder, codebook, flg_train)
    
    def _quantize(self, z, codebook, flg_train):
        z = z.permute(0, 2, 3, 1).contiguous()
        
        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_flattened, codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.size_dict).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)
        if flg_train:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
            #loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = 0.0
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, perplexity
    

    def _inference(self, z, codebook):
        
        z = z.permute(0, 2, 3, 1).contiguous()
        # import pdb; pdb.set_trace()

        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1) - 2 * \
            torch.matmul(z_flattened, codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.size_dict).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.view(z.shape)
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, min_encodings, min_encoding_indices, perplexity

    def set_temperature(self, value):
        self.temperature = value



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("EnsembleLinear") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DuelForm_WSVectorQuantizer(VectorQuantizer):
    def __init__(self, size_dict, dim_dict, kan_net1, kan_net2, cfgs, init_weights=None):
        super(DuelForm_WSVectorQuantizer, self).__init__(size_dict, dim_dict, cfgs)
        self.kl_regularization = cfgs.quantization.kl_regularization
        self.kan_net1 = kan_net1
        self.kan_net2 = kan_net2
        self.kan_lr = cfgs.quantization.kan_lr
        self.kan_iteration = cfgs.quantization.kan_iteration
        self.fixed_weight = cfgs.quantization.fixed_weight
        self.init_weights =init_weights

        self.reset_kan = True
        self.optim_kan1 = torch.optim.Adam(
            self.kan_net1.parameters(),
            lr=self.kan_lr,
            weight_decay=0.0,
            amsgrad=True,
        )

        
        self.optim_kan2 = torch.optim.Adam(
            self.kan_net2.parameters(),
            lr=self.kan_lr,
            weight_decay=0.0,
            amsgrad=True,
        )

        self.epsilon = cfgs.quantization.epsilon
        self.phi_net_troff = 1.0
        self.kl_loss = nn.KLDivLoss()
        print('---------------------------------------------------')
        print('Using DuelForm_WSVectorQuantizer')
        print('fixed_weight ', self.fixed_weight)
        print('kan_lr ', self.kan_lr)
        print('kan_iteration ', self.kan_iteration)
        print('epsilon ', self.epsilon)
        print('reset_kan ', self.reset_kan)
        print('beta ', self.beta)
        print('kl_regularization ', self.kl_regularization)
        print('---------------------------------------------------')

    def init_kan(self):
        self.kan_net1.apply(weights_init)
        self.kan_net2.apply(weights_init)

    def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
        return self._quantize_EN(z_from_encoder, codebook, codebook_weight, flg_train)
    
    def compute_OT_loss(self, ot_cost, kan_net, weight_X=None, weight_Y=None):
        phi_network = kan_net.mean(-1)
        # E_{P_y}[phi(y)]
        if weight_Y is None:
            phi_loss = torch.mean(phi_network)
        else:
            phi_loss = torch.sum(weight_Y * phi_network)

        # exp[(-d(x,y) + phi(y))/epsilon]
        exp_term = (-ot_cost + phi_network) / self.epsilon

        if weight_X is None:
            weight_X = torch.tensor(1.0 / ot_cost.shape[0])
        

        if weight_Y is None:
            OT_loss = torch.sum(weight_X*(- self.epsilon * (torch.log(torch.tensor(1.0 / exp_term.shape[1])) + torch.logsumexp(exp_term, dim=1)))) + self.phi_net_troff * phi_loss
        else:
            # using log-sum-exp trick            
            max_exp_term = exp_term.max(1)[0].clone().detach()
            sum_exp = torch.sum(weight_Y*torch.exp(exp_term-max_exp_term.unsqueeze(-1)),dim=1)
            #min_sum_exp = torch.zeros(sum_exp.size()).to(sum_exp.device)+1e-39
            #logsumexp = max_exp_term+torch.log(torch.max(sum_exp, min_sum_exp))
            logsumexp = max_exp_term+torch.log(sum_exp)
            OT_loss = torch.sum(weight_X*(- self.epsilon * (logsumexp))) + self.phi_net_troff * phi_loss

        return OT_loss

    def compute_Ensemble_OT_loss(self, ot_cost, kan_net, weight_X=None, weight_Y=None):
        phi_network = kan_net.mean(-1)
        # E_{P_y}[phi(y)]
        if weight_Y is None:
            phi_loss = torch.mean(phi_network, dim=1)
        else:
            phi_loss = torch.sum(weight_Y * phi_network, dim=1)

        # exp[(-d(x,y) + phi(y))/epsilon]

        exp_term = (-ot_cost + phi_network.unsqueeze(1)) / self.epsilon

        if weight_X is None:
            weight_X = torch.tensor(1.0 / ot_cost.shape[1])
        

        if weight_Y is None:
            #OT_loss = torch.sum(weight_X*(- self.epsilon * (torch.log(torch.tensor(1.0 / exp_term.shape[2])) + torch.logsumexp(exp_term, dim=2))),dim=1) + self.phi_net_troff * phi_loss
            OT_loss = torch.sum(weight_X*(- self.temperature * (torch.log(torch.tensor(1.0 / exp_term.shape[2])) + torch.logsumexp(exp_term, dim=2))),dim=1) + self.phi_net_troff * phi_loss
        else:
            # using log-sum-exp trick            
            max_exp_term = exp_term.max(2)[0].clone().detach()
            sum_exp = torch.sum(weight_Y.unsqueeze(1)*torch.exp(exp_term-max_exp_term.unsqueeze(-1)),dim=2)
            #min_sum_exp = torch.zeros(sum_exp.size()).to(sum_exp.device)+1e-39
            #logsumexp = max_exp_term+torch.log(torch.max(sum_exp, min_sum_exp))
            logsumexp = max_exp_term+torch.log(sum_exp)
            #OT_loss = torch.sum(weight_X*(- self.epsilon * (logsumexp)),dim=1) + self.phi_net_troff * phi_loss
            OT_loss = torch.sum(weight_X*(- self.temperature * (logsumexp)),dim=1) + self.phi_net_troff * phi_loss

        return torch.sum(OT_loss)

    def _quantize(self, z, codebook, codebook_weight, flg_train):
        size_batch = z.shape[0]
        z = z.permute(2, 3, 0, 1).contiguous()

        #import pdb; pdb.set_trace()
        num_iter = z.shape[0] * z.shape[1]

        z_flattened = z.view(-1, self.dim_dict)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1) - 2 * \
            torch.matmul(z_flattened, codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.size_dict).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        if flg_train:
  
            if self.fixed_weight:

                weight = nn.Softmax(dim=1)(codebook_weight)
                codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict
                cost_matrix=cost_matrix.reshape(64, -1, self.size_dict).repeat(1,8,1)
                
                if self.reset_kan:
                    self.init_kan()
                for i in range(0, self.kan_iteration):
                    kan_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict).repeat(1,8,1).clone().detach())
                    kan_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1).clone().detach()) 
                    
                    loss1 = -self.compute_Ensemble_OT_loss(cost_matrix.clone().detach(), kan_code_value)
                    loss2 = -self.compute_Ensemble_OT_loss(cost_matrix.permute(0,2,1).clone().detach(), kan_latent_value)
                    
                    self.optim_kan1.zero_grad()
                    loss1.backward(retain_graph=True)
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward(retain_graph=True)
                    self.optim_kan2.step()
                
                #import pdb; pdb.set_trace()            

                kan_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict).repeat(1,8,1))
                kan_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1))  

                loss1 = self.compute_Ensemble_OT_loss(cost_matrix, kan_code_value)
                loss2 = self.compute_Ensemble_OT_loss(cost_matrix.permute(0, 2, 1), kan_latent_value)
                loss = self.beta*(loss1 + loss2)

            else:
                weight = nn.Softmax(dim=1)(codebook_weight)
                codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict
                #import pdb; pdb.set_trace()            

                ensemble_cost_matrix=cost_matrix.reshape(64, -1, self.size_dict)
                
                if self.reset_kan:
                    self.init_kan()
                for i in range(0, self.kan_iteration):
                    kan2_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict).clone().detach())
                    #kan2_latent_value = self.kan_net2(z_flattened.clone().detach())
                    kan1_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1).clone().detach()) 
                    #kan1_code_value = self.kan_net1(codebook)
                    
                    #import pdb; pdb.set_trace()
                    
                    #loss1 = -self.compute_OT_loss(cost_matrix.clone().detach(), kan1_code_value, weight_Y=weight.detach().mean(0))
                    #loss2 = -self.compute_OT_loss(cost_matrix.permute(1, 0).clone().detach(), kan2_latent_value, weight_X=weight.mean(0).detach())

                    loss1 = -self.compute_Ensemble_OT_loss(ensemble_cost_matrix.clone().detach(), kan1_code_value, weight_Y=weight.detach())
                    loss2 = -self.compute_Ensemble_OT_loss(ensemble_cost_matrix.permute(0,2,1).clone().detach(), kan2_latent_value, weight_X=weight.detach())

                    self.optim_kan1.zero_grad()
                    loss1.backward(retain_graph=True)
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward(retain_graph=True)
                    self.optim_kan2.step()

                kan_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1))  
                kan_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict))
                #kan_code_value = self.kan_net1(codebook)
                #kan_latent_value = self.kan_net2(z_flattened)
                
                loss1 = self.compute_Ensemble_OT_loss(ensemble_cost_matrix, kan_code_value.detach(), weight_Y=weight)
                loss2 = self.compute_Ensemble_OT_loss(ensemble_cost_matrix.permute(0, 2, 1), kan_latent_value.detach(), weight_X=weight)
                #loss1 = self.compute_OT_loss(cost_matrix, kan_code_value.detach(), weight_Y=weight.mean(0))
                #loss2 = self.compute_OT_loss(cost_matrix.permute(1, 0), kan_latent_value.detach(), weight_X=weight.mean(0))

                loss = self.beta * (loss1 + loss2) / 64
                #import pdb; pdb.set_trace()            

                if self.kl_regularization > 0.0:
                    for i in range(64):
                        regularization_code_loss = self.kl_regularization * self.kl_loss(codeword_weight.log(), weight[i])#/10
                        loss += regularization_code_loss

                        #b1 5
                        #b2 5
                        # long1 b1. 1
                    # regularization_code_loss = self.kl_regularization * self.kl_loss(codeword_weight.unsqueeze(0).repeat(64,1).log(), weight)
                    # loss += regularization_code_loss
                    
                    #regularization_loss = self.kl_regularization * self.kl_loss(codeword_weight, weight.mean(0))
                    #loss += regularization_loss
       
        else:
            loss = 0.0
        
        z_q = z_q.view(z.shape)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(2, 3, 0, 1).contiguous()

        return z_q, loss, perplexity
  

    def _quantize_EN(self, z, codebook, codebook_weight, flg_train):
        size_batch = z.shape[0]
        z = z.permute(2, 3, 0, 1).contiguous()

        #import pdb; pdb.set_trace()
        num_iter = z.shape[0] * z.shape[1]

        z_flattened = z.view(-1, self.dim_dict)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1) - 2 * \
            torch.matmul(z_flattened, codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.size_dict).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        if flg_train:
  
            if self.fixed_weight:
                #import pdb; pdb.set_trace()            

                weight = nn.Softmax(dim=1)(codebook_weight)
                codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict
                cost_matrix=cost_matrix.reshape(64, -1, self.size_dict).repeat(1,8,1)
                
                if self.reset_kan:
                    self.init_kan()
                for i in range(0, self.kan_iteration):
                    kan_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict).repeat(1,8,1).clone().detach())
                    kan_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1).clone().detach()) 
                    
                    loss1 = -self.compute_Ensemble_OT_loss(cost_matrix.clone().detach(), kan_code_value)
                    loss2 = -self.compute_Ensemble_OT_loss(cost_matrix.permute(0,2,1).clone().detach(), kan_latent_value)
                    #loss2 = -self.compute_OT_loss(cost_matrix.permute(0,2,1).clone().detach(), kan_latent_value)
                    
                    self.optim_kan1.zero_grad()
                    loss1.backward(retain_graph=True)
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward(retain_graph=True)
                    self.optim_kan2.step()
                
                #import pdb; pdb.set_trace()            

                kan_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict).repeat(1,8,1))
                kan_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1))  

                loss1 = self.compute_Ensemble_OT_loss(cost_matrix, kan_code_value)
                loss2 = self.compute_Ensemble_OT_loss(cost_matrix.permute(0, 2, 1), kan_latent_value)
                loss = self.beta*(loss1 + loss2)

            else:
                weight = nn.Softmax(dim=1)(codebook_weight)
                codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict
                #import pdb; pdb.set_trace()            

                cost_matrix=cost_matrix.reshape(64, -1, self.size_dict)
                
                if self.reset_kan:
                    self.init_kan()
                for i in range(0, self.kan_iteration):
                    kan_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict).clone().detach())
                    kan_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1).clone().detach()) 
                    loss2 = -self.compute_Ensemble_OT_loss(cost_matrix.permute(0,2,1).clone().detach(), kan_latent_value, weight_X=weight)
                    loss1 = -self.compute_Ensemble_OT_loss(cost_matrix.clone().detach(), kan_code_value, weight_Y=weight.clone().detach())
                    
                    self.optim_kan1.zero_grad()
                    loss1.backward(retain_graph=True)
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward(retain_graph=True)
                    self.optim_kan2.step()


                kan_latent_value = self.kan_net2(z_flattened.reshape(64, -1, self.dim_dict))
                kan_code_value = self.kan_net1(codebook.unsqueeze(0).repeat(64, 1, 1))  
                loss2 = self.compute_Ensemble_OT_loss(cost_matrix.permute(0, 2, 1), kan_latent_value.detach(), weight_X=weight)
                loss1 = self.compute_Ensemble_OT_loss(cost_matrix, kan_code_value.detach(), weight_Y=weight)

                loss = self.beta*(loss1 + loss2)# / 64

                if self.kl_regularization > 0.0:
                    #import pdb; pdb.set_trace()            
                    for i in range(64):
                        regularization_code_loss = self.kl_regularization * self.kl_loss(codeword_weight.log(), weight[i])
                        loss += regularization_code_loss
                                        

                    # regularization_code_loss = self.kl_regularization * self.kl_loss(codeword_weight.unsqueeze(0).repeat(64,1).log(), weight)
                    # loss += regularization_code_loss
                    
                    #regularization_loss = self.kl_regularization * self.kl_loss(codeword_weight, weight.mean(0))
                    #loss += regularization_loss
       
        else:
            loss = 0.0
        
        z_q = z_q.view(z.shape)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(2, 3, 0, 1).contiguous()

        return z_q, loss, perplexity
    
class Global_DuelForm_WSVectorQuantizer(VectorQuantizer):
    
    def __init__(self, size_dict, dim_dict, kan_net1, kan_net2, cfgs, init_weights=None):
        super(Global_DuelForm_WSVectorQuantizer, self).__init__(size_dict, dim_dict, cfgs)

        self.kl_regularization = cfgs.quantization.kl_regularization
        self.kan_net1 = kan_net1
        self.kan_net2 = kan_net2
        self.kan_lr = cfgs.quantization.kan_lr
        self.kan_iteration = cfgs.quantization.kan_iteration
        self.fixed_weight = cfgs.quantization.fixed_weight
        self.softmax = nn.Softmax(dim=0)
        self.init_weights = init_weights

        self.reset_kan = True
        self.optim_kan1 = torch.optim.Adam(
            self.kan_net1.parameters(),
            lr=self.kan_lr,
            weight_decay=0.1,
            amsgrad=True,
        )

        
        self.optim_kan2 = torch.optim.Adam(
            self.kan_net2.parameters(),
            lr=self.kan_lr,
            weight_decay=0.1,
            amsgrad=True,
        )

        self.epsilon = cfgs.quantization.epsilon
        self.epsilon = 0.01
        self.phi_net_troff = 1.0
        self.kl_loss = nn.KLDivLoss()

        print('---------------------------------------------------')
        print('Using Global_DuelForm_WSVectorQuantizer')
        print('fixed_weight ', self.fixed_weight)
        print('kan_lr ', self.kan_lr)
        print('kan_iteration ', self.kan_iteration)
        print('epsilon ', self.epsilon)
        print('reset_kan ', self.reset_kan)
        print('beta ', self.beta)
        print('kl_regularization ', self.kl_regularization)
        print('---------------------------------------------------')


    def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
        return self._current_quantize(z_from_encoder, codebook, codebook_weight, flg_train)
    

    def init_kan(self):
        self.kan_net1.apply(weights_init)
        self.kan_net2.apply(weights_init)


    def compute_OT_loss(self, ot_cost, kan_net, weight_X=None, weight_Y=None):
        phi_network = kan_net.mean(-1)
        # E_{P_y}[phi(y)]
        if weight_Y is None:
            phi_loss = torch.mean(phi_network)
        else:
            phi_loss = torch.sum(weight_Y * phi_network)

        # exp[(-d(x,y) + phi(y))/epsilon]
        exp_term = (-ot_cost + phi_network) / self.epsilon

        if weight_X is None:
            weight_X = torch.tensor(1.0 / ot_cost.shape[0])
        

        if weight_Y is None:
            OT_loss = torch.sum(weight_X*(- self.epsilon * (torch.log(torch.tensor(1.0 / exp_term.shape[1])) + torch.logsumexp(exp_term, dim=1)))) + self.phi_net_troff * phi_loss
            # OT_loss = torch.sum(weight_X*(- self.temperature * (torch.log(torch.tensor(1.0 / exp_term.shape[1])) + torch.logsumexp(exp_term, dim=1)))) + self.phi_net_troff * phi_loss
        else:
            # using log-sum-exp trick            
            max_exp_term = exp_term.max(1)[0].clone().detach()
            sum_exp = torch.sum(weight_Y*torch.exp(exp_term-max_exp_term.unsqueeze(-1)),dim=1)
            #min_sum_exp = torch.zeros(sum_exp.size()).to(sum_exp.device)+1e-39
            #logsumexp = max_exp_term+torch.log(torch.max(sum_exp, min_sum_exp))
            logsumexp = max_exp_term + torch.log(sum_exp)
            OT_loss = torch.sum(weight_X*(- self.epsilon * (logsumexp))) + self.phi_net_troff * phi_loss
            
            # OT_loss = torch.sum(weight_X*(- self.temperature * (logsumexp))) + self.phi_net_troff * phi_loss
        return OT_loss

    def _quantize(self, z, codebook, codebook_weight, flg_train):

        z = z.permute(2, 3, 0, 1).contiguous()

        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        #import pdb; pdb.set_trace()
        #code_matrix = F.cosine_similarity(codebook.unsqueeze(1), codebook.unsqueeze(0), dim=2)
        code_matrix = torch.sum(codebook ** 2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(codebook, codebook.t())
        zeros = torch.zeros(code_matrix.shape).to(z.device)+1e-3
        codebook_regularization = torch.min(zeros, code_matrix).mean()

        # increasing samples to make duel-form work more stable
        multiplier = (z_flattened.shape[0]//self.size_dict)
        repeat_value = torch.ones(self.size_dict).to(z.device).int() * multiplier
        codebook = torch.repeat_interleave(codebook, repeat_value, dim=0)
        current_size_dict = self.size_dict * multiplier
        
        cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1) - 2 * \
            torch.matmul(z_flattened, codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], current_size_dict).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        if self.reset_kan:
            self.init_kan()
        if flg_train:
            #import pdb; pdb.set_trace()            
            if self.fixed_weight:
                if self.init_weights is None:
                    codeword_weight = torch.ones(current_size_dict).to(z.device) / current_size_dict
                else:
                    codeword_weight = (nn.Softmax(dim=0)(torch.ones(self.size_dict) * self.init_weights)).to(z.device) 
                    codeword_weight = torch.repeat_interleave(codeword_weight / multiplier, repeat_value, dim=0)
                                    
                for i in range(0, self.kan_iteration):
                    kan2_latent_value = self.kan_net2(z_flattened.clone().detach())
                    kan1_code_value = self.kan_net1(codebook.clone().detach())
                    
                    loss1 = -self.compute_OT_loss(cost_matrix.clone().detach(), kan1_code_value, weight_Y=codeword_weight)
                    loss2 = -self.compute_OT_loss(cost_matrix.permute(1, 0).clone().detach(), kan2_latent_value, weight_X=codeword_weight)
                    
                    self.optim_kan1.zero_grad()
                    loss1.backward() #loss1 = f(kan2)
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward() #loss2 = f(kan1)
                    self.optim_kan2.step()

                
                kan2_latent_value = self.kan_net2(z_flattened)
                kan1_code_value = self.kan_net1(codebook)
                loss1 = self.compute_OT_loss(cost_matrix, kan1_code_value, weight_Y=codeword_weight)
                loss2 = self.compute_OT_loss(cost_matrix.permute(1, 0), kan2_latent_value, weight_X=codeword_weight)
                loss = self.beta * (loss1 + loss2)

            else:
                weight = nn.Softmax()(codebook_weight)
                weight = torch.repeat_interleave(weight / multiplier, repeat_value, dim=0)

                codeword_weight = torch.ones(current_size_dict).to(z.device) / current_size_dict

                for i in range(0, self.kan_iteration):
                    kan2_latent_value = self.kan_net2(z_flattened.clone().detach())
                    kan1_code_value = self.kan_net1(codebook.clone().detach())
                    
                    loss1 = -self.compute_OT_loss(cost_matrix.clone().detach(), kan1_code_value, weight_Y=weight.clone().detach())
                    loss2 = -self.compute_OT_loss(cost_matrix.permute(1, 0).clone().detach(), kan2_latent_value, weight_X=weight.clone().detach())
                    
                    self.optim_kan1.zero_grad()
                    loss1.backward()
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward()
                    self.optim_kan2.step()
                
                kan2_latent_value = self.kan_net2(z_flattened)
                kan1_code_value = self.kan_net1(codebook)
                loss1 = self.compute_OT_loss(cost_matrix, kan1_code_value.detach(), weight_Y=weight)
                loss2 = self.compute_OT_loss(cost_matrix.permute(1, 0), kan2_latent_value.detach(), weight_X=weight)
                loss = self.beta * (loss1 + loss2)

                if self.kl_regularization > 0.0:
                    regularization_loss = self.kl_regularization * self.kl_loss(codeword_weight, weight)
                    loss += regularization_loss

                loss -= self.beta * codebook_regularization
       
        else:
            loss = 0.0
        
        z_q = z_q.view(z.shape)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(2, 3, 0, 1).contiguous()

        return z_q, loss, perplexity

    def _current_quantize(self, z, codebook, codebook_weight, flg_train):

        # import pdb; pdb.set_trace()
        # size_dict = 256
        # aa=torch.randint(0,512,(size_dict,))
        # partial_codebook = codebook[aa,:]

        z = z.permute(2, 3, 0, 1).contiguous()

        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1) - 2 * \
            torch.matmul(z_flattened, codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.size_dict).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        if self.reset_kan:
            self.init_kan()
        if flg_train:

               
            
            if self.fixed_weight:
                if self.init_weights is None:
                    codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict
                else:
                    codeword_weight = (nn.Softmax(dim=0)(torch.ones(self.size_dict) * self.init_weights)).to(z.device) 
                
                for i in range(0, self.kan_iteration):
                    kan2_latent_value = self.kan_net2(z_flattened.clone().detach())
                    kan1_code_value = self.kan_net1(codebook.clone().detach())
                    
                    loss1 = -self.compute_OT_loss(cost_matrix.clone().detach(), kan1_code_value, weight_Y=codeword_weight)
                    
                    loss2 = -self.compute_OT_loss(cost_matrix.permute(1, 0).clone().detach(), kan2_latent_value, weight_X=codeword_weight)
                    
                    self.optim_kan1.zero_grad()
                    loss1.backward() #loss1 = f(kan2)
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward() #loss2 = f(kan1)
                    self.optim_kan2.step()

                    #import pdb; pdb.set_trace()
                
                kan2_latent_value = self.kan_net2(z_flattened)
                kan1_code_value = self.kan_net1(codebook)
                loss1 = self.compute_OT_loss(cost_matrix, kan1_code_value, weight_Y=codeword_weight)
                loss2 = self.compute_OT_loss(cost_matrix.permute(1, 0), kan2_latent_value, weight_X=codeword_weight)
                loss = self.beta * (loss1 + loss2)

            else:
                # if optimize codebook_weight, we always initialize it as uniform
                weight = nn.Softmax()(codebook_weight)
                codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict

                for i in range(0, self.kan_iteration):
                    kan2_latent_value = self.kan_net2(z_flattened.clone().detach())
                    kan1_code_value = self.kan_net1(codebook.clone().detach())
                    
                    loss1 = -self.compute_OT_loss(cost_matrix.clone().detach(), kan1_code_value, weight_Y=weight.clone().detach())
                    loss2 = -self.compute_OT_loss(cost_matrix.permute(1, 0).clone().detach(), kan2_latent_value, weight_X=weight.clone().detach())
                    
                    self.optim_kan1.zero_grad()
                    loss1.backward() #loss1 = f(kan2)
                    self.optim_kan1.step()

                    self.optim_kan2.zero_grad()
                    loss2.backward() #loss2 = f(kan1)
                    self.optim_kan2.step()
                
                kan2_latent_value = self.kan_net2(z_flattened)
                kan1_code_value = self.kan_net1(codebook)
                loss1 = self.compute_OT_loss(cost_matrix, kan1_code_value.detach(), weight_Y=weight)
                loss2 = self.compute_OT_loss(cost_matrix.permute(1, 0), kan2_latent_value.detach(), weight_X=weight)
                loss = self.beta * (loss1 + loss2)

                if self.kl_regularization > 0.0:
                    regularization_loss = self.kl_regularization * self.kl_loss(codeword_weight, weight)
                    loss += regularization_loss
       
        else:
            loss = 0.0
        
        z_q = z_q.view(z.shape)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(2, 3, 0, 1).contiguous()

        return z_q, loss, perplexity


class WSVectorQuantizer(VectorQuantizer):
    def __init__(self, size_dict, dim_dict, cfgs, init_weights=None):
        super(WSVectorQuantizer, self).__init__(size_dict, dim_dict, cfgs)

        self.kl_loss = nn.KLDivLoss()
        self.kl_regularization = cfgs.quantization.kl_regularization
        self.global_optimization = cfgs.quantization.global_optimization
        self.fixed_weight = cfgs.quantization.fixed_weight
        self.init_weights = init_weights
        self.softmax = nn.Softmax(dim=0)
        print('---------------------------------------------------')
        print('Using WSVectorQuantizer')
        print('fixed_weight ', self.fixed_weight)
        print('global_optimization ', self.global_optimization)
        print('beta ', self.beta)
        print('kl_regularization ', self.kl_regularization)
        print('---------------------------------------------------')


    def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
        return self._quantize(z_from_encoder,codebook, codebook_weight, flg_train)
    

    def _quantize(self, z, codebook, codebook_weight, flg_train):
        z = z.permute(2, 3, 0, 1).contiguous()

        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1) - 2 * \
            torch.matmul(z_flattened, codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.size_dict).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        if flg_train:
            if self.global_optimization:
                size_batch = z_flattened.shape[0]
                sample_weight = torch.ones(size_batch).to(z.device) / size_batch

                if self.fixed_weight:
                    loss = self.beta * ot.emd2(codeword_weight, sample_weight, cost_matrix.t(), numItermax=500000)
                else:
                    # for regularization
                    codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict
                    weight = self.softmax(codebook_weight.t())
                    loss = self.beta * ot.emd2(weight, sample_weight, cost_matrix.t(), numItermax=500000)
                    if self.kl_regularization > 0.0:
                        loss += self.kl_regularization * self.kl_loss(codeword_weight.log(), weight)

            else:  

                loss = 0.0
                size_batch = z.shape[2]
                num_iter = z.shape[0] * z.shape[1]

                sample_weight = torch.ones(size_batch).to(z.device) / size_batch
                
                if self.fixed_weight:
                    if self.init_weights is None:
                        codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict
                    else:
                        codeword_weight = (nn.Softmax(dim=0)(torch.ones(self.size_dict) * self.init_weights)).to(z.device)

                    for i in range(num_iter):
                        loss += self.beta * ot.emd2(codeword_weight, sample_weight, cost_matrix[size_batch*i:size_batch*(i+1)].t()) 
                else:
                    codeword_weight = torch.ones(self.size_dict).to(z.device) / self.size_dict 
                    weight = nn.Softmax(dim=1)(codebook_weight)
                    
                    for i in range(num_iter):
                        loss += self.beta * ot.emd2(weight[i], sample_weight, cost_matrix[size_batch*i:size_batch*(i+1)].t())
                        if self.kl_regularization > 0.0:
                            loss += self.kl_regularization * self.kl_loss(codeword_weight.log(), weight[i])
    
                    
        else:
            loss = 0.0
        
        z_q = z_q.view(z.shape)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(2, 3, 0, 1).contiguous()

        return z_q, loss, perplexity



class GaussianVectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=0.5, param_var_q="gaussian_1"):
        super(GaussianVectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
        self.param_var_q = param_var_q

    def forward(self, z_from_encoder, param_q, codebook, flg_train, flg_quant_det=False):
        return self._quantize(z_from_encoder, param_q, codebook, flg_train=flg_train, flg_quant_det=flg_quant_det)
    
    def _quantize(self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        
        precision_q = 1. / torch.clamp(var_q, min=1e-10)
        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook, 0.5 * precision_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device="cuda")
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity


    def _inference(self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        precision_q = 1. / torch.clamp(var_q, min=1e-10)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook, 0.5 * precision_q)
        
        min_encoding_indices = torch.argmax(logit, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.size_dict, device="cuda")
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        z_quantized = torch.matmul(min_encodings, codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()

        # Latent loss
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
        loss = kld_discrete + kld_continuous
        
        return z_to_decoder, min_encodings, min_encoding_indices, loss

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):        
        distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        return distances
        
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1-x2)**2 * weight, dim=(1,2,3))

    def set_temperature(self, value):
        self.temperature = value
