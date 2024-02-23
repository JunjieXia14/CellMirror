import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import math
from tqdm import tqdm
from tqdm.autonotebook import trange

from layers import *

#cVAE_Pytorch_version
class cVAE(nn.Module):
    def __init__(
        self,
        args,
        n_input: int,
        dropout_rate: float = 0
    ):
        super().__init__()
        self.args = args
        self.n_input = n_input
        self.n_hidden_de = args.n_hidden_de
        self.n_latent_s = args.n_latent_s
        self.n_lantent_z = args.n_latent_z
        self.beta = args.beta
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.last_batch_size = args.last_batch_size

        self.s_encoder = Encoder(
            n_input = n_input,
            n_output = args.n_latent_s,
            n_hidden = [args.n_hidden_en] if isinstance(args.n_hidden_en, int) else args.n_hidden_en,
            dropout_rate = dropout_rate
        )
        
        self.z_encoder = Encoder(
            n_input = n_input,
            n_output = args.n_latent_z,
            n_hidden = [args.n_hidden_en] if isinstance(args.n_hidden_en, int) else args.n_hidden_en,
            dropout_rate = dropout_rate 
        )

        self.decoder = Decoder(
            n_input = args.n_latent_s + args.n_latent_z,
            n_output = n_input,
            n_hidden = [args.n_hidden_de] if isinstance(args.n_hidden_de, int) else args.n_hidden_de,
            dropout_rate = dropout_rate
        )

        self.discriminator = Discriminator(
            n_input = args.n_latent_s + args.n_latent_z,
            n_output = 1,
            use_sigmoid = True,
            dropout_rate = dropout_rate
        )

    def get_s_latents(
        self, dataloader, return_s_mean = True
    ) :
        tg_s_outputs = []
        bg_s_outputs = []

        for batch_idx, (tg, bg) in enumerate(dataloader):

            if self.args.use_cuda and torch.cuda.is_available():
                tg, bg = tg.cuda(), bg.cuda()
            
            tg = Variable(tg)
            bg = Variable(bg)

            outputs = self.inference(tg, bg)

            if return_s_mean:
                tg_s = outputs["tg_s_mean"]
                bg_s = outputs["bg_s_mean"]
            else:
                tg_s = outputs["tg_s"]
                bg_s = outputs["bg_s"]
            
            tg_s_outputs.append( tg_s.detach().cpu().data )
            bg_s_outputs.append( bg_s.detach().cpu().data )

        tg_s_outputs = torch.cat(tg_s_outputs).numpy()
        bg_s_outputs = torch.cat(bg_s_outputs).numpy()

        return dict( tg_s_outputs = tg_s_outputs, bg_s_outputs = bg_s_outputs)

    def get_z_latents(
        self, dataloader, return_z_mean = True
    ) :
        tg_z_outputs = []
        bg_z_outputs = []

        for batch_idx, (tg, bg) in enumerate(dataloader):

            if self.args.use_cuda and torch.cuda.is_available():
                tg, bg = tg.cuda(), bg.cuda()
            
            tg = Variable(tg)
            bg = Variable(bg)

            outputs = self.inference(tg, bg)

            if return_z_mean:
                tg_z = outputs["tg_z_mean"]
                bg_z = outputs["bg_z_mean"]
            else:
                tg_z = outputs["tg_z"]
                bg_z = outputs["bg_z"]
            
            tg_z_outputs.append( tg_z.detach().cpu().data )
            bg_z_outputs.append( bg_z.detach().cpu().data )

        tg_z_outputs = torch.cat(tg_z_outputs).numpy()
        bg_z_outputs = torch.cat(bg_z_outputs).numpy()

        return dict( tg_z_outputs = tg_z_outputs, bg_z_outputs = bg_z_outputs)

    def inference(self, tg, bg):

        tg_s_mean, tg_s_var, tg_s = self.s_encoder(tg)
        tg_z_mean, tg_z_var, tg_z = self.z_encoder(tg)
        bg_z_mean, bg_z_var, bg_z = self.z_encoder(bg)
        bg_s_mean, bg_s_var, bg_s = self.s_encoder(bg)

        tg_outputs = self.decoder(torch.cat([tg_z, tg_s], dim=1))
        zeros = torch.zeros_like(bg_s)
        bg_outputs = self.decoder(torch.cat([bg_z, zeros], dim=1))

        return dict(tg_s_mean = tg_s_mean, tg_s_var = tg_s_var, tg_s = tg_s,
                    tg_z_mean = tg_z_mean, tg_z_var = tg_z_var, tg_z = tg_z,
                    bg_s_mean = bg_s_mean, bg_s_var = bg_s_var, bg_s = bg_s,
                    bg_z_mean = bg_z_mean, bg_z_var = bg_z_var, bg_z = bg_z,
                    tg_outputs = tg_outputs, bg_outputs = bg_outputs)

    def get_cVAE_loss(self, tg, bg):
        
        outputs = self.inference(tg, bg)

        #Reconstruction Loss
        tg_outputs = outputs["tg_outputs"]
        bg_outputs = outputs["bg_outputs"]

        loss = nn.MSELoss()
        reconstruction_loss = loss(tg_outputs, tg)
        reconstruction_loss += loss(bg_outputs, bg)
        reconstruction_loss *= self.n_input

        #KL Divergence
        tg_z_mean = outputs["tg_z_mean"]
        tg_z_var = outputs["tg_z_var"]
        tg_s_mean = outputs["tg_s_mean"]
        tg_s_var = outputs["tg_s_var"]
        bg_z_mean = outputs["bg_z_mean"]
        bg_z_var = outputs["bg_z_var"]

        kl_loss = torch.sum((1 + torch.log(tg_z_var) - tg_z_mean.pow(2) - tg_z_var), dim=1)
        kl_loss += torch.sum((1 + torch.log(tg_s_var) - tg_s_mean.pow(2) - tg_s_var), dim=1)
        kl_loss += torch.sum((1 + torch.log(bg_z_var) - bg_z_mean.pow(2) - bg_z_var), dim=1)
        kl_loss *= -0.5

        #Total Relation Loss & Discriminator Loss
        tg_z = outputs["tg_z"]
        tg_s = outputs["tg_s"]

        z1 = Lambda(lambda x: x[:int(self.batch_size/2), :])(tg_z)
        z2 = Lambda(lambda x: x[int(self.batch_size/2):, :])(tg_z)
        s1 = Lambda(lambda x: x[:int(self.batch_size/2), :])(tg_s)
        s2 = Lambda(lambda x: x[int(self.batch_size/2):, :])(tg_s)

        if z1.shape[0] != z2.shape[0]:
            z1 = Lambda(lambda x: x[:int(self.last_batch_size/2), :])(tg_z)
            z2 = Lambda(lambda x: x[int(self.last_batch_size/2):, :])(tg_z)
            s1 = Lambda(lambda x: x[:int(self.last_batch_size/2), :])(tg_s)
            s2 = Lambda(lambda x: x[int(self.last_batch_size/2):, :])(tg_s)

        q_bar = torch.cat([torch.cat([s1, z2], dim=1), torch.cat([s2, z1], dim=1)], dim=0)
        q = torch.cat([torch.cat([s1, z1], dim=1), torch.cat([s2, z2], dim=1)], dim=0)

        q_bar_score = self.discriminator(q_bar)
        q_score = self.discriminator(q)

        tc_loss = torch.log(q_score / (1 - q_score))
        discriminator_loss = - torch.log(q_score) - torch.log(1 - q_bar_score)

        cVAE_loss = torch.mean(reconstruction_loss) + self.beta * torch.mean(kl_loss) + self.gamma * torch.mean(tc_loss) + torch.mean(discriminator_loss)

        return torch.mean(reconstruction_loss), self.beta * torch.mean(kl_loss), self.gamma * torch.mean(tc_loss), torch.mean(discriminator_loss), cVAE_loss

    def forward(self, tg, bg):

        outputs = self.inference(tg, bg)

        return outputs

    def predict(self, dataloader):

        noContamination_outputs = []
        bg_outputs = []

        tg_s_outputs = []
        tg_z_outputs = []
        bg_s_outputs = []
        bg_z_outputs = []

        for batch_idx, (tg, bg) in enumerate(dataloader):

            if self.args.use_cuda and torch.cuda.is_available():
                tg, bg = tg.cuda(), bg.cuda()

            tg = Variable(tg)
            bg = Variable(bg)

            outputs = self.inference(tg, bg)

            bg_outputs.append( outputs["bg_outputs"].detach().cpu().data )

            zeros = torch.zeros_like( outputs["tg_s"] )
            noContamination_output = self.decoder( torch.cat([outputs["tg_z"], zeros], dim=1) )
            noContamination_outputs.append( noContamination_output.detach().cpu().data )

            tg_s_outputs.append( outputs["tg_s"].detach().cpu().data )
            tg_z_outputs.append( outputs["tg_z"].detach().cpu().data )
            bg_s_outputs.append( outputs["bg_s"].detach().cpu().data )
            bg_z_outputs.append( outputs["bg_z"].detach().cpu().data )

        noContamination_outputs = torch.cat(noContamination_outputs).numpy()
        bg_outputs = torch.cat(bg_outputs).numpy()

        tg_s_outputs = torch.cat(tg_s_outputs).numpy()
        tg_z_outputs = torch.cat(tg_z_outputs).numpy()
        bg_s_outputs = torch.cat(bg_s_outputs).numpy()
        bg_z_outputs = torch.cat(bg_z_outputs).numpy()

        return dict( noContamination_outputs = noContamination_outputs,
                     bg_outputs = bg_outputs,
                     tg_s_outputs = tg_s_outputs,
                     tg_z_outputs = tg_z_outputs,
                     bg_s_outputs = bg_s_outputs,
                     bg_z_outputs = bg_z_outputs)

    def fit(self, train_loader, total_loader):

        params    = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam( params, lr = self.args.lr_cLDVAE, weight_decay = self.args.weight_decay, eps = self.args.eps )

        loss_list=[]
        patience_epoch = 0
        reco_epoch_test = 0
        loss_like_min = 1000000000
        flag_break = 0

        start = time.time()

        with trange(1, self.args.max_epoch + 1, total=self.args.max_epoch, desc='Epochs') as tq:
            for epoch in tq:

                self.train()
                optimizer.zero_grad()

                for batch_idx, (tg, bg) in enumerate(train_loader):

                    if self.args.use_cuda and torch.cuda.is_available():
                        tg, bg = tg.cuda(), bg.cuda()

                    tg = Variable(tg)
                    bg = Variable(bg)

                    _, _, _, _, loss= self.get_cVAE_loss(tg, bg)

                    loss.backward()
                    optimizer.step()

                if epoch % self.args.epoch_per_test == 0 and epoch > 0:
                    self.eval()
                    with torch.no_grad():
                        
                        '''
                        loss
                        '''

                        test_loss = 0
                        for batch_idx, (tg, bg) in enumerate(total_loader):

                            if self.args.use_cuda and torch.cuda.is_available():
                                tg, bg = tg.cuda(), bg.cuda()

                            tg = Variable(tg)
                            bg = Variable(bg)


                            _, _, _, _, loss = self.get_cVAE_loss(tg, bg)

                            if math.isnan(loss.item()):
                                flag_break = 1
                                break

                            test_loss += loss.item()
                        test_loss = test_loss / (batch_idx+1)

                        loss_list.append(test_loss)

                        if  test_loss < loss_like_min:
                            loss_like_min = test_loss
                            reco_epoch_test = epoch
                            patience_epoch = 0

                    if flag_break == 1:
                        print('containin NA')
                        print(epoch)
                        break

                    if patience_epoch >=30 :
                        print('patient with 40')
                        print(epoch, loss_list[-1])
                        break

                    if len(loss_list)>=2:
                        relative_loss = ( loss_list[-1] - loss_list[-2] ) / loss_list[-2]
                        # print(epoch, loss_list[-1], relative_loss)

                        if abs(relative_loss) < 1e-4 and test_loss < loss_like_min:
                            print('converged!!!')
                            break

        duration = time.time() - start

        print(f"Finish training, total time is: {duration}s")
        self.eval()
        print(self.training)

        print(f'loss likelihood is: {loss_like_min}, epoch: {reco_epoch_test}')

        return dict(duration=duration,
                    epoch=epoch,
                    loss=loss_list[-1])