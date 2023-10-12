# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:59:37 2022
    pseudo-Unconditional GAN with double backpropagation for DSF discriminator
@author: roberto.miele
"""
import os
import shutil
import argparse
import json
import torch
from torch import nn, optim
import torchvision
import numpy as np
import time
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
from GAN_module import Generator, DiscriminatorWNet
from FMclasses import ForwardModeling, ElasticModels
from Gslib import Gslib
from Custom_dataloader import FaciesSeismicDataset
pd.options.mode.chained_assignment = None  # default='warn'


def precompute_Seismic(args, Ip_models, FM):
    print('Pre-computing TI Seismic')
    import subprocess
    import os 
    if args.precomputed==True: 
        print('Already pre-computed')
        return FaciesSeismicDataset(args.project_path+args.TI_path,args.nsim)
    else:
        if args.ip_type!=0:
            if os.path.isdir(args.project_path+args.TI_path+'/Seismic_TI'): 
                shutil.rmtree(args.project_path+args.TI_path+'/Seismic_TI')
            if os.path.isdir(args.project_path+args.TI_path+'/Facies_TI'): 
                shutil.rmtree(args.project_path+args.TI_path+'/Facies_TI')
    
        #load TI dataset
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomInvert(1),
            torchvision.transforms.RandomRotation([180,180])
            ])
        dataset = torchvision.datasets.ImageFolder(root=args.project_path+args.TI_path, transform=transforms)
        N= len(dataset)
        dataloader= torch.utils.data.DataLoader(dataset=dataset, batch_size=N)
        
        os.mkdir(args.project_path+args.TI_path+'/Seismic_TI')
        os.mkdir(args.project_path+args.TI_path+'/Facies_TI') 
            
        for _,data in enumerate(dataloader):
            data= data[0][:,0,None,:,:]
            Ip_models.writeallfac_dss(data.detach().cpu().numpy())  #write TI facies in gslib format
            
            Ip_models.simulations= torch.zeros((args.nsim, 1, data.shape[2], data.shape[3])) 
            for i in range(N):
                Ip_models.write_parfile(i,'unc',args.nsim) #write parfile
                subprocess.run(args=[f'{Ip_models.inf}DSS.C.64.exe', f'{Ip_models.inf}ssdir.par'], stdout=subprocess.DEVNULL) #run DSS
                
                for ssi in range (args.nsim):
                    #read simulation
                    Ip_models.simulations[ssi]= torch.from_numpy(np.reshape(Gslib().Gslib_read(f'{Ip_models.ouf}/dss/ip_real_{ssi+1}.out').data.values.squeeze(),
                                  (1, args.nz, args.nx)))
                
                #calculate seismic
                syn_TI= FM.calc_synthetic(Ip_models.simulations.detach()).detach()
                
                for ssi in range (args.nsim): 
                    torch.save(syn_TI[ssi], args.project_path+args.TI_path+f'/Seismic_TI/{i}_{ssi}.pt')
                dtw= torch.from_numpy(data[i].detach().cpu().numpy())
                torch.save(dtw, args.project_path+args.TI_path+f'/Facies_TI/{i}.pt')
        
        dataset= FaciesSeismicDataset(args.project_path+args.TI_path,args.nsim)
        return dataset

def pplot(path, model, typ='imshow', label=None, cmap=None, vmax=None, vmin=None):
    plt.figure()
    if typ=='imshow': 
        plt.imshow(model, label=label, cmap=cmap, vmax=vmax, vmin=vmin)
        plt.colorbar()
        
    if typ=='hist': 
        plt.hist(model, label=label)
    
    plt.savefig(path)
    plt.close()
    return None
    
def main(args):
    
    def CLoss(image):
        # choose a mask as conditional information, generate bz images, compute the error 
        # between image mask and mask
        masked = conditioning_data_G!=0
        masked= masked*image
        loss = torch.norm((masked - conditioning_data_G.detach()).reshape(-1, args.nz * args.nx), p=1, dim=1, keepdim=True).mean()
        return loss
    
    def GDoptim(learning_rate):
            #NN optimizers
        G_optim = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-5)
        D_FS_optim = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-5)
                        
        return G_optim, D_FS_optim
    
    tottime=0
    Losses_Epoch = pd.DataFrame({'Loss D':[],'LossD FS':[],'LossD F':[], 'LossD S':[], 
                                 'Loss G':[], 'LossG FS':[], 'LossG F':[], 
                                 'Cont_loss':[]
                                 })
    
    Scores_Epoch = pd.DataFrame({'ScoreD FS':[], 'ScoreD F':[], 'ScoreD S':[], 
                                 'ScoreG FS':[], 'ScoreG F':[],
                                 'MSE_seis':[]
                                 })
    
    Losses_Batch = pd.DataFrame({'Loss D':[],'LossD FS':[],'LossD F':[], 'LossD S':[], 
                                 'Loss G':[], 'LossG FS':[], 'LossG F':[], 
                                 'Cont_loss':[]
                                 })
    
    Scores_Batch = pd.DataFrame({'ScoreD FS':[], 'ScoreD F':[], 'ScoreD S':[], 
                                 'ScoreG FS':[], 'ScoreG F':[],
                                 })

    device = args.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.type_of_FM= 'fullstack'

    G = Generator().to(device)
    D = DiscriminatorWNet().to(device)
        
    print('The architecture of G and D', G, D)
    params_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
    params_D_SF = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(params_D_SF, params_G)

    loss = nn.BCELoss().to(device)
    
    well_data= Gslib().Gslib_read(args.project_path + args.in_folder+args.well_data, ).data
    well_data= well_data[well_data!=args.null_val].dropna()
    
    #separate ip by facies
    bounds_zones={}
    def write_zones(filename, zone):
        with open(args.project_path + args.in_folder + filename, 'w') as ffid:
            ffid.write(filename+'\n')
            ffid.write('4\n')
            ffid.write('x\n')
            ffid.write('y\n')
            ffid.write('z\n')
            ffid.write('Ip\n')
            for lin in range(len(zone)):
                ffid.write(' '.join(zone.loc[lin].astype(str))+'\n')
    
    y_idx= 29 #slice seismic
    if y_idx+1 in well_data.j_index.values:
        well_data= well_data[well_data.j_index==y_idx+1]
        well_data.j_index=1
        typ='cond'
    else: 
        typ='unc'

    for fac in range(args.n_facies):
        zone= well_data[['i_index','j_index','k_index','Ip']][well_data.Facies==fac].reset_index(drop=True)
        
        #for DSS with wells
        filename= f'Ip_zone{fac}_cond.out'
        write_zones(filename, zone)
        bounds_zones[fac]= np.array([zone.Ip.min(),zone.Ip.max()])

        #save for DSS without wells
        zone.loc[:,'j_index']=1000
        filename= f'Ip_zone{fac}_unc.out'
        write_zones(filename, zone)
    
    #load seismic
    seismic= Gslib().Gslib_read(args.project_path + args.in_folder + args.real_seismic_file).data
    seismic= np.reshape(seismic.values,(args.nx,args.ny,args.nz),'F')#/2000
    
    #Init classes
    Ip_models= ElasticModels(args, ipmax=well_data.Ip.max(), ipmin=well_data.Ip.min(), ipzones=bounds_zones)
       
    FM= ForwardModeling(args)
    FM.load_wavelet(args)
    FM.real_seismic= torch.zeros((args.batch_size,1,args.nz,args.nx)).to(device).detach()
    
    seismic = seismic.squeeze()
    if seismic.ndim!=2: seismic= np.flip(seismic[:,y_idx,:].T, axis=0).copy()
    
    args.max_abs_amp= np.abs(seismic).max()
  
    FM.real_seismic[:,:]=torch.tensor(seismic.T/args.max_abs_amp)
    
    
    if (args.precompute_TI_seismic==False) or (args.ip_type==0):
        if os.path.isdir(args.project_path+args.TI_path+'/Seismic_TI'): 
            shutil.rmtree(args.project_path+args.TI_path+'/Seismic_TI')
        if os.path.isdir(args.project_path+args.TI_path+'/Facies_TI'): 
            shutil.rmtree(args.project_path+args.TI_path+'/Facies_TI')

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomInvert(p=1),
            torchvision.transforms.Normalize([0.5], [0.5]),
            
            torchvision.transforms.RandomRotation([180,180])
            ])
        dataset = torchvision.datasets.ImageFolder(root=args.project_path+args.TI_path, transform=transforms)
    
    else: dataset= precompute_Seismic(args,Ip_models,FM)

    dataloader= torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    pplot(args.project_path+args.out_folder+'/real_seismic', FM.real_seismic.cpu().numpy()[0,0],
          typ='imshow', label='Amplitude', cmap='seismic')
    
    if (args.alpha_c!=None) & (typ=='cond'): 
        
        fac_well= well_data[['i_index','k_index','Facies']]
        fac_well.loc[:,'k_index']-=1
        fac_well.loc[:,'i_index']-=1
        fac_well.Facies[fac_well.Facies==0]=-1
        
        temp= np.zeros((args.nz, args.nx))
        temp[fac_well.k_index,fac_well.i_index]= fac_well.Facies

        conditioning_data_G= torch.zeros_like(FM.real_seismic)
        conditioning_data_G[:,:]= torch.from_numpy(temp)
        
        del temp, fac_well
    else: 
        typ= 'unc'
    score_FS = torch.ones((1,))
    G_optim, D_optim = GDoptim(args.lr)
    Glr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=G_optim, gamma=args.decayRate, step_size=args.lr_steps)
    Dlr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=D_optim, gamma=args.decayRate, step_size=args.lr_steps)
    score_FS= torch.ones((1,))
    for epoch in range(args.epochs):
        
        start = time.time()

        for batch_idx, data in enumerate(dataloader):
            if (args.precompute_TI_seismic==False) or (args.ip_type==0):
                fac_TI= data[0][:,0,None,:,:].to(device)
                if args.ip_type==1: Ip_models.run_dss(fac_TI.detach(), nsim=1).to(device)
                else: Ip_models.det_Ip(fac_TI.detach()).to(device)
                FM.syn_seismic= FM.calc_synthetic(Ip_models.simulations.detach()).detach().cpu()/args.max_abs_amp
                FM.syn_seismic= FM.syn_seismic.to(device)
                realseis = FM.real_seismic.to(device)
            else: 
                fac_TI= data[0].to(device)
                FM.syn_seismic= (data[1]/args.max_abs_amp).to(device)
                realseis = FM.real_seismic.to(device)
                
            mini_batch_size= fac_TI.shape[0]
            mini_batch_size= fac_TI.shape[0]
           
            real = torch.Tensor(np.random.uniform(1,1,size=(mini_batch_size,1))).float().to(device)
            fake = torch.Tensor(np.random.uniform(0,0,size=(mini_batch_size,1))).float().to(device)
            
            # training D
            z = torch.randn(mini_batch_size, 100).to(device) 
            fake_image = G(z).detach()

            fac_TI= torch.randn_like(fac_TI)*0.1+fac_TI
            fake_image= torch.randn_like(fake_image)*0.1+fake_image

            score_F_TI, score_FS_TI,score_S_TI = D(fac_TI, FM.syn_seismic)
            score_F_fake, score_FS_fake,score_S_fake = D(fake_image, FM.syn_seismic)
            
            loss_D_FS= loss(score_FS_TI, real) + loss(score_FS_fake, fake)
            loss_D_S= loss(score_S_TI, real) + loss(score_S_fake,real)
            loss_D_F= loss(score_F_TI,real) + loss(score_F_fake,fake)
            
            loss_D= loss_D_FS+loss_D_S+loss_D_F
            
            D_optim.zero_grad()
            loss_D.backward()            
            D_optim.step()

                      
            del FM.syn_seismic            
            # train G
            z = torch.randn(mini_batch_size, 100).to(device)
            fake_image = G(z)
            
            score_F, score_FS, score_S = D(fake_image, realseis, epoch, batch_idx, args.project_path+args.out_folder+'/eval_train/')             
            
            score_F= score_F+score_FS.mean().item()/2

            score_F[score_F>1]=1

            loss_FS = loss(score_FS, real)
            loss_F = loss(score_F, real)
            
            loss_G =  loss_FS+loss_F
            
            if args.alpha_c:
                cont_loss = CLoss(fake_image)*args.alpha_c
                loss_G+= cont_loss
                cont= cont_loss.item()


            else: cont = None

            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            Losses_Batch.loc[batch_idx]= np.array([loss_D.item(),
                                                   loss_D_FS.item(),
                                                   loss_D_F.item(),
                                                   loss_D_S.item(),
                                                   loss_G.item(),
                                                   loss_F.item(),
                                                   loss_FS.item(),
                                                   cont
            ])
            
            Scores_Batch.loc[batch_idx]= np.array([score_FS_TI.mean().item(), 
                                                   score_F_TI.mean().item(),
                                                   score_S_TI.mean().item(),
                                                   score_F.mean().item(),
                                                   score_FS.mean().item(),
            ])
            
        Glr_scheduler.step()
        Dlr_scheduler.step()

        #check distance with real seismic with MSE
        fake_image= G(z).detach().cpu()
        most_fak= (fake_image.detach().cpu().numpy()+1)*0.5
        most_fak= torch.Tensor(stats.mode(most_fak, axis=0,keepdims=True)[0])
        simulations= Ip_models.run_dss(most_fak, typ, nsim=args.nsim)
        syn_f= FM.calc_synthetic(simulations).detach().cpu()/args.max_abs_amp
        mse_seis= torch.nn.MSELoss()(syn_f.squeeze(),FM.real_seismic[0,0].detach().cpu())

        Losses_Epoch.loc[epoch]= Losses_Batch.mean()
        Scores_Epoch.loc[epoch]= Scores_Batch.mean()

        Scores_Epoch['MSE_seis']= mse_seis.item()
        
        Losses_Epoch.plot()
        
        plt.savefig(args.project_path+args.out_folder+'/sample/losses')
        plt.close('all')
        end = time.time()   
        tottime += end - start

        print('\nEpoch=%d/%d - Time=%.4f' % (epoch, args.epochs, end - start)+'\n'+Losses_Epoch.loc[[epoch]].to_string()+'\n'+Scores_Epoch.loc[[epoch]].to_string())
        img= (fake_image.detach().cpu()+1)*0.5
        if epoch%10==0:
            path = args.project_path + args.out_folder\
                    + '/imgs/facies/Generated_image_in_epoch{}.png'.format(epoch)
            torchvision.utils.save_image(img[0], path, nrow=1, normalize=True)
            plt.close()

        pplot(args.project_path + args.out_folder+'/imgs/sand_prob_epoch_{}.png'.format(epoch), 
                    torch.mean(img, axis=0).detach().cpu().squeeze(),
                    typ='imshow', label= 'fac',  cmap='hot',vmin=0,vmax=1)

        if epoch%50==0:    
            torch.save(G.state_dict(), args.project_path + args.out_folder\
                        +'/save/GAN_generator_in_epoch{}.pth'.format(epoch))
            torch.save(D.state_dict(), args.project_path + args.out_folder\
                        +'/save/GAN_discriminatorFS_in_epoch{}.pth'.format(epoch))
        del fake_image
        plt.close()
    torch.save(G.state_dict(), args.project_path + args.out_folder\
                +'/save/GAN_generator_in_epoch{}.pth'.format(epoch))
    torch.save(D.state_dict(), args.project_path + args.out_folder\
                +'/save/GAN_discriminatorFS_in_epoch{}.pth'.format(epoch))

    print('Average time per epoch is [%.4f]s' % (tottime / args.epochs))  
    Scores_Epoch.plot()
    plt.savefig(args.project_path+args.out_folder+'/sample/scores')
    Losses_Epoch.plot()
    plt.savefig(args.project_path+args.out_folder+'/sample/losses')
    Scores_Epoch.to_csv(args.project_path+args.out_folder+'/sample/scores')
    Losses_Epoch.to_csv(args.project_path+args.out_folder+'/sample/losses')
            
if __name__=='__main__':
 
    for i in [4]:
        
        out_folder='Out_syn_model1'

        parser = argparse.ArgumentParser(description="cDGAN for PhysicsGuided Seismic inversion to Facies")

        parser.add_argument("--project_path", default='D:/', 
                            type=str, help="working directory"
                                        )
        parser.add_argument("--in_folder", default='Input_synthetic_case/', 
                            type=str, help="input folder"
                                        )
        parser.add_argument("--TI_path", default='dataset_syn/Facies_1/', 
                            type=str, help="Training Images folder"
                                        )
        parser.add_argument("--well_data", default='well_data', 
                            type=str, help="Well data filename"
                                        )
        parser.add_argument("--real_seismic_file", default='/Real_seismic_DSS1.out', 
                            type=str, help="seismic file name"
                                        )
        parser.add_argument("--wavelet_file", default='wavelet_simm.asc', 
                            type=str, help="Name of the W	avelet file"
                                        )
        parser.add_argument("--W_weight", default=1/300, 
                            type=float, help="Factor of scaling for wavelet (negative for normal polarity (python indexing))"
                                        )
        parser.add_argument("--batch_size", default=128, 
                            type=int, help="Batch size, batch subset fraction (for big datasets)"
                                        )
        parser.add_argument("--epochs", default=501, 
                            type=int, help="N of Epochs"
                                        )
        parser.add_argument("--lr",default=1e-3,
                            type=float, help="learning rate for the Adam optimizer"
                                        )
        parser.add_argument("--decayRate",default=0.5,
                            type=float, help="learning rate decay"
                                        )
        parser.add_argument("--lr_steps",default=20,
                            type=float, help="learning rate decay"
                                        )
        parser.add_argument("--alpha_c", default=0, 
                            type=float, help='Content loss weight (None if there is no well)'
                                        )
        parser.add_argument("--precompute_TI_seismic", default=True,
                            type=bool, help='if True, computes nsim Seismic realizations from TI facies, to use as training data. \
                            If False, it computes seismic at each epoch from a new DSS realization'
                                        )
        parser.add_argument("--precomputed", default=False,
                            type=bool, help='if True, Seismic_TI and Facies_TI folders contain already the Training Data'
                                        )
        parser.add_argument("--nsim",default=16,
                            type=int, help='If alpha_s is not None, calculates seismic from average of nsim simulations for seismic misfit'
                                        )
        parser.add_argument("--nx", default=100, 
                            type=int, help='x size of inversion grid'
                                        )
        parser.add_argument("--ny", default=1, 
                            type=int, help='y size of inversion grid'
                                        )
        parser.add_argument("--nz", default=80, 
                            type=int, help='z size of inversion grid'
                                        )
        parser.add_argument("--n_facies", default=2, 
                            type=int, help='Number of facies in the model'
                                        )
        parser.add_argument("--var_N_str", default=[1,1], 
                            type=int, help='number of variogram structures per facies [fac0, fac1,...]'
                                        )
        parser.add_argument("--var_nugget", default=[0,0], 
                            type=float, help='variogram nugget per facies [fac0, fac1,...]'
                                        )
        parser.add_argument("--var_type", default=[[1],[1]], 
                            type=int, help='variogram type per facies [fac0[str1,str2,...], fac1[str1,str2,...],...]: 1=spherical,2=exponential,3=gaussian'
                                        )
        parser.add_argument("--var_ang", default=[[0,0],[0,0]],
                            type=float, help='variogram angles per facies [fac0[angX,angZ], fac1[angX,angZ],...]'
                                        )
        parser.add_argument("--var_range", default=[[[80,30]],[[80,50]]],
                            type=float, help='variogram ranges per structure and per facies [fac0[str0[rangeX,rangeZ],str1[rangeX,rangeZ]...],fac1[str1[rangeX,rangeZ],...],...]'
                                        )
        parser.add_argument("--null_val", default=-9999.00, 
                            type=float, help='null value in well data'
                                        )
        parser.add_argument("--ip_type", default=1
                                        )

  
        if not os.path.isdir(parser.parse_args().project_path+out_folder):
            os.mkdir(parser.parse_args().project_path+out_folder)   
            os.mkdir(parser.parse_args().project_path+out_folder+'/dss')
            os.mkdir(parser.parse_args().project_path+out_folder+'/imgs')
            os.mkdir(parser.parse_args().project_path+out_folder+'/imgs/facies')
            os.mkdir(parser.parse_args().project_path+out_folder+'/sample')
            os.mkdir(parser.parse_args().project_path+out_folder+'/eval_train')
            os.mkdir(parser.parse_args().project_path+out_folder+'/save')
  
  
        parser.add_argument("--out_folder", default=out_folder, 
                            type=str,
                            help="output folder")
  
        args = parser.parse_args()
  
        with open(args.project_path+args.out_folder+'/run_commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        main(args)


 
  
