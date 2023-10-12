# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:35:50 2023
    create synthetic data for WNet GAN
@author: Roberto.Miele
"""
from Gslib import Gslib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import sys

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomInvert(1),
    torchvision.transforms.RandomRotation([180,180])
    ])

dataset= torchvision.datasets.ImageFolder(root='D:/dataset_syn/Facies_1/', transform=transforms)

for jj in range(10):
    real_facies= np.flip(dataset[jj][0][0].numpy(), axis=1)
    print(real_facies.max(), real_facies.min())
    print(real_facies.max(), real_facies.min())
    print(real_facies.max(), real_facies.min())
    inf= 'D:/Input_synthetic_case/'
    Gslib().Gslib_write(f'Real_Facies_{jj}', 'facies', real_facies.flatten(), 80,1,100, inf)

    fig, axs= plt.subplots(1,1)
    ax= axs.imshow(real_facies, cmap='gray')
    cax= fig.add_axes([0.9,0.25,0.05,0.5])
    cbar= fig.colorbar(ax, cax=cax, orientation='vertical',ticks=[0, 1])
    cbar.ax.set_yticklabels(['Shale', 'Sand'], rotation = 90)
    axs.set_xlabel('CMP')
    axs.set_ylabel('TWT [ms]')
    plt.savefig(inf+f'/1.Real_facies{jj}.png', bbox_inches='tight', dpi=300)
    plt.close()
    Ip_shales= Gslib().Gslib_read('D:/Input_synthetic_case/Ip_zone0_unc.out').data.Ip.values
    Ip_sands= Gslib().Gslib_read('D:/Input_synthetic_case/Ip_zone1_unc.out').data.Ip.values
    
    fig, axs= plt.subplots(1,1,dpi=300)
    ax= axs.hist(Ip_shales, color='black', label='Shales', alpha=0.7, density=True, weights=np.ones(100)/100)
    ax2= axs.hist(Ip_sands, color='orange', label='Sands', alpha=0.7, density=True, weights=np.ones(100)/100)
    axs.legend(loc='upper right')
    axs.set_xlabel(r'$I_P$')
    axs.set_ylabel('Density')
    plt.savefig(inf+'2.Hist_Ip.png', bbox_inches='tight')
    plt.close()

    
    grid= np.random.randint(low=0, high=100,size=(100,5)).astype(float)+10000
    shales_grid= grid.copy()
    shales_grid[:,3]=Ip_shales
    shales_grid[:,4]=np.zeros(len(shales_grid[:,4]))


    sands_grid= grid.copy()
    sands_grid[:,3]=Ip_sands
    sands_grid[:,4]=np.ones(len(sands_grid[:,4]))

    with open (inf+'Ip_zone0_unc.out','w') as file:
        file.write('Ip_zone0_unc\n')
        file.write('4\nx\ny\nz\nIp\n')
        for i in range(100):
            file.write(' '.join(list(shales_grid[i,0:4].astype(str)))+'\n')


    with open (inf+'Ip_zone1_unc.out','w') as file:
        file.write('Ip_zone1_unc\n')
        file.write('4\nx\ny\nz\nIp\n')
        for i in range(100):
            file.write(' '.join(list(sands_grid[i,0:4].astype(str)))+'\n')


    grid=np.append(shales_grid,sands_grid, axis=0)

    with open (inf+'/well_data','w') as file:
        file.write('Well data\n')
        file.write('5\ni_index\nj_index\nk_index\nIp\nFacies\n')
        for i in range(200):
            file.write(' '.join(list(grid[i].astype(str)))+'\n')

    ipzones= np.empty((2,2))
    ipzones[0]=[np.min(Ip_shales),np.max(Ip_shales)]
    ipzones[1]=[np.min(Ip_sands),np.max(Ip_sands)]

    real_Ip_det= (ipzones.max()-ipzones.min())*(real_facies*-1+1)+ipzones.min()
    fig, axs= plt.subplots(1,1,dpi=300)
    ax= axs.imshow(real_Ip_det.squeeze(), cmap='jet')
    cax= fig.add_axes([0.9,0.25,0.05,0.5])
    cbar= fig.colorbar(ax, cax=cax, orientation='vertical', ticks=[real_Ip_det.min(),6000,6500,7000,7500,8000,8500,9000,real_Ip_det.max()])
    cbar.ax.set_title(r'$I_P$')
    axs.set_xlabel('CMP')
    axs.set_ylabel('TWT [ms]')
    plt.savefig(inf+f'/3.real_Ip_det{jj}.png', bbox_inches='tight')
    plt.close()

    text=[]
    text.append(f'[ZONES]\nZONESFILE = {inf}/Real_Facies_{jj}.out  # File with zones\nNZONES=2  # Number of zones\n\n')
    for fac in range(2):
        text.append(f'[HARDDATA{fac+1}]\nDATAFILE = {inf}/Ip_zone{fac}_unc.out  # Hard Data file\n')
        text.append('COLUMNS = 4\nXCOLUMN = 1\nYCOLUMN = 2\nZCOLUMN = 3\nVARCOLUMN = 4\nWTCOLUMN = 0\n')
        text.append(f'MINVAL = {ipzones[fac][0]}  # Minimun threshold value\nMAXVAL = {ipzones[fac][1]}  # Minimun threshold value\n')
        text.append('USETRANS = 1\nTRANSFILE = Cluster.trn  #Transformation file\n\n')
    text.append(f'[HARDDATA]\nZMIN = {ipzones.min()}  # Minimum allowable data value\nZMAX = {ipzones.max()} \
                # Maximum allowable data value\nLTAIL = 1\nLTPAR = {ipzones.min()}\nUTAIL = 1\nUTPAR = {ipzones.max()}\n\n')
    text.append(f'[SIMULATION]\nOUTFILE = {inf}/Real_Ip  # Filename of the resultant simulations\nNSIMS = 1  # Number of Simulations to generate \nNTRY = 10\nAVGCORR = 1\nVARCORR = 1\n\n')
    text.append('[GRID]\nNX = 100\nNY = 1\nNZ = 80\nORIGX = 1\nORIGY = 1\nORIGZ = 1\nSIZEX = 1\nSIZEY = 1\nSIZEZ = 1\n\n')
    text.append('[GENERAL]\nNULLVAL = -9999.99\nSEED = 6906\nUSEHEADERS = 1\nFILETYPE = GEOEAS\n\n')
    text.append('[SEARCH]\nNDMIN = 1\nNDMAX = 32\nNODMAX = 12\nSSTRAT = 1\nMULTS = 0\nNMULTS = 1\nNOCT = 0\nRADIUS1 = 100\nRADIUS2 = 1\nRADIUS3 = 80\nSANG1 = 0\nSANG2 = 0\nSANG3 = 0\n\n')
    text.append('[KRIGING]\nKTYPE = 0  # Kriging type: 0=simple,1=ordinary,2=simple with locally varying mean, 3=external drif, 4=collo-cokrig global CC,5=local CC (KTYPE)\n')
    text.append('COLOCORR = 0.75\nSOFTFILE = No File\nLVMFILE = No File\nNVARIL = 1\nICOLLVM = 1\nCCFILE = No File\nRESCALE = 1\n\n')


    var_N_str=np.array([1,1])
    var_nugget=np.array([0,0])
    var_type=np.array([[1],[1]])
    var_ang= np.array([[0,0],[0,0]])
    var_range= np.array([[[80,30]],[[80,50]]])

    for fac in range(2):
        text.append(f'[VARIOGRAMZ{fac+1}]\nNSTRUCT = {var_N_str[fac]}  # Number of semivariograms structures\nNUGGET = {var_nugget[fac]}  # Nugget constant\n\n')
        for struct in range(var_N_str[fac]):
            text.append(f'[VARIOGRAMZ{fac+1}S{struct+1}]\nTYPE = {var_type[fac][struct]}\nCOV = 1\n')
            text.append(f'ANG1 = {var_ang[fac][0]}\nANG2 = 0\nANG3 = {var_ang[fac][1]}\n')
            text.append(f'AA = {var_range[fac][struct][0]}\nAA1 = 1\nAA2 = {var_range[fac][struct][1]}\n\n')
        text.append(f'[BIHIST{fac+1}]\nUSEBIHIST = 0\nBIHISTFILE = No File\nNCLASSES = 20\nAUXILIARYFILE = No File\n\n')
    text.append('[DEBUG]\nDBGLEVEL = 1\nDBGFILE = debug.dbg\n\n')
    text.append('[COVTAB]\nMAXCTX = 100\nMAXCTY = 1\nMAXCTZ = 80\n\n')
    text.append('[BLOCKS]\nUSEBLOCKS = 0\nBLOCKSFILE = NoFile\nMAXBLOCKS= 100\n\n[PSEUDOHARD]\nUSEPSEUDO = 0\nBLOCKSFILE = No File\nPSEUDOCORR = 0\n')
        
    text= ''.join(text)

    with open(inf+'/ssdir.par', 'w') as ssdir:
        ssdir.write(text)

    import subprocess
    subprocess.run(args=[inf+'/DSS.C.64.exe', inf+'/ssdir.par'])

    real_Ip= Gslib().Gslib_read(inf+f'/Real_Ip_1.out').data.values
    real_Ip= np.reshape(real_Ip, (80,100))

    from Fmodel_lib import fmodel_simple
    wavelet= (np.genfromtxt(inf+'/wavelet_simm.asc')/300)

    plt.figure(dpi=300)
    plt.plot(wavelet)
    plt.xlabel('ms')
    plt.ylabel('amplitude')
    plt.savefig(inf+'/4.wavelet.png', bbox_inches='tight')
    plt.close()
    real_seismic_det= fmodel_simple(real_Ip_det.T, wavelet*-1).squeeze().T
    real_seismic= fmodel_simple(real_Ip.T, wavelet*-1).squeeze().T

    fig, axs= plt.subplots(1,1,dpi=300)
    ax= axs.imshow(real_Ip, cmap='jet')
    cax= fig.add_axes([0.9,0.25,0.05,0.5])
    cbar= fig.colorbar(ax, cax=cax, orientation='vertical', ticks=[real_Ip.min(),6000,6500,7000,7500,8000,8500,9000,real_Ip.max()])
    cbar.ax.set_title(r'$I_P$')
    axs.set_xlabel('CMP')
    axs.set_ylabel('TWT [ms]')
    plt.savefig(inf+f'/3.Real_Ip{jj}.png', bbox_inches='tight')
    plt.close()

    fig, axs= plt.subplots(1,1,dpi=300)
    ax= axs.imshow(real_seismic, cmap='seismic')
    cax= fig.add_axes([0.9,0.25,0.05,0.5])
    cbar= fig.colorbar(ax, cax=cax, orientation='vertical')
    cbar.ax.set_title('Amplitude')
    axs.set_xlabel('CMP')
    axs.set_ylabel('TWT [ms]')
    plt.savefig(inf+f'/5.Real_Seismic{jj}.png', bbox_inches='tight')
    plt.close()

    fig, axs= plt.subplots(1,1,dpi=300)
    ax= axs.imshow(real_seismic_det, cmap='seismic')
    cax= fig.add_axes([0.9,0.25,0.05,0.5])
    cbar= fig.colorbar(ax, cax=cax, orientation='vertical')
    cbar.ax.set_title('Amplitude')
    axs.set_xlabel('CMP')
    axs.set_ylabel('TWT [ms]')
    plt.savefig(inf+f'/5.Real_Seismic_det{jj}.png', bbox_inches='tight')
    plt.close()

    with open (inf+f'/Real_seismic_DSS{jj}.out','w') as file:
        file.write('Real_seismic_DSS\n')
        file.write('1\nAmplitude\n')
        file.write('\n'.join(list(real_seismic.flatten().astype(str))))

    with open (inf+f'/Real_seismic_det{jj}.out','w') as file:
        file.write('Real_seismic_DSS\n')
        file.write('1\nAmplitude\n')
        file.write('\n'.join(list(real_seismic_det.flatten().astype(str))))

