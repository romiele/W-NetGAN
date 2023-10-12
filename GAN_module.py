
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.Gnet = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4,stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 256, kernel_size=5,stride=(2,3), padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=5,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2, padding=(0,1), bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4,stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.reshape(-1, 100, 1, 1)
        
        return self.Gnet(z)
		    

class DiscriminatorWNet(nn.Module):
    def __init__(self):
        super(DiscriminatorWNet,self).__init__()
        def block(in_channel, out_channel, ks=5, p=1, s=2, normalize=True,relu=True, sig=False, b=False, conv=True):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=s, padding=p, bias=b))
            
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
            if sig:
                layers.append(nn.Sigmoid())
            
            return layers

        self.DnetJ0 = nn.Sequential(*block(2, 64, 4, normalize=False))
        self.DnetS0 = nn.Sequential(*block(1, 64, 4, normalize=False))
        self.DnetF0 = nn.Sequential(*block(1, 64, 4, normalize=False))
        
        self.DnetJ1 = nn.Sequential(*block(64*3, 128,4,p=(0,1)))
        self.DnetS1 = nn.Sequential(*block(64, 128, 4,p=(0,1)))
        self.DnetF1 = nn.Sequential(*block(64, 128, 4,p=(0,1)))
       
        self.DnetJ2 = nn.Sequential(*block(128*3, 256))
        self.DnetS2 = nn.Sequential(*block(128, 256))
        self.DnetF2 = nn.Sequential(*block(128, 256))

        self.DnetJ3 = nn.Sequential(*block(256*3, 1, s=(2,3) ,normalize=False, relu=False, sig=True))
        self.DnetF3 = nn.Sequential(*block(256, 1, s=(2,3) ,normalize=False, relu=False, sig=True))
        self.DnetS3 = nn.Sequential(*block(256, 1, s=(2,3) ,normalize=False, relu=False, sig=True))

    def forward(self, facies,seismic, epoch=1, batch_idx=0, folder=''):

        j0= torch.cat([facies, seismic], 1)
        j1 = self.DnetJ0(j0)
        f1 = self.DnetF0(facies)
        s1 = self.DnetS0(seismic)
        
        j1= torch.cat([j1,f1,s1], 1)
        j2 = self.DnetJ1(j1)
        f2 = self.DnetF1(f1)
        s2 = self.DnetS1(s1)        
        
        j2= torch.cat([j2,f2,s2], 1)
        j3= self.DnetJ2(j2)
        f3= self.DnetF2(f2)
        s3= self.DnetS2(s2)
        
        j3= torch.cat([j3,f3,s3], 1)
        scoreJ= self.DnetJ3(j3)
        scoreF= self.DnetF3(f3)
        scoreS= self.DnetS3(s3)
        
        Nj=scoreJ.shape[-1]*scoreJ.shape[-2]
        Nf=scoreF.shape[-1]*scoreF.shape[-2]
        Ns=scoreS.shape[-1]*scoreF.shape[-2]
        #print (N)
        #print (scoreF.shape)
        
        #.reshape(-1,N).mean(dim=-1)
        return scoreF.reshape(-1,Nf).mean(dim=-1).view(-1,1), scoreJ.reshape(-1,Nj).mean(dim=-1).view(-1,1), scoreS.reshape(-1,Ns).mean(dim=-1).view(-1,1)



class Generator_norne(nn.Module):
    def __init__(self):
        super(Generator_norne, self).__init__()

        self.Gnet = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=5, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
        
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=(1,0), bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=(1,0), bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
        
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=(2,0), bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=(2,1), bias=False),
            nn.Tanh(),
        )

        return None 
    def forward(self, z):
        z = z.reshape(-1, 100, 1, 1)
        return self.Gnet(z)
    	
class DiscriminatorWNet_norne(nn.Module):
    def __init__(self):
        super(DiscriminatorWNet_norne,self).__init__()
        def block(in_channel, out_channel, ks=3, p=1, s=2, normalize=True,relu=True, sig=False, b=False, conv=True):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=s, padding=p, bias=b))
            
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sig:
                layers.append(nn.Sigmoid())
            
            return layers

        self.DnetJ0 = nn.Sequential(*block(2, 64, normalize=False))
        self.DnetS0 = nn.Sequential(*block(1, 64, normalize=False))
        self.DnetF0 = nn.Sequential(*block(1, 64, normalize=False))
        
        self.DnetJ1 = nn.Sequential(*block(64*3, 128))
        self.DnetS1 = nn.Sequential(*block(64, 128))
        self.DnetF1 = nn.Sequential(*block(64, 128))
       
        self.DnetJ2 = nn.Sequential(*block(128*3, 256))
        self.DnetS2 = nn.Sequential(*block(128, 256))
        self.DnetF2 = nn.Sequential(*block(128, 256))

        self.DnetJ3 = nn.Sequential(*block(256*3, 1 ,ks= 5,normalize=False, relu=False, sig=True))
        self.DnetF3 = nn.Sequential(*block(256, 1 ,ks= 5, normalize=False, relu=False, sig=True))
        self.DnetS3 = nn.Sequential(*block(256, 1 ,ks= 5, normalize=False, relu=False, sig=True))

    def forward(self, facies,seismic, epoch=1, batch_idx=0, folder=''):

        j0= torch.cat([facies, seismic], 1)
        j1 = self.DnetJ0(j0)
        f1 = self.DnetF0(facies)
        s1 = self.DnetS0(seismic)
        
        j1= torch.cat([j1,f1,s1], 1)
        j2 = self.DnetJ1(j1)
        f2 = self.DnetF1(f1)
        s2 = self.DnetS1(s1)        
        
        j2= torch.cat([j2,f2,s2], 1)
        j3= self.DnetJ2(j2)
        f3= self.DnetF2(f2)
        s3= self.DnetS2(s2)
        
        j3= torch.cat([j3,f3,s3], 1)
        scoreJ= self.DnetJ3(j3)
        scoreF= self.DnetF3(f3)
        scoreS= self.DnetS3(s3)

        Nj=scoreJ.shape[-1]*scoreJ.shape[-2]
        Nf=scoreF.shape[-1]*scoreF.shape[-2]
        Ns=scoreS.shape[-1]*scoreF.shape[-2]

        return scoreF.reshape(-1,Nf).mean(dim=-1).view(-1,1), scoreJ.reshape(-1,Nj).mean(dim=-1).view(-1,1), scoreS.reshape(-1,Ns).mean(dim=-1).view(-1,1)
    
    