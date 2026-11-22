# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:23:17 2022

@author: yimen
"""


import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as  F


def Getini_3D_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return torch.FloatTensor(A)#(outNum,inNum,expand,nNum)

class Fconv_3D_PCA(nn.Module):
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None,stride=1, ifIni=0, bias=True, Smooth = True, iniScale = 1.0):
       
        super(Fconv_3D_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.kernel_size = (sizeP, sizeP, sizeP)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.ifbias = bias
        Basis, Rank, weight = GetBasis_3D_PCA(sizeP,tranNum,inP,Smooth = Smooth)        
        self.register_buffer("Basis", Basis)
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, self.expand, Basis.size(4)), requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        
        if bias:
            self.c = nn.Parameter(torch.Tensor(1,outNum,1,1,1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()


    def forward(self, input):
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijosk,mnak->msnaoij', self.Basis, self.weights)
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
                   
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP, self.sizeP ])
            self.decay_weights = _filter
            if self.ifbias:
                _bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])
                self.register_buffer("bias", _bias)
        else:
            _filter = self.filter
            if self.ifbias:
                _bias   = self.bias

        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        stride=self.stride,
                        dilation=1,
                        groups=1)
        if self.ifbias:
            output = output + _bias
        return output
    
    def train(self, mode=True):
        if mode:
            # Clear evaluation caches before using trainable weights again.
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
            if hasattr(self, "decay_weights"):
                del self.decay_weights  # This is for weight decay
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijosk,mnak->msnaoij', self.Basis, self.weights)
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ,self.sizeP ])
            self.register_buffer("filter", _filter)

            if self.ifbias:
                _bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])
                self.register_buffer("bias", _bias)

        return super(Fconv_3D_PCA, self).train(mode)
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class F_BN_3D(nn.Module):
    def __init__(self,channels, tranNum=8):
        super(F_BN_3D, self).__init__()
        self.BN = nn.BatchNorm3d(channels)
        self.tranNum = tranNum
    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3), X.size(4)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3), X.size(4)])


def MaskC_3D(SizeP,tranNum=4):


    p = (SizeP-1)/2
    x = np.arange(-p,p+1)/p
    X,Y,Z  = np.meshgrid(x,x,x)
    C    =X**2+Y**2
    if tranNum == 4:
        Mask = np.ones([SizeP,SizeP,SizeP])
    
    else:
        Mask = np.exp(-np.maximum(C-1,0)/0.2)
        
    # X:(P,P,P) Y:(P,P,P) Z:(P,P,P) Mask:(P,P,P)
    
    return X, Y, Z, Mask


def GetBasis_3D_PCA(sizeP, tranNum=8, inP=None, Smooth = True):
    if inP==None:
        inP = sizeP
    inX, inY,inZ, Mask = MaskC_3D(sizeP)#(P,P,P)
    X0 = np.expand_dims(inX,3)#(P,P,P,1)
    Y0 = np.expand_dims(inY,3)#(P,P,P,1)
    Z0 = np.expand_dims(inZ,3)#(P,P,P,1)
    Mask = np.expand_dims(Mask,3)#(P,P,P,1)
    # Rotate spatial coordinates only; leave the spectral coordinate unchanged.
    theta = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    theta = np.expand_dims(np.expand_dims(np.expand_dims(theta,axis=0),axis=0),axis=0)#(1,1,1,tranNum)

    X = np.cos(theta)*X0+np.sin(theta)*Y0
    Y = -np.sin(theta)*X0+np.cos(theta)*Y0
    Z = Z0
    
    X = np.expand_dims(np.expand_dims(np.expand_dims(X,4),5),6)#(P,P,P,tranNm,1,1,1)
    Y = np.expand_dims(np.expand_dims(np.expand_dims(Y,4),5),6)#(P,P,P,tranNm,1,1,1)
    Z = np.expand_dims(np.expand_dims(np.expand_dims(Z,4),5),6)#(P,P,P,tranNm,1,1,1)
    
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    k = np.reshape(np.arange(inP),[1,1,1,1,inP,1,1])
    l = np.reshape(np.arange(inP),[1,1,1,1,1,inP,1])
    z = np.reshape(np.arange(inP),[1,1,1,1,1,1,inP])

    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y+(z-inP*(z>p))*v*Z)#(P,P,P,tranNum,P,P,P)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y+(z-inP*(z>p))*v*Z)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, sizeP, tranNum, inP*inP*inP])*np.expand_dims(Mask,4)#(P,P,P,tranNum,P*P*P)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, sizeP, tranNum, inP*inP*inP])*np.expand_dims(Mask,4)
    # Reduce the sampled Fourier basis with PCA.
    BasisC = np.reshape(BasisC,[sizeP*sizeP*sizeP*tranNum, inP*inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*sizeP*tranNum, inP*inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, sizeP, tranNum, Rank])
    
    temp = np.reshape(BasisR, [sizeP*sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP*sizeP, axis = 0),axis = 0)
    Trod = 1
    Ind = var<Trod
    Rank = np.sum(Ind)
    Weight = 1/np.maximum(var, 0.04)/25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0),0)*BasisR #(P,P,P,tranNum,P*P*P)

    return torch.FloatTensor(BasisR), Rank, Weight


class Fconv_3D_PCA_out(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True,iniScale = 1.0):
       
        super(Fconv_3D_PCA_out, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_3D_PCA(sizeP,tranNum,inP,Smooth = Smooth)        
        self.register_buffer("Basis", Basis)
        self.ifbias = bias
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1, Basis.size(4)), requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1,outNum,1,1,1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):
    
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijosk,mnak->mansoij', self.Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP, self.sizeP ])
            self.decay_weights = _filter 
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output + _bias
    def train(self, mode=True):
        if mode:
            # Clear evaluation caches before using trainable weights again.
            if hasattr(self, "filter"):
                del self.filter
            if hasattr(self, "decay_weights"):
                del self.decay_weights  # This is for weight decay
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijosk,mnak->mansoij', self.Basis, self.weights)
            
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP, self.sizeP ])
            self.register_buffer("filter", _filter)
        return super(Fconv_3D_PCA_out, self).train(mode)
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class Fconv_1X1X1(nn.Module):
    
    def __init__(self, inNum, outNum, tranNum=8, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, zeros=False):
       
        super(Fconv_1X1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

        self.ifbias = bias
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, self.expand, 1), requires_grad=True)

        self.padding = 0

        if bias:
            self.c = nn.Parameter(torch.Tensor(1,outNum,1,1,1))
        else:
            self.register_parameter('c', None)
        if not zeros:
            self.reset_parameters()
        else:
            nn.init.zeros_(self.weights)
            if self.c is not None:
                nn.init.zeros_(self.c)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(5).unsqueeze(1).repeat([1,tranNum,1,1,1,1,1])
        
        Num = tranNum//expand
        tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,...],tempW[:,i*Num:(i+1)*Num,:,:-i,...]], dim = 3) for i in range(expand)]   
        tempW = torch.cat(tempWList, dim = 1)

        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, 1, 1, 1 ])
        
        if self.ifbias:
            _bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])
        else:
            _bias = 0
        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output+_bias
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)
