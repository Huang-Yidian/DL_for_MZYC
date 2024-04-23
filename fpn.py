from typing import *
import torch
import torch.nn.functional as F

from torch import nn


class FPN(nn.Module):
    def __init__(self, inch: List[int], hidden_size, outch):
        super(FPN, self).__init__()        
        self.inch = inch
        self.outch = outch
        
        self.proj5 = nn.Conv2d(inch[3],hidden_size,1)
        self.proj4 = nn.Conv2d(inch[2],hidden_size,1)
        self.proj3 = nn.Conv2d(inch[1],hidden_size,1)
        self.proj2 = nn.Conv2d(inch[0],hidden_size,1)
        
        self.smooth4 = nn.Conv2d(hidden_size,outch,3,1,1)
        self.smooth3 = nn.Conv2d(hidden_size,outch,3,1,1)
        self.smooth2 = nn.Conv2d(hidden_size,outch,3,1,1)
    
    def forward(self, x:List[torch.Tensor]): 
        C2,C3,C4,C5 = x
        M5 = P5 = self.proj5(C5)
        M4 = self.proj4(C4) + F.interpolate(M5, size=[C4.size(-2),C4.size(-1)],mode="bilinear")
        P4 = self.smooth4(M4)
        M3 = self.proj3(C3) + F.interpolate(M4, size=[C3.size(-2),C3.size(-1)],mode="bilinear")
        P3 = self.smooth3(M3)
        M2 = self.proj2(C2) + F.interpolate(M3, size=[C2.size(-2),C2.size(-1)],mode="bilinear")
        P2 = self.smooth2(M2)
        return [M2,M3,M4,M5], [P2,P3,P4,P5]