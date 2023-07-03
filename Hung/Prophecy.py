import torch
import torch.nn as nn
class Prophecy:
    def __init__(self,net,data,layer):
        self.net=net
        self.data=data
        self.layer=layer
