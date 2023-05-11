import torch
import torch.nn.functional as F
import numpy as np
from NetVerifier import NetVerifier
class LayerProperty():
    def __init__(self,net,data):
        self.net=net
        self.data=data
    def get_pattern(self):
        input, output = self.data
        state_dict=self.net.state_dict()
        keys=list(state_dict.keys())
        input=torch.tensor(input,dtype=torch.float32)

        
        hidden= []

        for i in range(len(keys)//2):
            state_dict[keys[2 * i + 1]] = torch.reshape(state_dict[keys[2 * i + 1]],
                                                        (1, state_dict[keys[2 * i + 1]].shape[0]))


        for i in range(len(input)):
            out = input[i]
            pattern=[]
            for j in range(len(keys)//2):
                out = torch.matmul(out, torch.transpose(state_dict[keys[2 * j]],1,0)) + state_dict[keys[2 * j + 1]]
                out=F.relu(out)
                pattern.append(F.relu(out)>0)
            patternT=pattern.append(True)
            patternF=pattern.append(False)
            if patternT in hidden:
                hidden.append(patternT)
            elif patternF in hidden:
                hidden.append(patternF)
            else:
                verifier=NetVerifier()

                isverified=verifier.isverified(state_dict,pattern)
                pattern.append(isverified)
                hidden.append(pattern)

        self.layer_patterns=hidden
    @staticmethod
    def verify_property(pattern_dict):

        pass



