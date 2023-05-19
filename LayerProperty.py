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
        out_size=state_dict[keys[-2]].shape[0]
        
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
            pattern.append(output[i])
            patternF=pattern.copy()
            patternT=pattern.copy()
            patternT.append(True)
            patternF.append(False)

            out_property=LayerProperty.property_converter(output[i],out_size)
            if patternT in hidden:
                hidden.append(patternT)
                print("hit")
            elif patternF in hidden:
                hidden.append(patternF)
                print("hit")
            else:
                verifier=NetVerifier()

                isverified=verifier.isverified(self.net,pattern,out_property)
                pattern.append(isverified)
                hidden.append(pattern)

        self.layer_patterns=hidden
    @staticmethod
    def property_converter(tgt,size):
        #maxtrix [size-1,size]*y<=0
        res=[]
        for i in range(size):
            if i ==tgt:
                continue
            cur=[]
            for j in range(size):
                if j==tgt:
                    cur.append(-1)
                elif j==i:
                    cur.append(1)
                else:
                    cur.append(0)
            res.append(cur)
        return res



