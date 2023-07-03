import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
print(sys.path)
sys.path.append("/home/lqhung2001/Downloads/Marabou")
from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions
class NetVerifier:
    encode_status={
        'on':1,
        'off':-1,
        'skip':0
    }
    large=10e3
    small=10e-3
    def __init__(self):
        pass
    def composeDP(self,net:torch.nn.Module,precondition:list,postcondition:list, pattern:list,inp:torch.Tensor):
        DP=MarabouCore.InputQuery()
        num_nodes=self.getNumNodes(net,inp)
        DP.setNumberOfVariables(num_nodes)
        for i in range(num_nodes):
            DP.setLowerBound(i,-self.large)
            DP.setUpperBound(i,self.large)
        for i in range(len(precondition)):
            DP.setLowerBound(i,precondition[i][0])

            DP.setUpperBound(i,precondition[i][1])
        var={
            "DP":DP,
            "base":0,
            "patternid":0,
            'input':inp
        }
        for module in net.modules():
            if module.__class__.__name__!=net.__class__.__name__:
                var=self.netDP(module, var,pattern)
        DP=self.outDP(postcondition,var)
        option= createOptions(snc=True)
        stats=MarabouCore.solve(DP,option,"")
        if stats[0]=='sat':
            return False
        return True
    def netDP(self,module,var,pattern):
        if type(module)==nn.Linear:
            return self.LinearDP(module,var)
        if type(module)==nn.ReLU:
            return self.ReLUDP(module,var,pattern)

    def LinearDP(self,module:nn.Linear,var):
        params=[]
        for param in module.parameters():
            params.append(param.data.detach().numpy())
        out=module(var['input'])
        for i in range(params[0].shape[0]):
            equation=MarabouCore.Equation()
            equation.addAddend(-1,var['base']+i+params[0].shape[1])
            for j in range(params[0].shape[1]):
                equation.addAddend(params[0][i][j],var['base']+j)
            equation.setScalar(-params[1][i])
            var['DP'].addEquation(equation)
        var['input']=out
        var['base']+=params[0].shape[1]
        return var
    def ReLUDP(self,module,var,pattern):
        flat=module(var['input']).view(var['input'].shape[0],-1)
        for i in range(len(pattern[var['patternid']])):
            a=var['base']+i+len(pattern[var['patternid']])
            MarabouCore.addReluConstraint(var["DP"],var['base']+i,var['base']+i+len(pattern[var['patternid']]))
            #var['DP'].setLowerBound(var['base']+i+len(pattern[var['patternid']]),0)
            if self.encode_status[pattern[var['patternid']][i]]>0:
                var['DP'].setLowerBound(var['base']+i+len(pattern[var['patternid']]),self.small)
            if self.encode_status[pattern[var['patternid']][i]]<0:
                var['DP'].setUpperBound(var['base'] + i + len(pattern[var['patternid']]), 0)
        var['base']+=len(pattern[var['patternid']])
        var['input']=module(var['input'])
        var["patternid"]+=1
        return var

    def outDP(self,postcondition,var):
        disjunction=[]
        for condition in postcondition:
            equationtype=MarabouCore.Equation.EquationType(1)
            equation=MarabouCore.Equation(equationtype)
            for i in range(len(condition)-1):
                equation.addAddend(condition[i],i+var['base'])
            equation.setScalar(-condition[-1])
            disjunction.append([equation])
        MarabouCore.addDisjunctionConstraint(var["DP"],disjunction)
        return var["DP"]
    def getNumNodes(self,net,inp):

        # Define a dictionary to store the layer outputs
        layer_outputs = {}
        # Define a forward hook function
        def forward_hook(module, input, output):
            layer_outputs[module] = output
            # Register the forward hook for each layer
        for name, module in net.named_modules():
            module.register_forward_hook(forward_hook)
        # Perform forward pass
        output_tensor = net(inp)

        flatten=inp.view(-1)
        res=flatten.shape[0]
        # Access the forward output of each layer
        for module, output in layer_outputs.items():
            if module.__class__.__name__ != net.__class__.__name__:
                flatten=output.view(-1)
                res+=flatten.shape[0]
        return res
    @classmethod
    def DP(cls, pattern:list,property:list,net:torch.nn.Module):
        """
        args:
            pattern:
            - len(pattern)=num_of_layer-1
            - len(pattern[i])=num_of_ith_layer_nodes
            - dtype: int (-1=off, 1=on,0=skip_verify)
            property:
            - shape=(size_of_property,size_of_y + 1)
            - dtype= float
            - A[:,:-2]*y+A[:,-1]<=0
            net:
            -pytorch net has been fowarded
        return:
            True/False if net is verified with given pattern
        """
        state_dict = net.state_dict()
        # get num_node
        keys = list(state_dict.keys())
        num_node = 0
        for i,key in enumerate(keys):
            if i%2:
                continue
            else:
                num_node += state_dict[key].shape[0]+state_dict[key].shape[1]
        large = 10e2
        small=10e-2
        # define c
        for id,con_prop in enumerate(property):

            ip_net = MarabouCore.InputQuery()
            ip_net.setNumberOfVariables(num_node)
            # define inf,-inf
            for i in range(num_node):
                ip_net.setLowerBound(i, -large)
                ip_net.setUpperBound(i, large)
            # define relu constrain
            base = state_dict[keys[0]].shape[1]
            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    # a=base+j
                    # b=base+j+len(pattern[i])
                    MarabouCore.addReluConstraint(ip_net, base + j, base + j + len(pattern[i]))
                base += 2 * len(pattern[i])
            # define relu status constrain
            base = state_dict[keys[0]].shape[1]
            for i in range(len(pattern)):
                base += len(pattern[i])
                for j in range(len(pattern[i])):
                    a=base+j
                    ip_net.setLowerBound(base + j, 0)
                    if cls.encode_status[pattern[i][j]] > 0:
                        ip_net.setLowerBound(base+j-len(pattern[i]),small)
                        ip_net.setLowerBound(base + j, small)
                    if cls.encode_status[pattern[i][j]] < 0:
                        ip_net.setUpperBound(base+j-len(pattern[i]),0)
                        ip_net.setUpperBound(base + j, 0)
                base += len(pattern[i])
            # define net equations
            base=state_dict[keys[0]].shape[1]
            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    equation=MarabouCore.Equation()
                    equation.addAddend(-1,base+j)
                    for k in range(state_dict[keys[2*i]].shape[1]):
                        equation.addAddend(state_dict[keys[2*i]][j][k],base-state_dict[keys[2*i]].shape[1]+k)
                    equation.setScalar(-state_dict[keys[2*i+1]][j])
                    ip_net.addEquation(equation)
                base+=2*len(pattern[i])
            #define contradict property

            equationtype = MarabouCore.Equation.EquationType(2)
            out_eq = MarabouCore.Equation(equationtype)
            for i in range(len(con_prop)-1):

                out_eq.addAddend(con_prop[i],base+i)
            out_eq.setScalar(con_prop[-1]+1e-6)
            ip_net.addEquation(out_eq)

            options = createOptions(snc=True)

            stats = MarabouCore.solve(ip_net, options, "")

            for i in range(ip_net.getNumberOfVariables()):
                print(f"{i} upperbound {ip_net.getUpperBound(i)}, lower bound {ip_net.getLowerBound(i)}")
            if stats[0]=="sat":
                return stats,ip_net

        return True,ip_net









        # state_dict=net.state_dict()
        # keys=list(state_dict.keys())
        # for i, output_property in enumerate(output_properties):
        #
        #     cnt=0
        #     large=10e6
        #     small=10e-6
        #     cnt+=state_dict[keys[0]].shape[0]
        #     for i,name in enumerate(keys):
        #         if i%2==0:
        #             cnt+=2*state_dict[name].shape[1]
        #     cnt-=state_dict[keys[-2]].shape[1]
        #     inputQuery=MarabouCore.InputQuery()
        #     inputQuery.setNumberOfVariables(cnt)
        #     cur=0
        #     #set range input
        #     for i,name in enumerate(keys):
        #         if i==0:
        #             for j in range(state_dict[name].shape[0]):
        #                 inputQuery.setLowerBound(cur,-large)
        #                 inputQuery.setUpperBound(cur,large)
        #                 cur+=1
        #         if i%2==0 and i>0:
        #             for j in range(state_dict[name].shape[0]):
        #                 inputQuery.setLowerBound(cur,-large)
        #                 inputQuery.setUpperBound(cur,large)
        #                 cur+=1
        #             for j in range(state_dict[name].shape[0]):
        #                 if pattern[(i//2)-1][j]:
        #                     inputQuery.setLowerBound(cur,small)
        #                     inputQuery.setUpperBound(cur,large)
        #                     cur+=1
        #                 else:
        #                     inputQuery.setLowerBound(cur, 0)
        #                     inputQuery.setUpperBound(cur, 0)
        #                     cur += 1
        #     #set equation
        #     cur=0
        #     for i,name in enumerate(keys):
        #         if i%2==0:
        #             for j in range(state_dict[name].shape[1]):#set node j-th of layer i-th
        #                 equation=MarabouCore.Equation()
        #                 equation.addAddend(-1,cur+state_dict[name].shape[0]+j)
        #                 for k in range(state_dict[name].shape[0]):
        #                     equation.addAddend(state_dict[name][k][j],cur+k)
        #
        #                 equation.setScalar(-state_dict[keys[i+1]][j])
        #                 inputQuery.addEquation(equation)
        #             if i<len(keys)-2:
        #                 for j in range(state_dict[name].shape[1]):
        #                     #set relu constrain of node j-th of layer i-th
        #                     MarabouCore.addReluConstraint(inputQuery, cur+j, cur+j+state_dict[name].shape[1])
        #
        #             cur+=state_dict[name].shape[1]+state_dict[name].shape[0]
        #     cur-=len(output_property)
        #     equationtype=MarabouCore.Equation.EquationType(1)
        #     out_eq=MarabouCore.Equation(equationtype)
        #     for i,w in enumerate(output_property):
        #         out_eq.addAddend(w,i+cur)
        #     out_eq.setScalar(0.001)
        #     inputQuery.addEquation(out_eq)
        #     options = createOptions()
        #
        #     stats = MarabouCore.solve(inputQuery, options, "")
        #     if stats[0]=="sat":
        #         return False
        #
        # return True
