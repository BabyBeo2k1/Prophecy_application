import os
import sys
print(sys.path)
sys.path.append("/home/lqhung2001/Downloads/Marabou")
from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions

class NetVerifier:
    def __init__(self):
        pass
    def isverified(self, net,pattern,output_properties ):
        state_dict=net.state_dict()
        keys=list(state_dict.keys())
        for i, output_property in enumerate(output_properties):
            
            cnt=0
            large=10e6
            small=10e-6
            cnt+=state_dict[keys[0]].shape[0]
            for i,name in enumerate(keys):
                if i%2==0:
                    cnt+=2*state_dict[name].shape[1]
            cnt-=state_dict[keys[-2]].shape[1]
            cnt+=len(output_properties)
            inputQuery=MarabouCore.InputQuery()
            inputQuery.setNumberOfVariables(cnt)
            cur=0
            #set range input
            for i,name in enumerate(keys):
                if i==0:
                    for j in range(state_dict[name].shape[0]):
                        inputQuery.setLowerBound(cur,-large)
                        inputQuery.setUpperBound(cur,large)
                        cur+=1
                if i%2==0 and i>0:
                    for j in range(state_dict[name].shape[0]):
                        inputQuery.setLowerBound(cur,-large)
                        inputQuery.setUpperBound(cur,large)
                        cur+=1
                    for j in range(state_dict[name].shape[0]):
                        if pattern[(i//2)-1][j]:
                            inputQuery.setLowerBound(cur,small)
                            inputQuery.setUpperBound(cur,large)
                            cur+=1
                        else:
                            inputQuery.setLowerBound(cur, 0)
                            inputQuery.setUpperBound(cur, 0)
                            cur += 1
            #set equation
            cur=0
            for i,name in enumerate(keys):
                if i%2==0:
                    for j in range(state_dict[name].shape[1]):#set node j-th of layer i-th
                        equation=MarabouCore.Equation()
                        equation.addAddend(-1,cur+state_dict[name].shape[0]+j)
                        for k in range(state_dict[name].shape[0]):
                            equation.addAddend(state_dict[name][k][j],cur+k)
                                            
                        equation.setScalar(-state_dict[keys[i+1]][j])
                        inputQuery.addEquation(equation)
                    if i<len(keys)-2:
                        for j in range(state_dict[name].shape[1]):
                            #set relu constrain of node j-th of layer i-th
                            MarabouCore.addReluConstraint(inputQuery, cur+j, cur+j+state_dict[name].shape[1])

                    cur+=state_dict[name].shape[1]+state_dict[name].shape[0]
            cur-=len(output_property)
            equationtype=MarabouCore.Equation.EquationType(1)
            out_eq=MarabouCore.Equation(equationtype)
            for i,w in enumerate(output_property):
                out_eq.addAddend(w,i+cur)
            out_eq.setScalar(0.001)
            inputQuery.addEquation(out_eq)
            options = createOptions()

            stats = MarabouCore.solve(x, options, "")
            if stats[0]=="sat":
                return False
                
        return True
