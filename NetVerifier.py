import os
import sys
print(sys.path)
sys.path.append("/home/lqhung2001/Downloads/Marabou")
from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions

class NetVerifier:
    def __init__(self):
        pass
    def isverified(self, state_dict:dict,pattern, input_layer=None, output_layer=None,input_bound=None ):
        # inputCount=0
        # large=10e6
        # small=10e-6
        # #input constrain
        # inputQuery = MarabouCore.InputQuery()
        # for i in range(input_layer,output_layer):
        #     for j in range(len(pattern[i])):
        #         inputCount+=2
        # inputQuery.setNumberofVariables(inputCount)
        # curCount=0
        # if (input_layer==0):
        #     for i in range(len(input_bound)):
        #         inputQuery.setLowerBound(i,input_bound[i][0])
        #
        #         inputQuery.setUpperBound(i,input_bound[i][0])
        #     curCount=len(inputCount)
        # for i in range(input_layer,output_layer):
        #     for j in range(len(pattern[i])):
        #         if pattern[i][j]:
        #             inputQuery.setLowerBound(curCount,small)
        #             inputQuery.setUpperBound(curCount,large)
        #         else:
        #             inputQuery.setLowerBound(curCount, 0)
        #             inputQuery.setUpperBound(curCount, 0)
        # #equation
        # keys=list(state_dict.keys())
        return False
