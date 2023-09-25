'''
Implementation of PDE operators to enfore physical consistency
Does not solve the system, only computes residuals
Author: Christian Jacobsen, University of Michigan 2023
'''

class DarcyFlow(nn.Module):
    def __init__(self, 
