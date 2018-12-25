"""
from types import MethodType
 
def print_classname(a):
print a.__class__.__name__
 
class A(object):
pass
 
# this assigns the method to the instance a, but not to the class definition
a = A()
a.print_classname = MethodType(print_classname, a, A)
 
# this assigns the method to the class definition
A.print_classname = MethodType(print_classname, None, A)
"""

class SharedObjects:
    def __init__(self, con):
        self.connection = con
        
    