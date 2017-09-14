# When writing sorting toy examples, a linked list can be used to insert an element 
# into a sorted sublist. 
# I just add a simple implementation of Linked List for later use.
# A linked list consists of a list of nodes. 
#
# Author: Sen Jia
#

class Node:
    def __init__(self,data,nextNode=None):
        self.data = data
        self.nextNode = nextNode
