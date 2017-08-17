# When writing sorting toy examples, a linked list can be used to insert an element 
# into a sorted sublist. It may not be always the case because there are arguements on
# the efficiency of linked list is worse than array.
# I just add a simple implementation of Linked List for later use.
# A linked list consists of a list of node. 
#
# Author: Sen Jia
#

class Node:
    def __init__(self,data,nextNode=None):
        self.data = data
        self.nextNode = nextNode
