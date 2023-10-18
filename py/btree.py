#!/usr/bin/env python3
class Node:
    def __init__(self,
                 element,
                 parent=None,
                 left=None,
                 right=None,
                 ) -> None:
        self.left = left
        self.right = right
        self.parent = parent
        self.element = element

class BTree:
    def __init__(self) -> None:
        self.root = None
        self.size = 0

    def __len__(self):
        return self.size

    def add_root(self, e):
        # Only add if the tree is empty
        if self.root is None:
            tmp = Node(element=e)
            self.size = 1
            self.root = tmp

    def add_left(self, p: Node, e):
        if p.left is None:
            tmp = Node(element=e, parent=p)
            p.left = tmp
            self.size += 1

    def add_right(self, p: Node, e):
        if p.right is None:
            tmp = Node(element=e, parent=p)
            p.right = tmp
            self.size += 1
