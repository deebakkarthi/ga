#!/usr/bin/env python3
# This is for situtations where a instance member is also the same type
# as the instance
from __future__ import annotations


class BinaryTree:
    class _Node:
        def __init__(
            self,
            element,
            parent=None,
            left=None,
            right=None,
        ):
            self._left = left
            self._right = right
            self._parent = parent
            self._element = element

    class Position:
        """Position is a logical abstraction of a location in
        the binary tree. Instead of revealing the low level
        implementation details of the node, we are wrapping it in a
        implementation-independent class called Position"""

        def __init__(self, container, node) -> None:
            self._container = container
            self._node = node

        def element(self):
            return self._node._element

        def __eq__(self, other):
            """Check if both represent the same position"""
            return type(other) is type(self) and other._node is self._node

    def _validate(self, p: Position) -> _Node:
        """Return associated node if the position is valid"""
        if not isinstance(p, self.Position):
            raise TypeError("p must be proper Position type")
        if p._container is not self:
            raise ValueError("p does not belong to this container")
        if p._node._parent is p._node:
            """Deprecated Nodes"""
            raise ValueError("p is no longer valid")
        return p._node

    def _make_position(self, node: _Node | None):
        """Position() should only be called through this method. This
        ensures that the Position._container is properly set"""
        return self.Position(self, node) if node is not None else None

    def __init__(self) -> None:
        self._root = None
        self._size = 0

    def __len__(self):
        return self._size

    def root(self):
        """Return root position"""
        return self._make_position(self._root)

    def parent(self, p: Position):
        """Return parent position"""
        node = self._validate(p)
        return self._make_position(node._parent)

    def left(self, p: Position):
        node = self._validate(p)
        return self._make_position(node._left)

    def right(self, p: Position):
        node = self._validate(p)
        return self._make_position(node._right)

    def nr_children(self, p: Position):
        """Number of children of position p"""
        node = self._validate(p)
        count = 0
        if node._left is not None:
            count += 1
        if node._right is not None:
            count += 1
        return count

    def is_leaf(self, p: Position):
        return self.nr_children(p) == 0

    def is_root(self, p: Position):
        return self.root() == p

    def is_empty(self):
        return len(self) == 0

    def depth(self, p):
        if self.is_root(p):
            return 0
        else:
            return 1 + self.depth(self.parent(p))

    def sibling(self, p):
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            if p == self.left(parent):
                return self.right(parent)
            else:
                return self.left(parent)

    def children(self, p):
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)

    def _height(self, p):
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self._height(c) for c in self.children(p))

    def height(self, p=None):
        if p is None:
            p = self.root()
        return self._height(p)

    def _add_root(self, e):
        """Place an element e as the root of an empty tree and return
        the new Position
        raise ValueError if tree is non-empty"""
        if self._root is not None:
            raise ValueError("Root Exists")
        self._size = 1
        self._root = self._Node(e)
        return self._make_position(self._root)

    def _add_left(self, p: Position, e):
        """Add new child to position p
        raise ValueError if there is already a child"""
        node = self._validate(p)
        if node._left is not None:
            raise ValueError("Left child already exists")
        node._left = self._Node(e)
        return self._make_position(node._left)

    def _add_right(self, p: Position, e):
        """Add new child to position p
        raise ValueError if there is already a child"""
        node = self._validate(p)
        if node._right is not None:
            raise ValueError("Right child already exists")
        node._right = self._Node(e)
        return self._make_position(node._right)

    def _replace(self, p: Position, e):
        """Replace the element of p and return the old one"""
        node = self._validate(p)
        old = node._element
        node._element = e
        return old

    def _delete(self, p: Position):
        """Delete and replace with p's child
        return p's element
        """
        node = self._validate(p)
        if self.nr_children(p) == 2:
            raise ValueError("p has two children")

        child = node._left if node._left else node._right
        if child is not None:
            child._parent = node._parent
        if node is self._root:
            self._root = child
        else:
            parent = node._parent
            # This check is not actually required as the check for
            # self._root will suffice. But adding this so pyright will
            # shut up
            if parent is not None:
                if node is parent._left:
                    parent._left = child
                else:
                    parent._right = child
        self._size -= 1
        node._parent = node
        return node._element

    def _attach(self, p: Position, t1: BinaryTree, t2: BinaryTree):
        node = self._validate(p)
        if not self.is_leaf(p):
            raise ValueError("p must be a leaf")
        if not (type(self) is type(t1) is type(t2)):
            raise ValueError("All three trees must be of the same type")
        self._size = len(t1) + len(t2)
        if not t1.is_empty():
            # Again check not require. Just for pyright
            if t1._root is not None:
                t1._root._parent = node
                node._left = t1._root
                t1._root = None
                t1._size = 0

        if not t2.is_empty():
            # Again check not require. Just for pyright
            if t2._root is not None:
                t2._root._parent = node
                node._left = t2._root
                t2._root = None
                t2._size = 0
