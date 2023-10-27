#!/usr/bin/env python3
import random
import pandas as pd
from collections import deque

import numpy as np

LEAF_PROB = 0.1


class Node:
    def __init__(
        self,
        parent=None,
        lchild=None,
        rchild=None,
        is_leaf=False,
        feature=None,
        label=None,
        target=None,
    ) -> None:
        self.parent = parent
        self.lchild = lchild
        self.rchild = rchild

        self.is_leaf = is_leaf

        self.feature = feature
        self.target = target

        self.label = label
        self.depth = 0

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"{self.label}"
        else:
            return f"{self.feature} == {self.target}"

    def add_child(self, child):
        if not self.is_leaf:
            if not self.lchild:
                self.lchild = child
                return
            self.rchild = child
        return


def dtree_empty_create(max_depth: int, leaf_prob: float):
    curr_depth = 0
    root = Node()
    q = deque([root])
    while q:
        curr = q.popleft()
        # Add 2 children
        for _ in range(2):
            tmp: Node
            if curr.depth < max_depth - 1 and (random.random() > leaf_prob):
                tmp = Node(parent=curr, is_leaf=False)
                curr.add_child(tmp)
                q.append(tmp)
            else:
                tmp = Node(parent=curr, is_leaf=True)
                curr.add_child(tmp)
            tmp.depth = curr.depth + 1
            curr_depth = max(curr_depth, tmp.depth)
    return root


def evaluate(root: Node, row):
    curr = root
    while curr:
        if curr.is_leaf:
            return curr.label
        else:
            feature = curr.feature
            if getattr(row, feature) == curr.target:
                curr = curr.lchild
            else:
                curr = curr.rchild


def dtree_create(feature_dict, label, max_depth):
    """
    Returns a randomly generated decision tree

    Parameters
    ----------
        feature_dict: {str: [Node]}
            A dictionary from feature names to permissible values
        label: list of str
            Output labels
        max_depth: int
            maximum depth of the dtree


    Returns
    -------
        dtree: Node
            Root node of the decision tree
    """
    root = dtree_empty_create(max_depth, LEAF_PROB)

    def dfs(node: Node):
        if node:
            if node.is_leaf:
                node.label = random.choice(feature_dict[label])
            else:
                # Pick a feature other than the output
                tmp = label
                while tmp == label:
                    tmp = random.choice(list(feature_dict.keys()))
                node.feature = tmp
                node.target = random.choice(feature_dict[tmp])
            if node.lchild:
                dfs(node.lchild)
            if node.rchild:
                dfs(node.rchild)

    dfs(root)
    return root


def dtree_print(root: Node, depth: int = 0):
    if root:
        if root.is_leaf:
            print("\t" * depth, root.label)
        else:
            print("\t" * depth, f"{root.feature} == {root.target}")
        if root.lchild:
            dtree_print(root.lchild, depth + 1)
        if root.rchild:
            dtree_print(root.rchild, depth + 1)
    return


def dtree_height(root: Node):
    def dfs(root: Node):
        if root.is_leaf:
            return root.depth
        else:
            return max(dfs(root.lchild), dfs(root.rchild))

    return dfs(root)


def dtree_rleaf(root: Node):
    curr = root
    while curr and not curr.is_leaf:
        if curr.rchild:
            curr = curr.rchild
        else:
            return curr
    return curr


def dtree_len(root: Node):
    count = 0

    def dfs(root: Node):
        nonlocal count
        if root:
            count += 1
        if root.lchild:
            dfs(root.lchild)
        if root.rchild:
            dfs(root.rchild)

    dfs(root)
    return count


def feature_dict_create(df: pd.DataFrame):
    """
    Returns a dict of feature names to unique values of that feature

    Parameters
    ----------
    df: pandas.DataFrame

    Returns
    -------
    feature_dict: {str: [str]}
    """
    feature_dict = {}
    for col in df:
        feature_dict[col] = df[col].unique().tolist()
    return feature_dict


def fitness(root, df: pd.DataFrame, label: str):
    """
    Returns accuracy
    """
    correct = 0
    for row in df.itertuples():
        tmp = evaluate(root, row)
        if tmp == getattr(row, label):
            correct += 1
    return correct / len(df)


def selection(population, fitness):
    """
    Returns a mating pool where each element is present as many times as
    its rank in an ascendingly sorted population
    """
    mating_pool = []
    population = sorted(population, key=fitness)
    for i, v in enumerate(population):
        for _ in range(i):
            mating_pool.append(v)
    return mating_pool


def crossover(a, b):
    def inorder(root: Node, replace_prob: float):
        if random.random() < replace_prob:
            return root
        else:
            if root.lchild:
                inorder(root.lchild, replace_prob)
            if root.rchild:
                inorder(root.rchild, replace_prob)

    replace_prob = 0.3
    print(replace_prob)
    replace_a = inorder(a, replace_prob)
    if replace_a is None:
        replace_a = dtree_rleaf(a)
    replace_b = inorder(b, replace_prob)
    if replace_b is None:
        replace_b = dtree_rleaf(b)
    print(replace_a)
    print(replace_b)
    return


def main():
    df = pd.read_csv("./play_tennis.csv")
    feature_dict = feature_dict_create(df)
    label = "Play"
    max_depth = 6
    root = dtree_create(feature_dict, label, max_depth)
    dtree_print(root)
    print(f"Accuracy: {fitness(root, df, label):.2f}%")


if __name__ == "__main__":
    main()
