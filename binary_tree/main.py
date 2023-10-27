#!/usr/bin/env python3
import pandas as pd
from binary_tree import BinaryTree

"""
TODO
----

- 
"""


class DecisionTree(BinaryTree):
    def __init__(self) -> None:
        super().__init__()

    def create_empty_tree(self, max_depth, leaf_proba):
        curr_depth = self.depth()
        root = self.root()



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


def main():
    df = pd.read_csv("./play_tennis.csv")
    feature_dict = feature_dict_create(df)
    print(feature_dict)
    dtree = DecisionTree()
    return


if __name__ == "__main__":
    main()
