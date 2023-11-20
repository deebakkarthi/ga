import operator
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import gp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
from operator import add, mul, sub, lt, eq, and_, or_, not_


def protected_div(left, right):
    if right != 0:
        return left / right
    else:
        return 1


df = pd.read_csv("./merged.csv")
X = df.iloc[:, :-1]
X = X.astype(np.float64)
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y)


func = lambda IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, IN10, IN11, IN12, IN13, IN14, IN15, IN16, IN17, IN18, IN19, IN20, IN21, IN22: lt(
    IN9,
    add(
        add(
            mul(
                mul(
                    add(sub(0.5087350798531884, IN15), sub(IN12, IN17)),
                    sub(protected_div(IN7, IN21), add(IN7, IN12)),
                ),
                mul(
                    sub(mul(IN9, IN21), protected_div(IN7, IN21)),
                    sub(sub(IN4, IN15), sub(IN18, IN14)),
                ),
            ),
            sub(IN18, IN7),
        ),
        mul(IN5, IN6),
    ),
)
y_pred = [func(*data) for _, data in X_test.iterrows()]
y_labels = np.unique(y)
y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
print(accuracy_score(y_test, y_pred))


func = lambda IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, IN10, IN11, IN12, IN13, IN14, IN15, IN16, IN17, IN18, IN19, IN20, IN21, IN22: and_(
    not_(False), not_(or_(lt(IN7, IN9), and_(False, False)))
)
y_pred = [func(*data) for _, data in X_test.iterrows()]
y_labels = np.unique(y)
y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
print(accuracy_score(y_test, y_pred))
