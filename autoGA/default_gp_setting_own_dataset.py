#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import random
import operator
import csv
import itertools

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import multiprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("./merged.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y)


# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped(
    "MAIN", [type(X_train[col].iloc[0]) for col in X_train], bool, "IN"
)

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)


# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try:
        return np.float64(left / right)
    except ZeroDivisionError:
        return np.float64(1)


pset.addPrimitive(operator.add, [np.float64, np.float64], np.float64)
pset.addPrimitive(operator.sub, [np.float64, np.float64], np.float64)
pset.addPrimitive(operator.mul, [np.float64, np.float64], np.float64)
pset.addPrimitive(protectedDiv, [np.float64, np.float64], np.float64)


# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant(
    name="RAND", ephemeral=partial(np.float64, random.random()), ret_type=np.float64
)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=7)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evaluate(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    y_pred = [func(*data) for _, data in X_train.iterrows()]
    y_labels = np.unique(y)
    y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
    tmp = random.random()
    ret = accuracy_score(y_train, y_pred)
    return (ret,)


def test(individual):
    func = toolbox.compile(expr=individual)
    y_pred = [func(*data) for _, data in X_test.iterrows()]
    y_labels = np.unique(y)
    y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
    return accuracy_score(y_test, y_pred)


toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=6)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


if __name__ == "__main__":
    pool = multiprocessing.Pool()

    pop = toolbox.population(n=200)
    toolbox.register("map", pool.map)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 50, stats, halloffame=hof)

    print(test(hof[0]))
