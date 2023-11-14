#!/usr/bin/env python3

import operator
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import gp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
import pandas as pd


# TODO Make these local instead of global
df = pd.read_csv(
    "./german.tsv",
    sep="\t",
    names=[
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
        "A17",
        "A18",
        "A19",
        "A20",
        "A21",
        "A22",
        "A23",
        "target",
    ],
)

X = df.iloc[:, :-1]
X = X.astype(np.float64)
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y)


def individual_create():
    """
    00 tree_type - ARITHMETIC, LOGICAL, DECISION
    01 pop_size - 100, 200, 300
    02 tree_gen_method - FULL, GROW, HALF_AND_HALF
    03 initial_depth- [2, 15] [2, 8] for decision trees
    04 offspring_depth - [2, 15] [2, 8] for decision trees
    05 selection_method = FITNESS_PROP, TOURNAMENT
    06 selection_size - [2, 10]
    07 reproduction_rate - [0, 100]
    08 mut_type - GROW, SHRINK
    09 max_mut_depth - [2, 6]
    10 control_flow - FIXED, RANDOM
    11 op_combination - [0, 6] CX_MUT, CX, MUT, CX_MUT_PRESET, MUT_RAND_CX, CX_RAND_MUT, CREATE
    12 fitness_type - [0, 4] ACC, F_SCORE, W_ACC, RAND_W_ACC, TPR
    13 nr_gen - [50, 200]
    """
    tmp = np.empty(14, dtype=int)
    # TODO do decision tree later
    # change this to (0, 2)
    tmp[0] = random.randint(0, 1)
    tmp[1] = random.choice([100, 200, 300])
    tmp[2] = random.randint(0, 2)
    if tmp[0] == 2:
        tmp[3] = random.randint(2, 8)
        tmp[4] = random.randint(2, 8)
    else:
        tmp[3] = random.randint(2, 15)
        tmp[4] = random.randint(2, 15)
    tmp[5] = random.randint(0, 1)
    tmp[6] = random.randint(2, 10)
    tmp[7] = random.randint(0, 100)
    tmp[8] = random.randint(0, 1)
    tmp[9] = random.randint(2, 6)
    tmp[10] = random.randint(0, 1)
    tmp[11] = random.randint(0, 6)
    tmp[12] = random.randint(0, 4)
    tmp[13] = random.randint(50, 200)
    return tmp


def mutFlipValue(individual, indpb=0.05):
    tmp = individual_create()
    for idx, _ in enumerate(individual):
        if random.random() < indpb:
            individual[idx] = tmp[idx]
    return (individual,)


def protected_div(left, right):
    if right != 0:
        return left / right
    else:
        return 1


def pset_create(tree_type: int):
    if tree_type == 0:
        pset = gp.PrimitiveSetTyped("main", [type(X[col][0] for col in X)], float, "IN")
        pset.addPrimitive(operator.add, [np.float64, np.float64], np.float64)
        pset.addPrimitive(operator.sub, [np.float64, np.float64], np.float64)
        pset.addPrimitive(operator.mul, [np.float64, np.float64], np.float64)
        pset.addPrimitive(protected_div, [np.float64, np.float64], np.float64)
    elif tree_type == 1:
        pset = gp.PrimitiveSetTyped("main", [type(X[col][0] for col in X)], bool, "IN")
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.not_, [bool], bool)
        pset.addPrimitive(operator.eq, [np.float64, np.float64], bool)
        pset.addPrimitive(operator.ne, [np.float64, np.float64], bool)
    # TODO decision tree
    #   elif tree_type == 2:
    #       pset = gp.PrimitiveSetTyped("main", [], type(y[0]), "IN")
    else:
        raise (ValueError("TreeType belongs to [0, 1, 2]"))
    return pset


def inner_evaluate(individual, tree_type, fitness_type, toolbox):
    func = toolbox.compile(expr=individual)
    y_pred = [func(*data) for data in X_train.iterrows()]
    y_labels = y.unique()
    # Output of arithmetic trees is float
    if tree_type == 0:
        y_pred = map(lambda x: y_labels[0] if x > 0 else y_labels[1], y_pred)
    elif tree_type == 1:
        y_pred = map(lambda x: y_labels[0] if x else y_labels[1], y_pred)
    else:
        raise ValueError("tree_type should be in [0, 2]")

    if fitness_type == 0:
        return accuracy_score(y_true=y_train, y_pred=y_pred)
    elif fitness_type == 1:
        return f1_score(y_train, y_pred)
    elif fitness_type == 2:
        return (accuracy_score(y_train, y_pred) + f1_score(y_train, y_pred)) / 2
    elif fitness_type == 3:
        tmp = random.random()
        return tmp * accuracy_score(y_train, y_pred) + (1 - tmp) * f1_score(
            y_train, y_pred
        )
    elif fitness_type == 4:
        return recall_score(y_train, y_pred)


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    use_varAnd=True,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        if use_varAnd:
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        else:
            offspring = algorithms.varOr(
                offspring, toolbox, len(offspring), cxpb, mutpb
            )

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# TODO implement accuracy as the fitness measure
def eval_autoga(individual):
    pset = pset_create(individual[0])
    creator.create("InnerFitnessMax", base.Fitness, weights=(1.0,))
    creator.create("InnerIndividual", gp.PrimitiveTree, fitness=creator.InnerFitnessMax)

    toolbox = base.Toolbox()

    if individual[2] == 0:
        toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=individual[3])
    elif individual[2] == 1:
        toolbox.register("expr", gp.genGrow, pset=pset, min_=2, max_=individual[3])
    elif individual[2] == 2:
        toolbox.register(
            "expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=individual[3]
        )
    else:
        raise ValueError("tree_gen_method should be from [0, 1, 2]")

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register(
        "evaluate",
        inner_evaluate,
        fitness_type=individual[12],
        toolbox=toolbox,
        tree_type=individual[0],
    )
    if individual[5] == 0:
        toolbox.register("select", tools.selRoulette)
    elif individual[5] == 1:
        toolbox.register("select", tools.selTournament, tournsize=individual[6])
    else:
        raise ValueError("selection_method should be from [0, 1]")

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=2, max_=individual[9])

    if individual[8] == 0:
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    elif individual[8] == 1:
        toolbox.register("mutate", gp.mutShrink)
    else:
        raise ValueError("mut_type should be from [0, 1]")

    use_varAnd = True
    if individual[11] == 0:
        cxpb = individual[7]
        mutpb = 100 - cxpb
        mutpb = float(mutpb / 100)
        cxpb = float(cxpb / 100)
    elif individual[11] == 1:
        cxpb = 1
        mutpb = 0
    elif individual[11] == 2:
        cxpb = 0
        mutpb = 1
    elif individual[11] == 3:
        cxpb = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])
        mutpb = 100 - cxpb
        mutpb = float(mutpb / 100)
        cxpb = float(cxpb / 100)
    elif individual[11] == 4:
        cxpb = random.random()
        mutpb = 1
    elif individual[11] == 5:
        cxpb = 1
        mutpb = random.random()
    elif individual[11] == 6:
        cxpb = 0
        mutpb = 0
        use_varAnd = False
    else:
        raise ValueError("op_combination should be in [0, 6]")

    pop = toolbox.population(n=individual[1])
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = eaSimple(
        pop,
        toolbox,
        cxpb,
        mutpb,
        individual[13],
        stats,
        halloffame=hof,
        use_varAnd=use_varAnd,
    )
    return (sum(individual),)


def main():
    random.seed(42)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, individual_create
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_autoga)
    toolbox.register("mate", tools.cxUniform, indpb=0.80)
    toolbox.register("mutate", mutFlipValue, indpb=0.10)
    # TODO Implement elitism
    toolbox.register("select", tools.selRoulette)
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=50,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )
    return


if __name__ == "__main__":
    main()
