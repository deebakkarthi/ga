package main

import (
	"fmt"
	"math/rand"
	"sort"
)

type Gene = byte

type DNA struct {
	genes [geneLen]Gene
}

const charset = "abcdefghijklmnopqrstuvwxyz "
const goalString string = "dynamic configuration of algorithms"
const geneLen = len(goalString)
const mutRate = 0.01
const popSize = 150

func geneGen() Gene {
	return Gene(charset[rand.Intn(len(charset))])
}

func populationGen(n int) []DNA {
	ret := make([]DNA, n)
	for i := 0; i < n; i++ {
		var tmp DNA
		for j := 0; j < geneLen; j++ {
			tmp.genes[j] = geneGen()
		}
		ret[i] = tmp
	}
	return ret
}

func fitness(d DNA) int {
	tmp := 0
	for i, v := range d.genes {
		if v == goalString[i] {
			tmp++
		}
	}
	return tmp
}

func selection(population *[]DNA, fitness func(DNA) int) []DNA {
	var matingPool []DNA
	sort.Slice(*population, func(i, j int) bool {
		return fitness((*population)[i]) < fitness((*population)[j])
	})
	for i, v := range *population {
		for j := 0; j < i; j++ {
			matingPool = append(matingPool, v)
		}
	}
	return matingPool
}

func crossover(a, b DNA) DNA {
	var child DNA
	midpoint := rand.Intn(geneLen)
	for i := 0; i < geneLen; i++ {
		if i < midpoint {
			child.genes[i] = a.genes[i]
		} else {
			child.genes[i] = b.genes[i]
		}
	}
	return child
}

func mutate(d *DNA, mutRate float32) {
	for i := range d.genes {
		if rand.Float32() < mutRate {
			d.genes[i] = geneGen()
		}
	}
}

func reproduction(population *[]DNA, crossover func(a, b DNA) DNA,
	mutate func(d *DNA, mutRate float32)) {
	matingPool := selection(population, fitness)
	for i := range *population {
		var a, b DNA
		for {
			a = matingPool[rand.Intn(len(matingPool))]
			b = matingPool[rand.Intn(len(matingPool))]
			if a != b {
				break
			}
		}
		child := crossover(a, b)
		mutate(&child, mutRate)
		(*population)[i] = child
	}

}

func main() {
	population := populationGen(popSize)
	generations := 0
	for string(population[len(population)-1].genes[:]) != goalString {
		reproduction(&population, crossover, mutate)
		fmt.Printf("%s %d\n", population[len(population)-1], fitness(population[len(population)-1]))
		generations++
	}
	fmt.Println(generations)
}
