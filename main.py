# Author: Porter Zach
# Python 3.9

import net
import copy
import torch
import numpy as np
import environment

# Some genetic algorithm techniques derived from Paras Chopra's "Reinforcement learning without gradients: evolving agents using Genetic Algorithms"
# https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f

def random_agents(amount):
    agents = []

    for _ in range(amount):
        agent = net.NeuralNetwork()

        for param in agent.parameters():
            param.requires_grad = False

        agents.append(agent)

    return agents

# number of agents per generation
num_agents = 100
agents = random_agents(num_agents)

# number of best agents to choose for reproduction
top_agents = 10

# number of generations to simulate
num_generations = 1000

steps_per_test = 1000

mutation_power = 0.2 # hyper-parameter

def reproduce(parents, best_parents):
    children = []

    # take the best agents and breed them to create N-1 children
    for _ in range(len(parents) - 1):
        # get 2 random parents
        # it's ok if they're the same, mutations will prevent it from being identical
        parent_index_1 = best_parents[np.random.randint(len(best_parents))]
        parent_index_2 = best_parents[np.random.randint(len(best_parents))]
        # add the mutated child to the list of children
        children.append(mutate(crossover(parents[parent_index_1], parents[parent_index_2])))

    # add the very best parent to the list of children to prevent too much genetic drift
    children.append(parents[best_parents[0]])

    return children

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)

    parent2_params = [param for param in parent2.parameters()]
    
    for i, param in enumerate(child.parameters()):
        if len(param.shape) == 2: # weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    if np.random.randn() < 0.5:
                        param[i0][i1] = parent2_params[i][i0][i1]

        elif len(param.shape) == 1: # biases of linear layer
            for i0 in range(param.shape[0]):
                if np.random.randn() < 0.5:
                        param[i0] = parent2_params[i][i0]

    return child

def mutate(agent):
    for param in agent.parameters():
        if len(param.shape) == 2: # weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1] += mutation_power * np.random.randn()

        elif len(param.shape) == 1: # biases of linear layer
            for i0 in range(param.shape[0]):
                param[i0] += mutation_power * np.random.randn()

    return agent

def run_agents(agents, display = False):
    rewards = []

    for i, agent in enumerate(agents):
        # use net in evaluation mode; no training necessary
        agent.eval()

        observation = env.reset()

        reward_total = 0

        for _ in range(steps_per_test):
            input = torch.tensor(observation).type(torch.FloatTensor)
            actions = agent(input).detach().numpy()
            observation, reward, done = env.step(actions)
            reward_total += reward

            if done:
                break

            if display:
                env.draw()

        rewards.append(reward_total)

    return rewards

# disable gradients; we aren't doing any backprop
torch.set_grad_enabled(False)

"""Edit these variables to change various features.
"""
# the course that the cars will attempt
course = "course1.png"
# whether to save the best model from each generation in the file named save_name
save_generation_best = False
# the file to save the best model in
save_name = "model2.pth"

# if true, program will repeatedly show model_to_display attempt the selected course.
# if false, many models will be randomly initialized and the NN and GA process will begin.
display_saved_model = True
# the model to display if the above value is true
model_to_display = "model1.pth"

env = environment.Environment(course)

if not display_saved_model:
    for generation in range(num_generations):
        print("Running generation: {0}, number of agents: {1}".format(generation, len(agents)))

        # get agent fitness
        rewards = run_agents(agents)

        # sort by fitness
        # reverses (argsort is ascending but we want best to be first) and only get best few
        best_parents = np.argsort(rewards)[::-1][:top_agents]

        print("Displaying best agent in generation {0}.".format(generation))
        run_agents([agents[best_parents[0]]], display=True)

        if save_generation_best:
            print("Saving best agent as {0}".format(save_name))
            torch.save(agents[best_parents[0]], save_name)

        agents = reproduce(agents, best_parents)
else:
    print("Displaying model {0}".format(model_to_display))
    model = torch.load(model_to_display)
    while True:
        run_agents([model], display=True)
