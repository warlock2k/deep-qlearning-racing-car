## Abstract
Reinforcement learning is a strategy through which agents, often computer-based, learn to find an optimal solution to a problem by imitating how humans/animals learn in the real world. Often in reinforcement learning, similar to the real world, the environment rewards or punishes the agent based on how conducive its actions in a particular state are for it to reach the goal. In this project, I applied my learnings from the ECE-559B course to build a self-driving car that learns to drive towards the current goal location on the map while maximizing the reward it obtains from the environment, using the Deep Q Learning algorithm.

## Introduction
Reinforcement learning problems involve an agent learning to take actions at different states, such that the reward it obtains from the environment is maximized. An agent in a reinforcement problem can be any decision-making entity, put inside a Software or real-world environment. It is also important to mention that often an agent is equipped with sensors that give it awareness about its current situation, and this awareness forms the state of an agent. In this project, the agent is a car placed inside an environment built with a python framework called Kivy, a Cross-platform Python Framework for NUI Development. Self-driving cars today make use of supervised learning algorithms to train a machine learning model, such that a curve is fit through the training data points and consequently it can help in the prediction of actions that need to be taken on the real-world data. However the accuracy of the model is dependent on the volume and quality of the data used for training, and furthermore, the model cannot adapt if the environment changes. Hence I decided to use the reinforcement learning approach, where a car can learn about the environment by interacting with it and getting rewards based on the value of the actions that it takes. This very value of actions at different states can be learned by the car using the Q-learning algorithm. Furthermore, Q-learning would be a good starting point because in this case as the statistical description/model of system dynamics is not readily available, and it can only be learned by interacting with the environment. The Q-learning algorithm is based on Temporal difference learning, the simplest form of it is when an agent calculates the value of taking an action at a particular state using the following equation.

## Problem Forumlation
In this section, I will formally define the problem as an MDP (Markov decision process) and discuss individual components in detail.

### Environment
In this problem, the environment is a piece of land (made up of a track, 3 goal locations, some vegetation, and sand, etc.) that provides, information about itself to the agent along with the next states and rewards. To develop such an environment, an open-source Python UI development framework called Kivy is used as it made user interaction programming very easy.

### Agent
The agent is a car, equipped with an orientation and three sand density measurement sensors. The orientation sensor lets that car know of its orientation with respect to the goal location, and the three sand density measurement sensors let the car know of the density of the sand beneath the car on the left, middle and right sides of the car.

### Objective
The car must try to reach the goal as fast as possible from any point on the map while maximizing the reward it obtains from the environment. There are three different goal locations in this problem (i.e A, B, C). As soon as the car reaches one goal (say A), the goal is updated to be at B, and as soon as the car reaches location B, the goal is updated to be at C and this process repeats as the task is continuous. For the car to reach a goal as fast as possible, it must be rewarded for moving towards the goal and penalized for moving away or going off-road. It is important to note that the states and rewards must be defined in such a way that they can contain enough information for the agent to make decisions that make it move towards the goal.

### State
State is a vector that best describes the current situation of an agent in the environment with respect to the goal. Each state vector must contain information, with which the car can take actions that lead to the goal. In this problem, we want the car to move towards the current goal location as fast as possible, this can only happen if the car moves towards the goal while being on the track. Therefore the car is equipped with certain sensors which inform it of its orientation with respect to the goal, and whether if it is being on the top of the road. The values of these sensors form the state vector.

### Total state space
It is important to estimate the total state space as it helps us to pick the most appropriate learning algorithm to solve the problem. After discretizing the orientation sensor to have a resolution of 0.1 degrees, there can be a total of 3600 different states because of it. Furthermore, sand density measurement sensors have a value range between 0 to 1 with a resolution of 0.0025, hence, they contribute a total of 400 * 400 * 400 states to the total state space. Therefore the total state space is estimated to be around 3600 * 400 * 400 * 400. 

### Actions
The actions must be defined in such a way that they allow the agent to interact with the environment, obtain rewards and move towards the goal. Hence, the actions that the car can take are listed below.
- Move forward.
- Move left.
- Move right.

### Rewards
The rewards must be defined intelligently so that the agent can infer from its interaction with the environment as to what's a valuable action in a particular state and what isn't. In this problem, the rewards are deterministic and are defined as follows.

- +0.5 for if the agent moves close to the goal.
- -1 if the car moves away from the goal.
- -5 if the car drives into the sand.
- -1 for every step the cars takes while on top of the road.
- -6 if the car hits the boundary of the map.
