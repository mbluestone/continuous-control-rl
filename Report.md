# Report

### Model

The agent in this project learns with a Deep Deterministic Policy Gradient learning algorithm. As an agent explores the environment - through simulation - the DDPG algorithm learns both the value of choosing specific actions in specific states and also a policy for choosing actions when in specific states through observation of final cumulative reward after an entire episode of exploration.

What makes the algorithm "deep" is that two neural networks are used to model the value of state-action pairs (Critic Model) and the policy for choosing actions from states (Actor Model). In this project, the Critic Model consists of a fully-connected neural network with 3 linear layers and 128 nodes in the hidden layers, where the input is the current state of the environment, the action taken is added after the first layer, and the output is a number, which represents the value of the state-action pair. The Actor Model consists of a fully-connected neural network with 3 linear layers and 128 nodes in the hidden layers, where the input is the current state of the environment, and the output is a vector that is the same size as the action space, where the highest value in the vector indicates which action should be taken.

The hyperparameters were set to:
* Replay buffer size = 1000000
* Batch size = 128
* Discount factor (Gamma) = 0.99
* Target network soft update parameter (Tau) = 0.001
* Learning rate for actor model = 0.0001
* Learning rate for critic model = 0.001
* How often to update the network (in timepoints) = 20
* Number of learning passes each time learning occurs = 10

### Performance

The goal of this model is to gain an average final reward of at least +30 over 100 episodes. The graph below show's the reward of the DDPG agent at each timepoint and also averaged across 100 episodes.

![model_training](https://github.com/mbluestone/continuous_control_rl/blob/master/img/model_training.png)

Using 20 simultaneous arms, the model learns very quickly: it only takes 99 episodes to learn this environment.

### Ideas for Future Work

An obvious first next step would be to perform hyperparameter optimization to determine the best hyperparameters for training the DDPG agent. This could potentially lead to an agent that learns faster.

Another potential next step would be implement different variations of the Critic model. One thought would be to using the Dueling DQN architecture for the Critic model.

The [D4PG](Distributed Distributional Deterministic Policy Gradients (D4PG)) algorithm would also probably work quite well for this problem.
