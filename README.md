[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Deep Deterministic Policy Gradient for an RL Agent in the Continuous Control Unity Environment

### Introduction

This was a project for the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) offered by Udacity. For this project, I trained a robotic arm-like agent to continuously follow a moving object in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment offered by [Unity](https://www.unity.com).

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Critically, this environment contains 20 identical agents, each with its own copy of the environment. The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes. In other words, the score is averaged over all 20 agents in the environment, and the average of this averaged score must be +30 for 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Clone this GitHub repository, place the app file in the repository folder, and unzip (or decompress) the file. 

### Requirements
* NumPy
* PyTorch
* unityagents  

### Training a DQN Agent

To train your own agent, you must first initialize the Reacher environment:

```
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="./Reacher.app")
```

Get the sizes of the environment state and action spaces:

```
# get the default brain of the environment
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# size of the state space
states = env_info.vector_observations
state_size = states.shape[1]
```

Then, you must initialize an agent with the `Agent` class:

```
from agent import Agent
training_agent = Agent(state_size=state_size,
                       action_size=action_size)
```

To train the agent, just call:
```
_, _ = agent.train(env)
```
A `checkpoint_actor.pth` file and a `checkpoint_critic.pth` file will automatically be saved with the model weights in the `checkpoints/` folder.

My model training is completed in `model_training.ipynb`.

### Loading a Pre-Trained DQN Agent

To train your own agent, you must first initialize the Bananas environment and Agent instance like above. Make sure that the parameters of the new Agent instance match those that were used for training.

Once the environment and agent are set up, you can load the actor and critic model weights into the `agent` like so:

```
actor_path = 'checkpoints/checkpoint_actor.pth'
critic_path = 'checkpoints/checkpoint_critic.pth'
agent.actor_local.load_state_dict(torch.load(actor_path))
agent.critic_local.load_state_dict(torch.load(critic_path))
```