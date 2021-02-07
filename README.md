# Udacity Deep Reinforcement Learning Project 2: Continuous Control

## Project Details
The code in this repo interacts with a modified version of the [Reacher Environment.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
This environment puts agents in control of double-jointed arms, with the goal of moving the arms to a specific place then leaving them there.  For each step that the agent's arm spends in the goal area, that agent receives a reward of +0.1.  Notably, this environment supports a variable number of agents; in this project, I will work with 20 parallel agents.  

To achieve this goal, the agent takes action by sending the environment a vector of 4 numbers in the range [-1,1].  These numbers correspond to torque which is applied to the two joints of the arm. State information is given to the agent as a vector of length 33; this state information contains data about each arm segment's position, rotation, velocity, and angular velocity.

The environment is considered "solved" when the average score across all agents is >= +30.

## My Solution
To solve the environment, I implemented an Advantage Actor Critic model (A2C) with n-step rollout.  For more details on my findings, see the writeup in [report.ipynb](report.ipynb).

## Getting Started

### Python Setup
This project has been tested on Python 3.6; it may work on later versions but is incompatible with earlier ones.
It is recommended that you use a virtual environment using conda or another tool when installing project dependencies.
You can find the instructions for installing miniconda and creating an environment using conda on the
[conda docs](https://docs.conda.io/en/latest/miniconda.html).

### Python Dependencies
After creating and activating your environment (if you're using one), you should install the dependencies for this project
by following the instructions in the [Udacity DRLND Repository.](https://github.com/udacity/deep-reinforcement-learning#dependencies)


### Unity environment
Once you have the python dependencies installed, download the version of the unity environment appropriate for
your operating system.  Links for each operating system can be found below:

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* [Mac Os](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* [Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* [Windows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

After downloading, use 7zip or another archive tool to extract the environment file into the root project directory.
By default, the code is set up to look for the Linux version of the environment, so you will need to modify the
UNITY_ENV_PATH variable in `train_agent.py` or `run_agent.py` to point to your new version.

## Running The Code
The `train_agent.py` python file at the project root contains the logic necessary to train both the actor and the critic networks.  You can run it with the command `python train_agent.py`.  Note that you will need to update the UDACITY_ENV_PATH variable to point to your version of the Unity environment.  By changing the other variables in ALL_CAPS at the top of the file, you can modify various hyperparameters used by the agent during training.  After training, this script will store the final actor and critic weights in the `model_weights` directory. It will also store the average score per-episode as a csv in the `scores` directory.

The `run_agent.py` python file at the project root contains the logic necessary to run the network in the environment.  You can run it with the command `python run_agent.py`. Once again, you will need to update the UDACITY_ENV_PATH variable to point to your version of the Unity environment before running this script.  By modifying the `TRAIN_MODE` variable to False, you can watch the agent as it runs.

# Acknowledgements
During completion of this project, I used several sources as references and inspiration for my implementation.  This does not include any direct code usage, but does include hyperparameter decisions. These sources are listed below:
- [Shangtong Zhang's A2C implementation](https://github.com/ShangtongZhang/DeepRL)
- [Alvaro Durán Tovar's continuous A2C implementation](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-continuous-case-implementation-f55ce5da6b4c)
- [Chris Yoon's "Understanding Actor Critic Methods and A2C" Medium Post](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- [*Continuous control with deep reinforcement learning* by Lillicrap et al.](https://arxiv.org/abs/1509.02971v6)