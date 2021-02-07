# Udacity Deep Reinforcement Learning Project 2: Continuous Control

## Project Details
The code in this repo interacts with a modified version of the [Reacher Environment.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
This environment puts agents in control of double-jointed arms, with the goal of moving the arms to a specific place then leaving them there.  For each step that the agent's arm spends in the goal area, that agent receives a reward of +0.1.  Notably, this environment supports a variable number of agents; in this project, I will work with 20 parallel agents.  

To achieve this goal, the agent takes action by sending the environment a vector of 4 numbers in the range [-1,1].  These numbers correspond to torque which is applied to the two joints of the arm. State information is given to the agent as a vector of length 33; this state information contains data about each arm segment's position, rotation, velocity, and angular velocity.

The environment is considered "solved" when the average 100-episode rolling score is >= +30.  Specifically, since I am solving the 20-agent version of the environment, the average is taken across all agents as well as the 100 episode rolling window.
## My Solution
To solve the environment, I implemented three variations on Deep Q Networks: "vanilla" DQN, Double DQN, and Dueling DQN.
I also implemented a basic experiment runner and serialization format to more easily run the different setups and
compare results.  For more details on my findings, see the writeup in [report.ipynb](report.ipynb)

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
UNITY_ENV_PATH variable in `run_experiments.py` or `run_agent.py` to point to your new version.

## Instructions
The experiment runner, `navigation.experiment.run_and_save_experiment`, accepts dictionaries of keyword args which
modify behavior at the experiment level (number of epochs, epsilon decay rate, etc.) as well as behavior on the agent
level (type of network used, size of hidden layer, double DQN vs "vanilla" Q-value calculations).  By default, the file
`run_experiments.py` is configured to run 5 experiments and write the results to the `experiments` folder.  You can modify
the `EXPERIMENT_SETUPS` list at the top of the file to adjust the experiment setups if you'd like to compare
different setups.  

To run the default experiments, navigate to the root of the project and run the command `python run_experiments.py`.
If there is data in an experiment's folder already, the program will ask you whether you want to overwrite the data or
skip the experiment.  If neither option is chosen, the program will terminate to avoid overwriting the existing
experiment.

The experiments directory contains one folder per experiment run.  Inside each folder are two files: `experiment.json`
and `model_weights.pt`.  `experiment.json` contains experiment metadata, including the parameters used for the experiment,
runtime, and training errors.  `model_weights.pt` contains the final trained weights at the end of the experiment which
can be loaded into a PyTorch model using the command `my_model.load_state_dict(torch.load(weights_path))`.  Alternatively,
you can load the model from the experiment directory without first instantiating a model using the 
`navigation.experiment.load_experiment_model(experiment_folder_path)` function.
