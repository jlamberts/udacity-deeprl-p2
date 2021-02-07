from continuous_control.networks import ActorNetwork
from continuous_control.env import UnityEnvWrapper
import torch

# Set Unity settings
UNITY_ENV_PATH = "Reacher_Linux_NoVis/Reacher.x86_64"
TRAIN_MODE = True

# Set number of episodes to run
NUM_EPISODES = 10

# Set Pretrained Model Info Here
ACTOR_HIDDEN_SIZE = 64
ACTOR_BATCHNORM = False
WEIGHTS_PATH = "model_weights/actor_weights.pt"


if __name__ == "__main__":
    env = UnityEnvWrapper(file_name=UNITY_ENV_PATH, train_mode=TRAIN_MODE)

    actor = ActorNetwork(
        hidden_layer_size=ACTOR_HIDDEN_SIZE, batchnorm_inputs=ACTOR_BATCHNORM
    )
    actor.load_state_dict(torch.load(WEIGHTS_PATH))

    for episode in range(NUM_EPISODES):
        all_rewards = 0
        env.reset()
        next_state = torch.tensor(env.states, dtype=torch.float)
        done = [False]

        while not any(done):
            dists = actor(next_state)
            actions = torch.clamp(dists.sample(), -1, 1).numpy()
            next_state, reward, done = env.step(actions)
            all_rewards += float(reward.mean())

        print(f"Average rewards for episode {episode}: {all_rewards}")