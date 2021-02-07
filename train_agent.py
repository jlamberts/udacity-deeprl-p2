from continuous_control.networks import ActorNetwork, CriticNetwork
from continuous_control.env import UnityEnvWrapper
import pandas as pd
import torch

# Set Unity settings
UNITY_ENV_PATH = "Reacher_Linux_NoVis/Reacher.x86_64"
TRAIN_MODE = True

# Set number of episodes to use when training
NUM_EPISODES = 1000

# Set Hyperparameters Here
ACTOR_HIDDEN_SIZE = 64
ACTOR_LR = 1e-3
ACTOR_BATCHNORM = False

CRITIC_HIDDEN_SIZE = 64
CRITIC_LR = 1e-3
CRITIC_BATCHNORM = False

GAMMA = 0.99
ENTROPY_WEIGHT = 5e-4
ROLLOUT_LENGTH = 5


def get_tensor_from_rollout(rollout, key):
    """Helper function for n-step rollout.

    Pulls a key out of each dict in a list of dictionaries, then concatenates them into a single tensor.
    """
    return torch.cat([r[key] for r in rollout])


if __name__ == "__main__":
    env = UnityEnvWrapper(file_name=UNITY_ENV_PATH, train_mode=TRAIN_MODE)

    # initialize networks
    actor = ActorNetwork(
        hidden_layer_size=ACTOR_HIDDEN_SIZE, batchnorm_inputs=ACTOR_BATCHNORM
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)

    critic = CriticNetwork(
        hidden_layer_size=CRITIC_HIDDEN_SIZE, batchnorm_inputs=CRITIC_BATCHNORM
    )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    scores = []

    for episode in range(NUM_EPISODES):
        done = False
        score = 0
        env.reset()
        state = torch.tensor(env.states, dtype=torch.float)
        step_count = 0
        actor_loss_total = 0
        critic_loss_total = 0
        policy_means = torch.zeros(4)
        policy_stds = torch.zeros(4)

        while not done:
            rollout = []
            # n-step rollout
            for _ in range(ROLLOUT_LENGTH):
                values = critic(state)
                dists = actor(state)
                actions = torch.clamp(dists.sample(), -1, 1)
                policy_means += dists.mean.mean(0)
                policy_stds += dists.stddev.mean(0)
                next_state, reward, done = env.step(actions.numpy())
                score += float(reward.mean())
                episode_done_mask = (
                    1.0 - torch.tensor(done, dtype=torch.float)
                ).unsqueeze(-1)

                rollout.append(
                    {
                        "state": state,
                        "value": values,
                        "actions": actions,
                        "reward": reward,
                        "done_mask": episode_done_mask,
                        "log_probs": dists.log_prob(actions),
                        "entropy": dists.entropy(),
                    }
                )

                state = next_state

                # need to terminate early if we finish because the unity env will just keep giving
                # the last step's rewards otherwise
                if any(done):
                    break

            # calculate value and advantage for each timestep in the rollout
            future_value = critic(state)
            for i in reversed(range(len(rollout))):
                rollout_dict = rollout[i]

                future_value = (
                    rollout_dict["reward"]
                    + GAMMA * future_value * rollout_dict["done_mask"]
                )
                advantage = future_value - rollout_dict["value"]
                rollout_dict["future_value"] = future_value.detach()
                rollout_dict["advantage"] = advantage.detach()

            advantage = get_tensor_from_rollout(rollout, "advantage")
            log_probs = get_tensor_from_rollout(rollout, "log_probs")
            entropy = get_tensor_from_rollout(rollout, "entropy")
            future_value = get_tensor_from_rollout(rollout, "future_value")
            value = get_tensor_from_rollout(rollout, "value")

            # update critic weights
            critic_loss = 0.5 * (future_value - value).pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
            critic_optimizer.step()
            critic_loss_total += float(critic_loss)

            # update actor weights
            policy_loss = -(log_probs * advantage.detach()).mean()
            entropy_loss = -entropy.mean() * ENTROPY_WEIGHT
            actor_loss = policy_loss + entropy_loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
            actor_optimizer.step()
            actor_loss_total += float(actor_loss)

            done = any(done)
            step_count += 1
        if episode % 5 == 0:
            print(f"episode {episode} finished with average score {score}")
        scores.append(score)

    # save results
    pd.Series(scores).to_csv("scores/training_scores.csv")
    torch.save(actor.state_dict(), "model_weights/actor_weights.pt")
    torch.save(critic.state_dict(), "model_weights/critic_weights.pt")