import argparse
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import matplotlib.pyplot as plt

from pokemon_rl.environment.cartpole_env import CartPoleImageWrapper
from pokemon_rl.rl.dqn_agent import DQNAgent

def main(args):
    gym_env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = CartPoleImageWrapper(gym_env)
    agent = DQNAgent(
          args,
          (4, 84, 84),
          2)
    # env = gym.make('CartPole-v1')
    # agent = DQNAgent(
    #       args,
    #       (4, ),
    #       2)


    writer = SummaryWriter()
    train_iter = 0
    for ep_idx in range(10000):
        done = terminated = False
        state, _ = env.reset()
        total_reward = 0
        i = 0

        while not done and not terminated:
            action = agent(state)
            # if i % 2 == 0:
            #     fig, ax = plt.subplots(1)
                

            #     ax.imshow(state.reshape(84, 84, 1), interpolation='nearest')
            #     plt.show()
            i += 1
            if ep_idx >= 50 and i % 2 == 0:
                loss = agent.update()
                train_iter += 1
                writer.add_scalar("Loss/value", loss.item(), train_iter)

 
            next_state, reward, done, terminated, _ = env.step(action)
            total_reward += reward
            agent.add_trajectory(state, action, reward, next_state, done)
            state = next_state

        if ep_idx >= 50:
            # for i in range(64):
            #     agent.update()
            agent.mark_episode()
        print(f"EPISODE {ep_idx} REWARD {total_reward} epsilon {agent.epsilon}")
        writer.add_scalar("Episode/total_reward", total_reward, ep_idx)
        writer.flush()
    writer.close()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--continuous", action="store_true",
            help="Indicates the RL alg. should ignore done.")
        parser.add_argument("--lr", type=float, default=3e-4,
            help="Learning rate used for updating the models.")
        parser.add_argument("--gamma", type=float, default=0.995,
            help="Gamma used for computing the return.")
        parser.add_argument("--tgt_tau", type=float, default=0.01,
            help="Tau used for smooth target updates.")
        parser.add_argument("--eps", type=float, default=0.99,
            help="Initial epsilon value used for epsilon-greedy exploration.")
        parser.add_argument("--eps_decay", type=float, default=0.998,
            help="The ammount epsilon is decayed after each episode.")
        parser.add_argument("--eps_min", type=float, default=0.05,
            help="Minimum epsilon value.")
        parser.add_argument("--replay_buffer_size", type=int, default=int(1e4),
            help="Maximum size of the replay buffer.")
        parser.add_argument("--batch_size", type=int, default=64,
            help="Batch size for training the model.")
        
        main(parser.parse_args())
