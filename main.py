from pokemon_rl.utils import set_seed
import argparse

def main(args):
    # Must call set seed before importing other modules
    set_seed(args.seed)
    from torch.utils.tensorboard import SummaryWriter
    import gymnasium as gym
    import matplotlib.pyplot as plt

    from pokemon_rl.environment.cartpole_env import CartPoleImageWrapper
    from pokemon_rl.environment.lunar_env import LunarImageWrapper
    from pokemon_rl.rl.dqn_agent import DQNAgent

    gym_env = gym.make('LunarLander-v2', render_mode="rgb_array")
    env = LunarImageWrapper(gym_env)
    # gym_env = gym.make('CartPole-v1', render_mode="rgb_array")
    # env = CartPoleImageWrapper(gym_env)
    agent = DQNAgent(
          args,
          env.observation_space.shape,
          gym_env.action_space.n)
    suffix = "_betanstep_cv"

    # env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v1')
    # agent = DQNAgent(
    #       args,
    #       env.observation_space.shape,
    #       env.action_space.n)
    # suffix = "_ep_buffer_multepsample"


    writer = SummaryWriter(comment=f"_nstep_{args.n_step}_buffer_{args.replay_buffer_size}_sprlam_{args.spr_lam}_BufferEps_{args.buffer_episodes}{suffix}")
    train_iter = 0

    # Reset should only be called with the seed once
    env.reset(seed=args.seed)
    
    for ep_idx in range(10000):
        done = terminated = False
        state, _ = env.reset()
        total_reward = 0
        i = 0
        num_steps = 0
        ep_dict = {
             "states": [],
             "next_states": [],
             "rewards": [],
             "actions": [],
             "dones": [],
        }
        while not done and not terminated:
            num_steps += 1
            action = agent(state)
            # if i % 2 == 0:
            #     fig, ax = plt.subplots(4)
                
            #     for i in range(4):
            #         ax[i].imshow(state[i], interpolation='nearest')
            #         ax[i].set_axis_off()
            #     plt.show()
            # i += 1

 
            next_state, reward, done, terminated, _ = env.step(action)
            ep_dict["states"].append(state)
            ep_dict["next_states"].append(next_state)
            ep_dict["rewards"].append(reward)
            ep_dict["actions"].append(action)
            ep_dict["dones"].append(done)


            total_reward += reward
            agent.add_trajectory(state, action, reward, next_state, done)
            state = next_state

        agent.add_episode(ep_dict)
        if ep_idx >= 50 and ep_idx % 10 == 0:
            for _ in range(50):
                loss = agent.update()
                train_iter += 1
                writer.add_scalar("Loss/value", loss.item(), train_iter)

        if ep_idx >= 50:
            # for i in range(64):
            #     agent.update()
            agent.mark_episode()
        if ep_idx % 5 == 0: 
            print(f"EPISODE {ep_idx} REWARD {total_reward} epsilon {agent.epsilon} done {done} terminated {terminated} num_steps {num_steps}")
        writer.add_scalar("Episode/total_reward", total_reward, ep_idx)
        writer.flush()
    writer.close()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=100,
            help="Seed used for reproducibility.")
        parser.add_argument("--continuous", action="store_true",
            help="Indicates the RL alg. should ignore done.")
        parser.add_argument("--lr", type=float, default=3e-4,
            help="Learning rate used for updating the models.")
        parser.add_argument("--gamma", type=float, default=0.99,
            help="Gamma used for computing the return.")
        parser.add_argument("--tgt_tau", type=float, default=0.01,
            help="Tau used for smooth target updates.")
        parser.add_argument("--eps", type=float, default=0.99,
            help="Initial epsilon value used for epsilon-greedy exploration.")
        parser.add_argument("--eps_decay", type=float, default=0.998,
            help="The ammount epsilon is decayed after each episode.")
        parser.add_argument("--eps_min", type=float, default=0.05,
            help="Minimum epsilon value.")
        parser.add_argument("--replay_buffer_size", type=int, default=int(1e5),
            help="Maximum size of the replay buffer.")
        parser.add_argument("--batch_size", type=int, default=64,
            help="Batch size for training the model.")
        parser.add_argument("--n_step", type=int, default=1,
            help="Numner of steps used for multi-step learning.")
        parser.add_argument("--spr_lam", type=float, default=1.0,
            help="Lambda for SPR loss.")
        parser.add_argument("--spr_k", type=int, default=5,
            help="Number of steps for SPR loss to forecast into the future.")
        parser.add_argument("--buffer_episodes", type=int, default=20,
            help="Number of episodes to store in the episode buffer.")
        main(parser.parse_args())
