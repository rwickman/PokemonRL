from pokemon_rl.utils import set_seed
import argparse, os

import pygame
import time

def main(args):
    if not os.path.exists(args.save_dir):
         os.makedirs(args.save_dir)

    # Must call set seed before importing other modules
    set_seed(args.seed)
    from torch.utils.tensorboard import SummaryWriter
    import gymnasium as gym
    import matplotlib.pyplot as plt

    from pokemon_rl.environment.cartpole_env import CartPoleImageWrapper
    from pokemon_rl.environment.lunar_env import LunarImageWrapper
    from pokemon_rl.rl.dqn_agent import DQNAgent

    #env_name = 'CartPole-v1'
    env_name = "LunarLander-v2"

    # gym_env = gym.make('LunarLander-v2', render_mode="rgb_array")
    # env = LunarImageWrapper(gym_env)
    # gym_env = gym.make('CartPole-v1', render_mode="rgb_array")
    # env = CartPoleImageWrapper(gym_env)
    # agent = DQNAgent(
    #       args,
    #       env.observation_space.shape,
    #       gym_env.action_space.n)
    suffix = "_betanstep_cv_dueling"

    env = gym.make(env_name, render_mode=args.render_mode)
    agent = DQNAgent(
          args,
          env.observation_space.shape,
          env.action_space.n)
    suffix = "_slowincbeta_actornet"

    # Reset should only be called with the seed once
    env.reset(seed=args.seed)
    if args.render_mode == "human":
        agent.train_dict["num_episodes"] = 10000000
        agent.load(args.model_file)
        # pygame.init()
        # print(env.observation_space.shape)
        # screen = pygame.display.set_mode((env.observation_space.shape[2], env.observation_space.shape[1]))
    else:
        writer = SummaryWriter(comment=f"_{env_name}_nstep_{args.n_step}_buffer_{args.replay_buffer_size}_sprlam_{args.spr_lam}_BufferEps_{args.buffer_episodes}_actor_temp_decay_steps_{args.actor_temp_decay_steps}_actor_temp_final_{args.actor_temp_final}{suffix}")
        train_iter = 0
    for ep_idx in range(args.num_episodes):
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
            #     fig, ax = plt.subplots(state.shape[0])
                
            #     for i in range(state.shape[0]):
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

            # if args.render_mode == "human":
            #     screen.fill("purple")
            #     surf = pygame.surfarray.make_surface(state[0].T * 255)
            #     screen.blit(surf, (0, 0))
            #     pygame.display.flip()
            #     time.sleep(0.01)
        if args.render_mode != "human":             
            agent.add_episode(ep_dict)
            if ep_idx >= 50 and ep_idx % 10 == 0:
                for _ in range(50):
                    loss, actor_loss, contrastive_loss = agent.update()
                    train_iter += 1
                    writer.add_scalar("Loss/value", loss.item(), train_iter)
                    writer.add_scalar("Loss/actor_loss", actor_loss.item(), train_iter)
                    writer.add_scalar("Loss/contrastive_loss", contrastive_loss.item(), train_iter)


            if ep_idx >= 50:
                # for i in range(64):
                #     agent.update()
                agent.mark_episode()

            if ep_idx % 1000 == 0:
                agent.save()

            if ep_idx % 5 == 0: 
                print(f"EPISODE {ep_idx} REWARD {total_reward} epsilon {agent.epsilon} done {done} terminated {terminated} num_steps {num_steps}")
            writer.add_scalar("Episode/total_reward", total_reward, ep_idx)
            writer.flush()
        else:
            print(f"EPISODE {ep_idx} REWARD {total_reward} epsilon {agent.epsilon} done {done} terminated {terminated} num_steps {num_steps}")

    writer.close()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=100,
            help="Seed used for reproducibility.")
        parser.add_argument("--save_dir", default="models/",
            help="Location to save the policy.")
        parser.add_argument("--render_mode", default=None,
            help="How to render the environment.")
        parser.add_argument("--model_file", default="",
            help="Location of the model to load.")

        parser.add_argument("--num_episodes", type=int, default=int(1e6),
            help="Number of episodes to run.")
        parser.add_argument("--continuous", action="store_true",
            help="Indicates the RL alg. should ignore done.")
        parser.add_argument("--hidden_dim", type=int, default=64,
            help="Number of parameters in the hidden dimension.")
        parser.add_argument("--n_layers", type=int, default=2,
            help="Number of layers.")

        parser.add_argument("--lr", type=float, default=3e-4,
            help="Learning rate used for updating the models.")
        parser.add_argument("--actor_lr", type=float, default=1e-4,
            help="Learning rate used for updating the models.")
        parser.add_argument("--actor_temp_init", type=float, default=2.0,
            help="Initial temperature used for control.")
        parser.add_argument("--actor_temp_final", type=float, default=0.01,
            help="Final temperature used for control.")
        parser.add_argument("--actor_temp_decay_steps", type=int, default=10000,
            help="Number of steps before the temperature decays to final")
        parser.add_argument("--contrastive_lam", type=float, default=0.01,
            help="Loss use for contrastive loss.")

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
        parser.add_argument("--replay_buffer_size", type=int, default=int(5e3),
            help="Maximum size of the replay buffer.")
        parser.add_argument("--batch_size", type=int, default=64,
            help="Batch size for training the model.")
        parser.add_argument("--n_step", type=int, default=1,
            help="Numner of steps used for multi-step learning.")
        parser.add_argument("--spr_lam", type=float, default=0.0,
            help="Lambda for SPR loss.")
        parser.add_argument("--spr_k", type=int, default=1,
            help="Number of steps for SPR loss to forecast into the future.")
        parser.add_argument("--buffer_episodes", type=int, default=30,
            help="Number of episodes to store in the episode buffer.")
        main(parser.parse_args())
