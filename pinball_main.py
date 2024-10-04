from pokemon_rl.utils import set_seed
import argparse, os


EP_BEFORE_TRAIN = 40

def update(agent, writer, train_iter):
    loss, actor_loss, contrastive_loss, did_reset = agent.update()

    writer.add_scalar("Loss/value", loss.item(), train_iter)
    writer.add_scalar("Loss/actor_loss", actor_loss.item(), train_iter)
    writer.add_scalar("Loss/contrastive_loss", contrastive_loss.item(), train_iter)
    return did_reset

def main(args):
    if not os.path.exists(args.save_dir):
         os.makedirs(args.save_dir)

    # Must call set seed before importing other modules
    set_seed(args.seed)
    from torch.utils.tensorboard import SummaryWriter
    import gymnasium as gym
    from pokemon_rl.rl.dqn_agent import DQNAgent
    from pokemon_rl.environment.pokemon_pinball import PinballPyBoyEnv
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    env = PinballPyBoyEnv(os.path.join(cur_dir, 'ROMs/Pokemon Pinball (U) [C][!].gbc'), debug=args.render_mode == "human", do_render=args.render_mode == "human")
    agent = DQNAgent(
          args,
          env.observation_space.shape,
          env.action_space.n)
    suffix = ""

    # Reset should only be called with the seed once
    env.reset(seed=args.seed)
    if args.render_mode == "human":
        agent.train_dict["num_episodes"] = 10000000
        agent.load(args.model_file)
    else:
        writer = SummaryWriter(comment=f"__nstep_{args.n_step}_buffer_{args.replay_buffer_size}_BufferEps_{args.buffer_episodes}_tau_{args.tgt_tau}_qvaltemp_{args.q_val_dist_temp}_contrastive_lam_{args.contrastive_lam}_gamma_{args.gamma}{suffix}")
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
            if ep_idx >= EP_BEFORE_TRAIN or args.render_mode != None:
                action = agent(state)
            else:
                action = env.action_space.sample()

            next_state, reward, done, terminated, _ = env.step(action)
            ep_dict["states"].append(state)
            ep_dict["next_states"].append(next_state)
            ep_dict["rewards"].append(reward)
            ep_dict["actions"].append(action)
            ep_dict["dones"].append(done)

            total_reward += reward
            agent.add_trajectory(state, action, reward, next_state, done)
            state = next_state.copy()
            # print(state)

            if ep_idx >= EP_BEFORE_TRAIN and num_steps % 32 == 0 and args.render_mode != "human":
                train_iter += 1
                did_reset = update(agent, writer, train_iter)
                if did_reset:
                    for _ in range(args.ramp_up_updates):
                        train_iter += 1
                        update(agent, writer, train_iter)

            num_steps += 1

        if args.render_mode != "human":             
            agent.add_episode(ep_dict)

            if ep_idx >= EP_BEFORE_TRAIN:
                agent.mark_episode()

            if ep_idx % 25 == 0:
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
        parser.add_argument("--num_episodes", type=int, default=int(1e8),
            help="Number of episodes to run.")
        parser.add_argument("--continuous", action="store_true",
            help="Indicates the RL alg. should ignore done.")
        parser.add_argument("--hidden_dim", type=int, default=128,
            help="Number of parameters in the hidden dimension.")
        parser.add_argument("--n_layers", type=int, default=3,
            help="Number of hidden layers.")
        
        parser.add_argument("--lr", type=float, default=3e-4,
            help="Learning rate used for updating the models.")
        parser.add_argument("--actor_lr", type=float, default=1e-4,
            help="Learning rate used for updating the models.")
        parser.add_argument("--gamma", type=float, default=0.997,
            help="Gamma used for computing the return.")
        parser.add_argument("--tgt_tau", type=float, default=0.001,
            help="Tau used for smooth target updates.")
        parser.add_argument("--eps", type=float, default=0.99,
            help="Initial epsilon value used for epsilon-greedy exploration.")
        parser.add_argument("--eps_decay", type=float, default=0.9995,
            help="The ammount epsilon is decayed after each episode.")
        parser.add_argument("--eps_min", type=float, default=0.05,
            help="Minimum epsilon value.")
        parser.add_argument("--replay_buffer_size", type=int, default=int(1e6),
            help="Maximum size of the replay buffer.")
        parser.add_argument("--batch_size", type=int, default=128,
            help="Batch size for training the model.")
        parser.add_argument("--n_step", type=int, default=16,
            help="Numner of steps used for multi-step learning.")
        parser.add_argument("--spr_lam", type=float, default=0.0,
            help="Lambda for SPR loss.")
        parser.add_argument("--spr_k", type=int, default=1,
            help="Number of steps for SPR loss to forecast into the future.")
        parser.add_argument("--buffer_episodes", type=int, default=30,
            help="Number of episodes to store in the episode buffer.")
        parser.add_argument("--actor_temp_init", type=float, default=1.0,
            help="Initial temperature used for control.")
        parser.add_argument("--actor_temp_final", type=float, default=0.01,
            help="Final temperature used for control.")

        parser.add_argument("--contrastive_lam", type=float, default=0.1,
            help="Loss use for contrastive loss.")
        parser.add_argument("--actor_temp_decay_steps", type=int, default=200000,
            help="Number of steps before the temperature decays to final")
        parser.add_argument("--q_val_dist_temp", type=float, default=0.2,
            help="Temperature used for the target q-value distribution.")

        parser.add_argument("--network_reset_step", type=int, default=200000,
            help="Number of updates before resetting the networks.")
        parser.add_argument("--ramp_up_updates", type=int, default=200,
            help="Number of network updates to perform after resetting to ramp up.")
        parser.add_argument("--reset_tgt_tau", type=float, default=0.5,
            help="Percentage of parameters from target network to preserve.")
        parser.add_argument("--reset_temp_offset", type=float, default=0.3,
            help="Add offset to actor temperature to explore more after resetting.")
        parser.add_argument("--knn_k", type=int, default=8,
            help="KNN k for APT.")
        parser.add_argument("--knn_clip", type=float, default=0.0005,
            help="KNN Clip value.")
        parser.add_argument("--apt_reward_lam", type=float, default=0.1,
            help="Scaling on the APT reward.")
        main(parser.parse_args())

