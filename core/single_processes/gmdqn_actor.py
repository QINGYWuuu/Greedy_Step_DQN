import time
import numpy as np
from collections import deque
import torch

from utils.helpers import reset_experience


def gmdqn_actor(actor_id,
                args,
                global_logs,
                actor_logs,
                env_prototype,
                model_prototype,
                global_memory,
                dqn):

    # env
    env = env_prototype(args.env_params, actor_id, args.num_envs_per_actor)
    env.train()

    local_device = torch.device('cuda')
    actor_dqn = model_prototype(args.model_params,
                                args.norm_val,
                                args.state_shape,
                                args.action_space,
                                args.action_shape).to(local_device)
    actor_dqn.load_state_dict(dqn.state_dict())
    actor_dqn.eval()

    # params
    if args.num_actors <= 1:    # NOTE: should avoid this situation, here just for debugging
        eps = 0.1
    else:                       # as described in top of Pg.6
        eps = args.agent_params.eps ** (1. + (actor_id-1)/(args.num_actors-1) * args.agent_params.eps_alpha)

    # setup
    torch.set_grad_enabled(False)
    # main control loop
    experience = reset_experience()
    # counters
    step = 0
    episode_steps = 0
    episode_reward = 0.
    total_steps = 0
    total_reward = 0.
    nepisodes = 0
    nepisodes_solved = 0
    # flags
    flag_reset = True   # True when: terminal1 | episode_steps > self.early_stop

    while global_logs.learner_step.value < args.agent_params.steps:
        if flag_reset:
            # reset episode stats
            episode_steps = 0
            episode_reward = 0.
            # reset game
            experience = env.reset()
            # reset the cache
            traj_state = []
            traj_action = []
            traj_reward = []
            assert experience.state1 is not None # state1 is the initial state
            flag_reset = False

        # run a single step
        with torch.no_grad():
            action, qvalue, max_qvalue, qvalues = actor_dqn.get_action(experience.state1, eps, device=local_device)

        traj_state.append(experience.state1)
        experience = env.step(action)
        traj_action.append(experience.action)
        traj_reward.append(experience.reward)

        if experience.terminal1:
            nepisodes_solved += 1
            flag_reset = True
        if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
            flag_reset = True

        if flag_reset:
            traj_state.append(experience.state1)
            global_memory.feed((traj_state, traj_action, traj_reward))
            print("traj was saved")
            traj_state = []
            traj_action = []
            traj_reward = []

        with global_logs.actor_step.get_lock():
            global_logs.actor_step.value += 1
        step += 1
        episode_steps += 1
        episode_reward += experience.reward
        if flag_reset:
            nepisodes += 1
            total_steps += episode_steps
            total_reward += episode_reward

        if step % args.agent_params.actor_sync_freq == 0:
            actor_dqn.load_state_dict(dqn.state_dict())

        if step % args.agent_params.actor_freq == 0: # then push local stats to logger & reset local
            with actor_logs.nepisodes.get_lock():
                actor_logs.total_steps.value += total_steps
                actor_logs.total_reward.value += total_reward
                actor_logs.nepisodes.value += nepisodes
                actor_logs.nepisodes_solved.value += nepisodes_solved
            total_steps = 0
            total_reward = 0.
            nepisodes = 0
            nepisodes_solved = 0



