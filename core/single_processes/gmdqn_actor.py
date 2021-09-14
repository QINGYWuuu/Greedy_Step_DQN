import time
import numpy as np
from collections import deque
import torch

from utils.helpers import reset_experience


def gmdqn_actor(process_ind, args,
              global_logs,
              actor_logs,
              env_prototype,
              model_prototype,
              global_memory,
              maxmin_dqns):
    # logs
    print("---------------------------->", process_ind, "actor")

    # env
    env = env_prototype(args.env_params, process_ind, args.num_envs_per_actor)
    env.train()
    # memory
    # model # initialize the local model based on the global model
    local_device = torch.device('cuda:' + str(args.gpu_ind))#('cpu')
    local_target_maxmin_dqns = {}
    print("actor create {} nets".format(args.maxmin_dqn_num))
    for dqn_id in range(args.maxmin_dqn_num):
        local_target_maxmin_dqn = model_prototype(args.model_params,
                                         args.norm_val,
                                         args.state_shape,
                                         args.action_space,
                                         args.action_shape).to(local_device)
        local_target_maxmin_dqn.load_state_dict(maxmin_dqns[dqn_id].state_dict())
        local_target_maxmin_dqns.update({dqn_id: local_target_maxmin_dqn})
        local_target_maxmin_dqns[dqn_id].eval()

    # params
    if args.num_actors <= 1:    # NOTE: should avoid this situation, here just for debugging
        eps = 0.1
    else:                       # as described in top of Pg.6
        eps = args.agent_params.eps ** (1. + (process_ind-1)/(args.num_actors-1) * args.agent_params.eps_alpha)
    # eps = 0.
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

    while global_logs.learner_step.value < args.agent_params.steps: # the learner step is less than the maxstep
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
            # flags
            flag_reset = False

        # run a single step
        maxmin_dqn_qvalues = torch.zeros(args.maxmin_dqn_num, args.action_shape, args.action_space)
        for dqn_id in range(args.maxmin_dqn_num):
            action, qvalue, max_qvalue, qvalues = maxmin_dqns[dqn_id].get_action(experience.state1, args.memory_params.enable_per, eps, device=local_device)
            if qvalues == None:
                break
            maxmin_dqn_qvalues[dqn_id] = qvalues

        if qvalues != None:
            action = np.array([maxmin_dqn_qvalues.min(dim=0)[0].max(dim=1)[1].__array__()])
        traj_state.append(experience.state1)
        experience = env.step(action)
        traj_action.append(experience.action)
        traj_reward.append(experience.reward)

        if experience.terminal1:
            nepisodes_solved += 1
            flag_reset = True

        if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
            flag_reset = True

        if flag_reset: #feed the traj to the global memory todo the method of feed the traj
            traj_state.append(experience.state1)
            global_memory.feed((traj_state, traj_action, traj_reward))

            traj_state = []
            traj_action = []
            traj_reward = []

            # update counters & stats
        with global_logs.actor_step.get_lock():
            global_logs.actor_step.value += 1
        step += 1
        episode_steps += 1
        episode_reward += experience.reward
        if flag_reset:
            nepisodes += 1
            total_steps += episode_steps
            total_reward += episode_reward

            # sync global model to local
        if step % args.agent_params.actor_sync_freq == 0:
            for dqn_id in range(args.maxmin_dqn_num):
                local_target_maxmin_dqns[dqn_id].load_state_dict(maxmin_dqns[dqn_id].state_dict())

            # report stats
        if step % args.agent_params.actor_freq == 0: # then push local stats to logger & reset local
            # push local stats to logger
            with actor_logs.nepisodes.get_lock():
                actor_logs.total_steps.value += total_steps
                actor_logs.total_reward.value += total_reward
                actor_logs.nepisodes.value += nepisodes
                actor_logs.nepisodes_solved.value += nepisodes_solved
                # reset local stats
            total_steps = 0
            total_reward = 0.
            nepisodes = 0
            nepisodes_solved = 0



