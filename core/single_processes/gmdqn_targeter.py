import time
import torch
import torch.nn as nn
import numpy as np
from utils.helpers import update_target_model
from utils.helpers import ensure_global_grads


def gmdqn_learner(process_ind, args,
                  global_logs,
                  learner_logs,
                  model_prototype,
                  global_memory,
                  global_model,
                  global_optimizer):
    # logs
    print("---------------------------->", process_ind, "learner")
    # env
    # memory
    # model
    local_device = torch.device('cuda:' + str(args.gpu_ind))
    global_device = torch.device('cuda:' + str(args.gpu_ind))
    local_target_model = model_prototype(args.model_params,
                                         args.norm_val,
                                         args.state_shape,
                                         args.action_space,
                                         args.action_shape).to(local_device)
    # # sync global model to local
    update_target_model(global_model, local_target_model)  # do a hard update in the beginning
    # optimizer Adam
    local_optimizer = args.agent_params.optim(global_model.parameters(),
                                              lr=args.agent_params.lr,
                                              weight_decay=args.agent_params.weight_decay)
    # params
    # setup
    # local_model.train()
    global_model.train()
    torch.set_grad_enabled(True)

    # main control loop
    step = 0
    while global_logs.learner_step.value < args.agent_params.steps:
        if global_memory.size > args.agent_params.learn_start:  # the memory buffer size is satisty the requirement of training
            # sample batch from global_memory
            trajactory = global_memory.sample(args.agent_params.batch_size)  # return a dict
            batch_traj_nstep, batch_traj_state, batch_traj_action, batch_traj_reward = trajactory  # todo compute the target and predict
            gamma = args.agent_params.gamma

            predict_q_values = torch.zeros([args.agent_params.batch_size]).cuda(non_blocking=True).to(local_device)
            target_q_values = torch.zeros([args.agent_params.batch_size]).cuda(non_blocking=True).to(local_device)
            for batch_ind in range(args.agent_params.batch_size):
                nstep = batch_traj_nstep[batch_ind]
                traj_state = batch_traj_state[batch_ind][nstep:]
                traj_action = batch_traj_action[batch_ind][nstep:]
                traj_reward = batch_traj_reward[batch_ind][nstep:]

                # move to cuda
                traj_state = traj_state.cuda(non_blocking=True).to(local_device)
                traj_reward = traj_reward.cuda(non_blocking=True).to(local_device)

                # gamma
                gamma_vector = torch.tensor(np.array([pow(gamma, i) for i in range(len(traj_state) - 1)]))
                gamma_vector = gamma_vector.cuda(non_blocking=True).to(local_device)

                discounted_reward = gamma_vector * traj_reward
                up_tri_matrix = torch.tensor(np.triu(np.ones(len(traj_state) - 1)))
                up_tri_matrix = up_tri_matrix.cuda(non_blocking=True).to(local_device)
                discounted_culmulated_reward = torch.matmul(discounted_reward, up_tri_matrix)
                # traget value
                max_target_q = local_target_model(traj_state[1:]).max(dim=1)[0]
                max_target_q[-1] = 0  # terminal state set zero
                max_discounted_target_q = gamma * gamma_vector * max_target_q
                nstep_target = discounted_culmulated_reward + max_discounted_target_q
                # max nstep target
                max_nstep_target = nstep_target.max()
                target_q_values[batch_ind] = max_nstep_target
                # predict Q value
                predict_value = global_model(traj_state[0].unsqueeze(dim=0))[0][traj_action[0].int()[0]]
                predict_q_values[batch_ind] = predict_value

            # update counters
            with global_logs.learner_step.get_lock():

                local_optimizer.zero_grad()
                critic_loss = args.agent_params.value_criteria(predict_q_values, target_q_values)
                critic_loss.backward()
                nn.utils.clip_grad_value_(global_model.parameters(), args.agent_params.clip_grad)
                local_optimizer.step()
                update_target_model(global_model, local_target_model, args.agent_params.target_model_update, step)
                print("learner {} update".format(process_ind))

                global_logs.learner_step.value += 1
            step += 1

            # report stats
            if step % args.agent_params.learner_freq == 0:  # then push local stats to logger & reset local
                learner_logs.critic_loss.value += critic_loss.item()
                learner_logs.loss_counter.value += 1

        else:  # wait for memory_size to be larger than learn_start
            time.sleep(1)
