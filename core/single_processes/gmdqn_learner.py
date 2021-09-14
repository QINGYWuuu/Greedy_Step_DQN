import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from utils.helpers import update_target_model
from utils.helpers import ensure_global_grads


# def gmdqn_learner(process_ind, args,
#                 global_logs,
#                 learner_logs,
#                 model_prototype,
#                 global_memory,
#                 global_model,
#                 global_optimizer):
#     # logs
#     print("---------------------------->", process_ind, "learner")
#     # env
#     # memory
#     # model
#     local_device = torch.device('cuda:' + str(args.gpu_ind)) # local device compute target only
#     # local_device = torch.device('cpu') # local device compute target only
#     # global_device = torch.device('cuda:' + str(args.gpu_ind)) # global device contain the computed target and predict
#     local_target_model = model_prototype(args.model_params,
#                                          args.norm_val,
#                                          args.state_shape,
#                                          args.action_space,
#                                          args.action_shape).to(local_device)
#     # # sync global model to local
#     update_target_model(global_model, local_target_model) # do a hard update in the beginning
#     # optimizer Adam
#     local_optimizer = args.agent_params.optim(global_model.parameters(),
#                                               lr=args.agent_params.lr,
#                                               weight_decay=args.agent_params.weight_decay)
#     # params
#     gamma = args.agent_params.gamma
#     # setup
#     global_model.train()
#     torch.set_grad_enabled(True)
#
#     # main control loop
#     step = 0
#     traj_cache_num = 0 # this is the count which record the sample number situation, and it's range is [0, +inf]
#     # if traj_cache_num >= batch_size  sample one batch to update, and traj_cache_num = traj_cache_num - batch_size
#     # if traj_cache_num < batch_size sample another trajactory add to the cache, and traj_cache_num = traj_cache_num + batch_size
#
#     while global_logs.learner_step.value < args.agent_params.steps:
#         if global_memory.size > args.agent_params.learn_start: # the memory buffer size is satisty the requirement of training
#
#             batch_target_cache = torch.zeros([args.agent_params.batch_size]).cuda(non_blocking=True).to(local_device)  # store the computed target
#             # batch_predict_cache = torch.zeros([args.agent_params.batch_size]).cuda(non_blocking=True).to(local_device)  # store the computed predict
#             batch_state_cache = torch.zeros((args.agent_params.batch_size,) + tuple( args.state_shape)).cuda(non_blocking=True).to(local_device)  # store the state for the predict Q(s,a)
#             batch_action_cache = torch.zeros([args.agent_params.batch_size]).cuda(non_blocking=True).to(local_device)  # store the action for the predict Q(s,a)
#             local_optimizer.zero_grad()
#
#             for sample_id in range(args.agent_params.batch_size):
#                 traj_state, traj_action, traj_reward = global_memory.sample(1)
#
#                 traj_state = traj_state.cuda(non_blocking=True).to(local_device)
#                 traj_reward = traj_reward.cuda(non_blocking=True).to(local_device)
#
#                 with torch.no_grad():
#                         # gamma_vector [1, gamma, gamma^2, ...]
#                     gamma_vector = torch.tensor(np.array([pow(gamma, i) for i in range(len(traj_reward))])).cuda(non_blocking=True).to(local_device)
#                     discounted_reward = gamma_vector * traj_reward.squeeze()
#                     up_tri_matrix = torch.tensor(np.triu(np.ones(len(traj_reward)))).cuda(non_blocking=True).to(local_device)
#                         # [r_0, r_0 + gamma * r_1, r_0 + gamma * r_1 + gamma^2 * r_2, ...]
#                     discounted_reward = torch.matmul(discounted_reward, up_tri_matrix)
#                     target_q_values = local_target_model(traj_state[1:]).max(dim=1)[0]
#                     target_q_values[-1] = 0
#                     target_q_values = discounted_reward + gamma * gamma_vector * target_q_values
#
#                     batch_target_cache[sample_id] = (target_q_values.max(dim=0)[0])
#
#                     batch_state_cache[sample_id] = traj_state[0]
#                     batch_action_cache[sample_id] = traj_action[0]
#
#             batch_target_cache = batch_target_cache.unsqueeze(dim=1)
#             batch_predict_cache = global_model(batch_state_cache).gather(1, batch_action_cache.long().unsqueeze(dim=1)).cuda(non_blocking=True).to(local_device)
#
#             critic_loss = args.agent_params.value_criteria(batch_predict_cache, batch_target_cache)
#             critic_loss.backward()
#             nn.utils.clip_grad_value_(global_model.parameters(), args.agent_params.clip_grad)
#             local_optimizer.step()
#             update_target_model(global_model, local_target_model, args.agent_params.target_model_update, step)
#
#             with global_logs.learner_step.get_lock():
#                 global_logs.learner_step.value += 1
#                 print("learners update times = {}, loss = {}".format(global_logs.learner_step.value, critic_loss.item()))
#             step += 1
#
#                 # report stats
#             if step % args.agent_params.learner_freq == 0: # then push local stats to logger & reset local
#                 learner_logs.critic_loss.value += critic_loss.item()
#                 learner_logs.loss_counter.value += 1
#
#         else: # wait for memory_size to be larger than learn_start
#             time.sleep(1)

def gmdqn_learner(learner_id,
                  args,
                  global_logs,
                  learner_logs,
                  model_prototype,
                  global_memory,
                  dqn):

    local_device = torch.device('cuda')

    target_dqn = model_prototype(args.model_params,
                                 args.norm_val,
                                 args.state_shape,
                                 args.action_space,
                                 args.action_shape).to(local_device)
    target_dqn.load_state_dict(dqn.state_dict())

    dqn.train()
    optimizer = args.agent_params.optim(dqn.parameters(),
                                        lr=args.agent_params.lr,
                                        weight_decay=args.agent_params.weight_decay)

    gamma = args.agent_params.gamma
    torch.set_grad_enabled(True)
    step = 0
    traj_cache_num = 0 # this is the count which record the sample number situation, and it's range is [0, +inf]
    # if traj_cache_num >= batch_size  sample one batch to update, and traj_cache_num = traj_cache_num - batch_size
    # if traj_cache_num < batch_size sample another trajactory add to the cache, and traj_cache_num = traj_cache_num + batch_size
    traj_target_cache = []
    traj_state_cache = []
    traj_action_cache = []
    Q_VALUES = []

    while global_logs.learner_step.value <= args.agent_params.steps:
        if global_memory.size > args.agent_params.learn_start: # the memory buffer size is satisty the requirement of training
            if traj_cache_num < args.agent_params.batch_size:
                for sample_times in range(1):
                    with torch.no_grad():
                        traj_state, traj_action, traj_reward = global_memory.sample(1)
                        traj_state = traj_state.cuda(non_blocking=True).to(local_device)
                        traj_len = len(traj_reward)
                        with torch.no_grad():
                            traj_target_qvalues = target_dqn(traj_state[1:])
                        if args.enable_double:
                            double_dqn_action = dqn(traj_state[1:]).max(dim=1)[1].unsqueeze(dim=1)
                            traj_target_q = gamma * traj_target_qvalues.gather(1, double_dqn_action).squeeze() # double_dqn
                        else:
                            traj_target_q = gamma * traj_target_qvalues.max(dim=1)[0]
                        traj_target_q[-1] = 0

                        traj_target_q = torch.eye(traj_len).cuda()*traj_target_q.unsqueeze(dim=1) + traj_reward.cuda()*torch.tensor(np.triu(np.ones(traj_len))).cuda()
                        for row in range(traj_len-2, -1, -1):
                            traj_target_q[row] = traj_target_q[row] + gamma * traj_target_q[row + 1]
                            traj_target_q[row + 1][:row + 1] = -np.inf
                        traj_target_q = traj_target_q.max(dim=1)[0].unsqueeze(dim=1)

                    if traj_target_cache == [] or traj_state_cache == [] or traj_action_cache == []:
                        traj_target_cache = traj_target_q
                        traj_state_cache = traj_state[:-1]
                        traj_action_cache = traj_action
                        traj_cache_num += len(traj_action)
                    else:
                        traj_target_cache = torch.cat((traj_target_cache, traj_target_q))
                        traj_state_cache = torch.cat((traj_state_cache, traj_state[:-1]))
                        traj_action_cache = torch.cat((traj_action_cache, traj_action))
                        traj_cache_num += len(traj_action)

            elif traj_cache_num >= args.agent_params.batch_size:
                optimizer.zero_grad()
                sample_index = random.sample(list(range(traj_cache_num)), args.agent_params.batch_size)
                sample_index.sort(reverse=True)
                remain_index = list(set(list(range(traj_cache_num)))^set(sample_index))

                sample_index = torch.tensor(sample_index).cuda()

                batch_target = torch.index_select(traj_target_cache, 0, sample_index)
                batch_state = torch.index_select(traj_state_cache, 0, sample_index)
                batch_action = torch.index_select(traj_action_cache, 0, sample_index.cpu())
                if remain_index == []:
                    traj_target_cache = []
                    traj_state_cache = []
                    traj_action_cache = []
                    traj_cache_num = 0
                else:
                    remain_index = torch.tensor(remain_index).cuda()
                    traj_target_cache = torch.index_select(traj_target_cache, 0, remain_index)
                    traj_state_cache = torch.index_select(traj_state_cache, 0, remain_index)

                    traj_action_cache = torch.index_select(traj_action_cache, 0, remain_index.cpu())
                    traj_cache_num -= args.agent_params.batch_size

                batch_predict = dqn(batch_state).gather(1, batch_action.long().cuda()).cuda(non_blocking=True).to(local_device)
                if step % 5000 == 0:
                    Q_VALUES.append([step, batch_predict.mean().detach().cpu()])
                    df = pd.DataFrame(data=Q_VALUES)
                    df.to_csv(args.Q_value_csv_file)

                critic_loss = args.agent_params.value_criteria(batch_predict, batch_target.float())
                critic_loss.backward()
                nn.utils.clip_grad_value_(dqn.parameters(), args.agent_params.clip_grad)
                optimizer.step()
                update_target_model(dqn, target_dqn, args.agent_params.target_model_update, step)

                with global_logs.learner_step.get_lock():
                    global_logs.learner_step.value += 1
                step += 1

                if step % args.agent_params.learner_freq == 0: # then push local stats to logger & reset local
                    learner_logs.critic_loss.value += critic_loss.item()
                    learner_logs.loss_counter.value += 1

        else: # wait for memory_size to be larger than learn_start
            time.sleep(1)