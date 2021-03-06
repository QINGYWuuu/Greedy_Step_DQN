import numpy as np
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils.options_alterdqn import Options
from utils.factory import GlobalLogsDict, ActorLogsDict, LearnerLogsDict, EvaluatorLogsDict
from utils.factory import LoggersDict, ActorsDict, LearnersDict, EvaluatorsDict, TestersDict
from utils.factory import EnvsDict, MemoriesDict, ModelsDict

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    opt = Options()
    torch.manual_seed(opt.seed)

    env_prototype = EnvsDict[opt.env_type]
    memory_prototype = MemoriesDict[opt.memory_type]
    model_prototype = ModelsDict[opt.model_type] # dqn-cnn

    # dummy env to get state/action/reward/gamma/terminal_shape & action_space
    dummy_env = env_prototype(opt.env_params, 0)
    opt.norm_val = dummy_env.norm_val # use the max val of env states to normalize model inputs
    opt.state_shape = dummy_env.state_shape # state shape
    opt.action_shape = dummy_env.action_shape # action shape
    opt.action_space = dummy_env.action_space
    opt.reward_shape = opt.agent_params.num_tasks
    opt.gamma_shape = opt.agent_params.num_tasks
    opt.terminal_shape = opt.agent_params.num_tasks
    opt.dqn_num = opt.agent_params.dqn_num
    del dummy_env

    processes = []
    if opt.mode == 1:
        # shared memory
        opt.memory_params.norm_val = opt.norm_val
        opt.memory_params.state_shape = opt.state_shape
        opt.memory_params.action_shape = opt.action_shape
        opt.memory_params.reward_shape = opt.reward_shape
        opt.memory_params.gamma_shape = opt.gamma_shape
        opt.memory_params.terminal_shape = opt.terminal_shape
        global_memory = memory_prototype(opt.memory_params) # create shared replay buffer

        # shared model
        # the alter dqns include opt.dqn_num model
        global_models = {}
        for dqn_id in range(opt.dqn_num):
            global_model = model_prototype(opt.model_params, opt.norm_val, opt.state_shape, opt.action_space,
                                           opt.action_shape)  # create network
            if opt.model_file is not None: global_model.load_state_dict(
                torch.load(opt.model_file))  # this means finetuning on model_file
            global_model.to(torch.device('cuda'))  # TODO: set w/ args
            global_models.update({dqn_id: global_model})

        # optimizer
        global_optimizer = None#opt.agent_params.optim(global_model.parameters())

        # logs
        global_logs = GlobalLogsDict[opt.agent_type]()
        actor_logs = ActorLogsDict[opt.agent_type]()
        learner_logs = LearnerLogsDict[opt.agent_type]()
        evaluator_logs = EvaluatorLogsDict[opt.agent_type]()

        # logger
        logger_fn = LoggersDict[opt.agent_type]
        p = mp.Process(target=logger_fn,
                       args=(0, opt,
                             global_logs,
                             actor_logs,
                             learner_logs,
                             evaluator_logs
                            ))
        p.start()
        processes.append(p)

        # actor
        actor_fn = ActorsDict[opt.agent_type]
        for process_ind in range(opt.num_actors):
            p = mp.Process(target=actor_fn,
                           args=(process_ind+1, opt,
                                 global_logs,
                                 actor_logs,
                                 env_prototype,
                                 model_prototype,
                                 global_memory,
                                 global_model
                                ))
            p.start()
            processes.append(p)
        # learner
        learner_fn = LearnersDict[opt.agent_type]
        for process_ind in range(opt.num_learners):
            p = mp.Process(target=learner_fn,
                           args=(opt.num_actors+process_ind+1, opt,
                                 global_logs,
                                 learner_logs,
                                 model_prototype,
                                 global_memory,
                                 global_model,
                                 global_optimizer
                                ))
            p.start()
            processes.append(p)

        # evaluator
        evaluator_fn = EvaluatorsDict[opt.agent_type]
        p = mp.Process(target=evaluator_fn,
                       args=(opt.num_actors+opt.num_learners+1, opt,
                             global_logs,
                             evaluator_logs,
                             env_prototype,
                             model_prototype,
                             global_model
                            ))
        p.start()
        processes.append(p)
    elif opt.mode == 2:
        # tester
        tester_fn = TestersDict[opt.agent_type]
        p = mp.Process(target=tester_fn,
                       args=(opt.num_actors+opt.num_learners+2, opt,
                             env_prototype,
                             model_prototype))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
