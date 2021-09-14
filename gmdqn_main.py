import os
import torch
import torch.multiprocessing as mp

from utils.options_gmdqn import Options
from utils.factory import GlobalLogsDict, ActorLogsDict, LearnerLogsDict, EvaluatorLogsDict
from utils.factory import LoggersDict, ActorsDict, LearnersDict, EvaluatorsDict, TestersDict
from utils.factory import EnvsDict, MemoriesDict, ModelsDict


def greedy_step_dqn(config, gpu_ind, dqn_num, random_seed):
    mp.set_start_method("spawn", force=True)

    opt = Options(config=config, gpu_ind=gpu_ind, dqn_num=dqn_num, random_seed=random_seed)
    torch.manual_seed(opt.seed)

    env_prototype = EnvsDict[opt.env_type]
    memory_prototype = MemoriesDict[opt.memory_type]
    model_prototype = ModelsDict[opt.model_type]

    # dummy env to get state/action/reward/gamma/terminal_shape & action_space
    dummy_env = env_prototype(opt.env_params, 0)
    opt.norm_val = dummy_env.norm_val # use the max val of env states to normalize model inputs
    opt.state_shape = dummy_env.state_shape # state shape
    opt.action_shape = dummy_env.action_shape # action shape
    opt.action_space = dummy_env.action_space
    opt.reward_shape = opt.agent_params.num_tasks
    opt.gamma_shape = opt.agent_params.num_tasks
    opt.terminal_shape = opt.agent_params.num_tasks
    del dummy_env
    opt.maxmin_dqn_num = opt.maxmin_num
    print("maxmin_{}".format(opt.maxmin_num))
    # DEBUG FOR THE

    processes = []
    if opt.mode == 1:
        # shared memory
        opt.memory_params.norm_val = opt.norm_val
        opt.memory_params.state_shape = opt.state_shape
        opt.memory_params.action_shape = opt.action_shape
        opt.memory_params.reward_shape = opt.reward_shape
        opt.memory_params.gamma_shape = opt.gamma_shape
        opt.memory_params.terminal_shape = opt.terminal_shape
        global_memory = memory_prototype(opt.memory_params) # create shared replay buffer # TODO the traj framework

        # shared model #todo the maxmin dqn
        maxmin_dqns = {}
        for dqn_id in range(opt.maxmin_dqn_num):
            dqn = model_prototype(opt.model_params, opt.norm_val, opt.state_shape, opt.action_space, opt.action_shape) # create network
            dqn.to(torch.device('cuda'))
            maxmin_dqns.update({dqn_id: dqn})

        global_optimizers = None#opt.agent_params.optim(global_model.parameters())

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
                                 maxmin_dqns
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
                                 maxmin_dqns,
                                 global_optimizers,
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
                             maxmin_dqns
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


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if __name__ == '__main__':
    for config in range(25,-1,-1):
    # 24, 32, 52, 58
    # for config in [52]:
        greedy_step_dqn(config=config, gpu_ind=0, dqn_num=1, random_seed=101)