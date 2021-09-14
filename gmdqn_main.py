import os
import torch
import torch.multiprocessing as mp

from utils.options_gmdqn import Options
from utils.factory import GlobalLogsDict, ActorLogsDict, LearnerLogsDict, EvaluatorLogsDict
from utils.factory import LoggersDict, ActorsDict, LearnersDict, EvaluatorsDict, TestersDict
from utils.factory import EnvsDict, MemoriesDict, ModelsDict


def greedy_step_dqn(config, random_seed, num_actors):
    mp.set_start_method("spawn", force=True)
    args = Options(config=config,random_seed=random_seed, num_actors=num_actors)



    torch.manual_seed(args.seed)
    env_prototype = EnvsDict[args.env_type]
    memory_prototype = MemoriesDict[args.memory_type]
    model_prototype = ModelsDict[args.model_type]

    # dummy env to get state/action/reward/gamma/terminal_shape & action_space
    dummy_env = env_prototype(args.env_params, 0)
    args.norm_val = dummy_env.norm_val # use the max val of env states to normalize model inputs
    args.state_shape = dummy_env.state_shape # state shape
    args.action_shape = dummy_env.action_shape # action shape
    args.action_space = dummy_env.action_space
    args.reward_shape = args.agent_params.num_tasks
    args.gamma_shape = args.agent_params.num_tasks
    args.terminal_shape = args.agent_params.num_tasks
    del dummy_env

    processes = []
    if args.mode == 1:
        # shared memory
        args.memory_params.norm_val = args.norm_val
        args.memory_params.state_shape = args.state_shape
        args.memory_params.action_shape = args.action_shape
        args.memory_params.reward_shape = args.reward_shape
        args.memory_params.gamma_shape = args.gamma_shape
        args.memory_params.terminal_shape = args.terminal_shape
        global_memory = memory_prototype(args.memory_params) # create shared replay buffer # TODO the traj framework

        dqn = model_prototype(args.model_params, args.norm_val, args.state_shape, args.action_space, args.action_shape)
        dqn.to(torch.device('cuda'))

        global_logs = GlobalLogsDict[args.agent_type]()
        actor_logs = ActorLogsDict[args.agent_type]()
        learner_logs = LearnerLogsDict[args.agent_type]()
        evaluator_logs = EvaluatorLogsDict[args.agent_type]()

        logger_fn = LoggersDict[args.agent_type]
        p = mp.Process(target=logger_fn,
                       args=(args,
                             global_logs,
                             actor_logs,
                             learner_logs,
                             evaluator_logs
                            ))
        p.start()
        processes.append(p)

        # actor
        actor_fn = ActorsDict[args.agent_type]
        for actor_id in range(args.num_actors):
            p = mp.Process(target=actor_fn,
                           args=(actor_id,
                                 args,
                                 global_logs,
                                 actor_logs,
                                 env_prototype,
                                 model_prototype,
                                 global_memory,
                                 dqn
                                ))
            p.start()
            processes.append(p)

        # learner
        learner_fn = LearnersDict[args.agent_type]
        for learner_id in range(args.num_learners):
            p = mp.Process(target=learner_fn,
                           args=(learner_id,
                                 args,
                                 global_logs,
                                 learner_logs,
                                 model_prototype,
                                 global_memory,
                                 dqn
                                ))
            p.start()
            processes.append(p)

        # evaluator
        evaluator_fn = EvaluatorsDict[args.agent_type]
        p = mp.Process(target=evaluator_fn,
                       args=(args,
                             global_logs,
                             evaluator_logs,
                             env_prototype,
                             model_prototype,
                             dqn
                            ))
        p.start()
        processes.append(p)

    elif args.mode == 2:
        # tester
        tester_fn = TestersDict[args.agent_type]
        p = mp.Process(target=tester_fn,
                       args=(args.num_actors+args.num_learners+2, args,
                             env_prototype,
                             model_prototype))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    # for config in range(26):
    for config in [20]:
        greedy_step_dqn(config=config, random_seed=1, num_actors=1)