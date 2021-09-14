import time
import numpy as np
from collections import deque
import torch
import pandas as pd
from utils.helpers import reset_experience


def evaluator(process_ind, args,
              global_logs,
              evaluator_logs,
              env_prototype,
              model_prototype,
              maxmin_dqns):
    # logs
    print("---------------------------->", process_ind, "evaluator")
    # env
    env = env_prototype(args.env_params, process_ind)
    env.eval()
    # memory
    # model
    local_device = torch.device('cuda')#('cpu')
    local_maxmin_dqns = {}
    for dqn_id in range(args.maxmin_dqn_num):
        local_maxmin_dqn = model_prototype(args.model_params,
                                                  args.norm_val,
                                                  args.state_shape,
                                                  args.action_space,
                                                  args.action_shape).to(local_device)
        local_maxmin_dqn.load_state_dict(maxmin_dqns[dqn_id].state_dict())
        local_maxmin_dqns.update({dqn_id: local_maxmin_dqn})
        local_maxmin_dqns[dqn_id].eval()

    torch.set_grad_enabled(False)

    last_eval_learner_step = 0
    print("csv log file save as {}".format(args.save_log_csv))
    eval_csv_logs = []

    while global_logs.learner_step.value < args.agent_params.steps:

        if global_logs.learner_step.value % 10000 == 0 and global_logs.learner_step.value > last_eval_learner_step: #evaluate once every train 1000 times

            eval_record = []  # [step, eva 1, eva 2, ..., eva n, eva aver]
            last_eval_learner_step = global_logs.learner_step.value
            eval_record.append(last_eval_learner_step)

            # sync global model to local
            for dqn_id in range(args.maxmin_dqn_num):
                local_maxmin_dqns[dqn_id].load_state_dict(maxmin_dqns[dqn_id].state_dict())
                local_maxmin_dqns[dqn_id].eval()
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
            while nepisodes < args.agent_params.evaluator_nepisodes:
                # deal w/ reset
                if flag_reset:
                    # reset episode stats
                    episode_steps = 0
                    episode_reward = 0.
                    # reset game
                    experience = env.reset()
                    assert experience.state1 is not None
                    # flags
                    flag_reset = False

                maxmin_dqn_qvalues = torch.zeros(args.maxmin_dqn_num, args.action_shape, args.action_space)
                for dqn_id in range(args.maxmin_dqn_num):
                    action, qvalue, max_qvalue, qvalues = maxmin_dqns[dqn_id].get_action(experience.state1, device=local_device)
                    maxmin_dqn_qvalues[dqn_id] = qvalues
                action = np.array([maxmin_dqn_qvalues.min(dim=0)[0].max(dim=1)[1].__array__()])
                # run a single step
                experience = env.step(action)

                # check conditions & update flags
                if experience.terminal1:
                    nepisodes_solved += 1
                    flag_reset = True
                if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
                    flag_reset = True

                # update counters & stats
                step += 1
                episode_steps += 1
                episode_reward += experience.reward
                if flag_reset:
                    nepisodes += 1
                    total_steps += episode_steps
                    total_reward += episode_reward
                    eval_record.append(episode_reward)
            eval_record.append(total_reward / nepisodes)
            eval_csv_logs.append(eval_record)
            print("evaluation {}".format(eval_record))
            df = pd.DataFrame(data=eval_csv_logs)
            df.to_csv(args.save_log_csv)
            print("csv log was saved as {}".format(args.save_log_csv))

            # report stats
            # push local stats to logger
            with evaluator_logs.logger_lock.get_lock():
                evaluator_logs.total_steps.value = total_steps
                evaluator_logs.total_reward.value = total_reward
                evaluator_logs.nepisodes.value = nepisodes
                evaluator_logs.nepisodes_solved.value = nepisodes_solved
                evaluator_logs.logger_lock.value = True

            # save model
            # print("Saving model " + args.model_name + " ...")
            # torch.save(local_model.state_dict(), args.model_name)
            # print("Saved  model " + args.model_name + ".")

    df = pd.DataFrame(data=eval_csv_logs)
    df.to_csv(args.save_log_csv)
    print("csv log was saved as {}".format(args.save_log_csv))
