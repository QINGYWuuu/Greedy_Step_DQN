import time
import numpy as np
from collections import deque
import torch
import pandas as pd
from utils.helpers import reset_experience


def evaluator(args,
              global_logs,
              evaluator_logs,
              env_prototype,
              model_prototype,
              dqn):
    # env
    env = env_prototype(args.env_params, 0)
    env.eval()
    # model
    local_device = torch.device('cuda')#('cpu')


    evaluate_dqn = model_prototype(args.model_params,
                                   args.norm_val,
                                   args.state_shape,
                                   args.action_space,
                                   args.action_shape).to(local_device)
    evaluate_dqn.load_state_dict(dqn.state_dict())
    evaluate_dqn.eval()

    torch.set_grad_enabled(False)
    last_eval_learner_step = 0
    eval_csv_logs = []
    while global_logs.learner_step.value < args.agent_params.steps:

        if global_logs.learner_step.value % 10000 == 0 and global_logs.learner_step.value > last_eval_learner_step:

            eval_record = []  # [step, eva 1, eva 2, ..., eva n, eva aver]
            last_eval_learner_step = global_logs.learner_step.value
            eval_record.append(last_eval_learner_step)

            # sync global model to local
            evaluate_dqn.load_state_dict(dqn.state_dict())
            evaluate_dqn.eval()
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

                with torch.no_grad():
                    action, qvalue, max_qvalue, qvalues = evaluate_dqn.get_action(experience.state1, 0, device=local_device)
                experience = env.step(action)

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
            df.to_csv(args.evaluation_csv_file)
            print("csv log was saved as {}".format(args.evaluation_csv_file))

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
    df.to_csv(args.evaluation_csv_file)
    print("csv log was saved as {}".format(args.evaluation_csv_file))
