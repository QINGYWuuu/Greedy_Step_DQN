import numpy as np
import os
import torch
import torch.nn as nn
import time


CONFIGS = [
# agent_type, env_type, game,              memory_type, model_type
[ "gmdqn",      "atari",  "alien",           "shared-traj",    "gmdqn-cnn" ], # 0
[ "gmdqn",      "atari",  "amidar",          "shared-traj",    "gmdqn-cnn" ], # 1
[ "gmdqn",      "atari",  "assault",         "shared-traj",    "gmdqn-cnn" ], # 2
[ "gmdqn",      "atari",  "asterix",         "shared-traj",    "gmdqn-cnn" ], # 3
[ "gmdqn",      "atari",  "bank_heist",      "shared-traj",    "gmdqn-cnn" ], # 4
[ "gmdqn",      "atari",  "battle_zone",     "shared-traj",    "gmdqn-cnn" ], # 5
[ "gmdqn",      "atari",  "boxing",          "shared-traj",    "gmdqn-cnn" ], # 6
[ "gmdqn",      "atari",  "breakout",        "shared-traj",    "gmdqn-cnn" ], # 7
[ "gmdqn",      "atari",  "chopper_command", "shared-traj",    "gmdqn-cnn" ], # 8
[ "gmdqn",      "atari",  "crazy_climber",   "shared-traj",    "gmdqn-cnn" ], # 9
[ "gmdqn",      "atari",  "demon_attack",    "shared-traj",    "gmdqn-cnn" ], # 10
[ "gmdqn",      "atari",  "freeway",         "shared-traj",    "gmdqn-cnn" ], # 11
[ "gmdqn",      "atari",  "frostbite",       "shared-traj",    "gmdqn-cnn" ], # 12
[ "gmdqn",      "atari",  "gopher",          "shared-traj",    "gmdqn-cnn" ], # 13
[ "gmdqn",      "atari",  "hero",            "shared-traj",    "gmdqn-cnn" ], # 14
[ "gmdqn",      "atari",  "jamesbond",       "shared-traj",    "gmdqn-cnn" ], # 15
[ "gmdqn",      "atari",  "kangaroo",        "shared-traj",    "gmdqn-cnn" ], # 16
[ "gmdqn",      "atari",  "krull",           "shared-traj",    "gmdqn-cnn" ], # 17
[ "gmdqn",      "atari",  "kung_fu_master",  "shared-traj",    "gmdqn-cnn" ], # 18
[ "gmdqn",      "atari",  "ms_pacman",       "shared-traj",    "gmdqn-cnn" ], # 19
[ "gmdqn",      "atari",  "pong",            "shared-traj",    "gmdqn-cnn" ], # 20
[ "gmdqn",      "atari",  "private_eye",     "shared-traj",    "gmdqn-cnn" ], # 21
[ "gmdqn",      "atari",  "qbert",           "shared-traj",    "gmdqn-cnn" ], # 22
[ "gmdqn",      "atari",  "road_runner",     "shared-traj",    "gmdqn-cnn" ], # 23
[ "gmdqn",      "atari",  "seaquest",        "shared-traj",    "gmdqn-cnn" ], # 24*****************
[ "gmdqn",      "atari",  "up_n_down",       "shared-traj",    "gmdqn-cnn" ], # 25
# add custom configs ...
# agent_type, env_type, game,              memory_type, model_type
[ "gmdqn",      "atari",  "adventure",       "shared-traj",    "gmdqn-cnn" ], # 26
[ "gmdqn",      "atari",  "air_raid",        "shared-traj",    "gmdqn-cnn" ], # 27
[ "gmdqn",      "atari",  "asteroids",       "shared-traj",    "gmdqn-cnn" ], # 28
[ "gmdqn",      "atari",  "atlantis",        "shared-traj",    "gmdqn-cnn" ], # 29
[ "gmdqn",      "atari",  "beam_rider",      "shared-traj",    "gmdqn-cnn" ], # 30
[ "gmdqn",      "atari",  "berzerk",         "shared-traj",    "gmdqn-cnn" ], # 31
[ "gmdqn",      "atari",  "bowling",         "shared-traj",    "gmdqn-cnn" ], # 32*****************
[ "gmdqn",      "atari",  "carnival",        "shared-traj",    "gmdqn-cnn" ], # 33
[ "gmdqn",      "atari",  "centipede",       "shared-traj",    "gmdqn-cnn" ], # 34
[ "gmdqn",      "atari",  "defender",        "shared-traj",    "gmdqn-cnn" ], # 35
[ "gmdqn",      "atari",  "double_dunk",     "shared-traj",    "gmdqn-cnn" ], # 36
[ "gmdqn",      "atari",  "elevator_action", "shared-traj",    "gmdqn-cnn" ], # 37
[ "gmdqn",      "atari",  "enduro",          "shared-traj",    "gmdqn-cnn" ], # 38
[ "gmdqn",      "atari",  "fishing_derby",   "shared-traj",    "gmdqn-cnn" ], # 39
[ "gmdqn",      "atari",  "gravitar",        "shared-traj",    "gmdqn-cnn" ], # 40
[ "gmdqn",      "atari",  "ice_hockey",      "shared-traj",    "gmdqn-cnn" ], # 41
[ "gmdqn",      "atari",  "journey_escape",  "shared-traj",    "gmdqn-cnn" ], # 42
[ "gmdqn",      "atari",  "kaboom",          "shared-traj",    "gmdqn-cnn" ], # 43
[ "gmdqn",      "atari",  "montezuma_revenge","shared-traj",    "gmdqn-cnn" ], #44
[ "gmdqn",      "atari",  "name_this_game",  "shared-traj",    "gmdqn-cnn" ], # 45
[ "gmdqn",      "atari",  "phoenix",         "shared-traj",    "gmdqn-cnn" ], # 46
[ "gmdqn",      "atari",  "pitfall",         "shared-traj",    "gmdqn-cnn" ], # 47
[ "gmdqn",      "atari",  "pooyan",          "shared-traj",    "gmdqn-cnn" ], # 48
[ "gmdqn",      "atari",  "riverraid",       "shared-traj",    "gmdqn-cnn" ], # 49
[ "gmdqn",      "atari",  "robotank",        "shared-traj",    "gmdqn-cnn" ], # 50
[ "gmdqn",      "atari",  "skiing",          "shared-traj",    "gmdqn-cnn" ], # 51
[ "gmdqn",      "atari",  "solaris",         "shared-traj",    "gmdqn-cnn" ], # 52*****************
[ "gmdqn",      "atari",  "space_invaders",  "shared-traj",    "gmdqn-cnn" ], # 53
[ "gmdqn",      "atari",  "star_gunner",     "shared-traj",    "gmdqn-cnn" ], # 54
[ "gmdqn",      "atari",  "tennis",          "shared-traj",    "gmdqn-cnn" ], # 55
[ "gmdqn",      "atari",  "time_pilot",      "shared-traj",    "gmdqn-cnn" ], # 56
[ "gmdqn",      "atari",  "tutankham",       "shared-traj",    "gmdqn-cnn" ], # 57
[ "gmdqn",      "atari",  "venture",         "shared-traj",    "gmdqn-cnn" ], # 58*****************
[ "gmdqn",      "atari",  "video_pinball",   "shared-traj",    "gmdqn-cnn" ], # 59
[ "gmdqn",      "atari",  "wizard_of_wor",   "shared-traj",    "gmdqn-cnn" ], # 60
[ "gmdqn",      "atari",  "yars_revenge",    "shared-traj",    "gmdqn-cnn" ], # 61
[ "gmdqn",      "atari",  "zaxxon",          "shared-traj",    "gmdqn-cnn" ], # 62
]
class Params(object):
    def __init__(self, config, random_seed, num_actors):
        self.mode = 1
        self.config = config
        self.agent_type, self.env_type, self.game, self.memory_type, self.model_type = CONFIGS[config]
        self.seed = random_seed
        self.render = False
        self.visualize = True
        self.num_envs_per_actor = 1
        self.num_actors = num_actors
        self.num_learners = 1
        self.enable_double = True
        self.configs = "/All_RESULTS/greedy_step_dqn,env={},seed={},double={},actor={}".format(self.game, self.seed, self.enable_double, self.num_actors)


        self.root_dir = os.getcwd()
        if os.path.exists(self.root_dir + self.configs) is False:
            os.makedirs(self.root_dir + self.configs)
        self.model_file = self.root_dir + self.configs + "/model.pth"
        self.evaluation_csv_file = self.root_dir + self.configs + "/evaluation.csv"
        self.Q_value_csv_file = self.root_dir + self.configs + "/Q_value.csv"
        self.tensorboard_file = self.root_dir + self.configs

        self.model_file = None
        if self.mode == 2:
            self.model_file = self.model_name
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.visualize = False

class EnvParams(Params):
    def __init__(self, config, random_seed, num_actors):
        super(EnvParams, self).__init__(config=config, random_seed=random_seed, num_actors=num_actors)
        if "mlp" in self.model_type:    # low dim inputs, no preprocessing or resizing
            self.state_cha = 1          # NOTE: equals hist_len
            self.state_hei = 1          # NOTE: always 1 for mlp's
            self.state_wid = None       # depends on the env
        elif "cnn" in self.model_type:  # raw image inputs, need to resize or crop to this step_size
            self.state_cha = 4          # NOTE: equals hist_len
            self.state_hei = 84
            self.state_wid = 84

        if self.env_type == "atari":
            self.early_stop = 12500


class MemoryParams(Params):
    def __init__(self, config, random_seed, num_actors):
        super(MemoryParams, self).__init__(config=config, random_seed=random_seed, num_actors=num_actors)
        if self.memory_type == "shared-traj":
            if self.agent_type == "gmdqn":
                self.memory_size = 50000
            if "mlp" in self.model_type:
                self.tensortype = torch.FloatTensor
            elif "cnn" in self.model_type:      # save image as byte to save space
                self.tensortype = torch.ByteTensor

class ModelParams(Params):
    def __init__(self, config, random_seed, num_actors):
        super(ModelParams, self).__init__(config=config, random_seed=random_seed, num_actors=num_actors)

class AgentParams(Params):
    def __init__(self, config, random_seed, num_actors):
        super(AgentParams, self).__init__(config=config, random_seed=random_seed, num_actors=num_actors)
        if self.agent_type == "gmdqn":
            self.value_criteria = nn.MSELoss()
            self.optim = torch.optim.Adam
            self.num_tasks           = 1
            self.steps               = int(1e6)+1000
            self.gamma               = 0.99
            self.clip_grad           = np.inf#40.#100
            self.lr                  = 1e-4#2.5e-4/4.
            self.lr_decay            = False
            self.weight_decay        = 0.
            self.actor_sync_freq     = 100  # sync global_model to actor's local_model every this many steps
            # logger configs
            self.logger_freq         = 100   # log every this many secs
            self.actor_freq          = 250  # push & reset local actor stats every this many actor steps
            self.learner_freq        = 100  # push & reset local learner stats every this many learner steps
            self.evaluator_freq      = 300   # eval every this many secs
            self.evaluator_nepisodes = 10
            self.tester_nepisodes    = 50
            # off-policy specifics
            self.learn_start         = 5000 # start update params after this many steps 5000
            self.batch_size          = 128 # 128
            self.target_model_update = 250
            # dqn specifics
            self.eps                 = 0.4
            self.eps_alpha           = 7


class Options(Params):
    def __init__(self, config, random_seed, num_actors):
        Params.__init__(self, config=config, random_seed=random_seed, num_actors=num_actors)
        self.env_params = EnvParams(config=config, random_seed=random_seed, num_actors=num_actors)
        self.memory_params = MemoryParams(config=config, random_seed=random_seed, num_actors=num_actors)
        self.model_params = ModelParams(config=config, random_seed=random_seed, num_actors=num_actors)
        self.agent_params = AgentParams(config=config, random_seed=random_seed, num_actors=num_actors)
