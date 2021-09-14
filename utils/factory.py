from core.single_processes.logs import GlobalLogs
from core.single_processes.logs import ActorLogs
from core.single_processes.logs import DQNLearnerLogs, DDPGLearnerLogs
from core.single_processes.logs import EvaluatorLogs
GlobalLogsDict = {"dqn":  GlobalLogs,
                  "alter_dqn":  GlobalLogs,
                  "gmdqn":  GlobalLogs,
                  "ddpg": GlobalLogs}

ActorLogsDict = {"dqn":  ActorLogs,
                 "alter_dqn":  ActorLogs,
                 "gmdqn":  ActorLogs,
                 "ddpg": ActorLogs}

LearnerLogsDict = {"dqn":  DQNLearnerLogs,
                   "alter_dqn":  DQNLearnerLogs,
                   "gmdqn":  DQNLearnerLogs,
                   "ddpg": DDPGLearnerLogs}
EvaluatorLogsDict = {"dqn":  EvaluatorLogs,
                     "alter_dqn":  EvaluatorLogs,
                     "gmdqn":  EvaluatorLogs,
                     "ddpg": EvaluatorLogs}

from core.single_processes.dqn_logger import dqn_logger
from core.single_processes.ddpg_logger import ddpg_logger
from core.single_processes.dqn_actor import dqn_actor
from core.single_processes.ddpg_actor import ddpg_actor
from core.single_processes.dqn_learner import dqn_learner
from core.single_processes.ddpg_learner import ddpg_learner
from core.single_processes.evaluators import evaluator
from core.single_processes.testers import tester
# for gmdqn
from core.single_processes.gmdqn_actor import gmdqn_actor
from core.single_processes.gmdqn_learner import gmdqn_learner

LoggersDict = {"dqn":  dqn_logger,
               "alter_dqn":  dqn_logger,
               "gmdqn": dqn_logger,
               "ddpg": ddpg_logger}

ActorsDict = {"dqn":  dqn_actor,
              "alter_dqn":  dqn_actor,
              "gmdqn":  gmdqn_actor,
              "ddpg": ddpg_actor}

LearnersDict = {"dqn":  dqn_learner,
                "alter_dqn":  dqn_learner,
                "gmdqn":  gmdqn_learner,
                "ddpg": ddpg_learner}
EvaluatorsDict = {"dqn":  evaluator,
                  "alter_dqn":  evaluator,
                  "gmdqn":  evaluator,
                  "ddpg": evaluator}
TestersDict = {"dqn":  tester,
               "alter_dqn":  tester,
               "gmdqn": tester,
               "ddpg": tester}

from core.envs.atari_env import AtariEnv
EnvsDict = {"atari": AtariEnv}

from core.memories.shared_memory import SharedMemory
from core.memories.shared_traj_memory import SharedTrajMemory
MemoriesDict = {"shared": SharedMemory,
                "shared-traj": SharedTrajMemory,
                "none":   None}

from core.models.dqn_cnn_model import DqnCnnModel
from core.models.dqn_mlp_model import DqnMlpModel
from core.models.ddpg_mlp_model import DdpgMlpModel
ModelsDict = {"dqn-cnn":  DqnCnnModel,
              "gmdqn-mlp": DqnMlpModel,
              "gmdqn-cnn": DqnCnnModel,
              "ddpg-mlp": DdpgMlpModel}
