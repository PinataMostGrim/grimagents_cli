"""Stores string constants used throughout the package."""

# ML-Agents
ML_TRAINER_CONFIG_PATH = 'trainer-config-path'
ML_ENV = '--env'
ML_CURRICULUM = '--curriculum'
ML_SAMPLER = '--sampler'
ML_KEEP_CHECKPOINTS = '--keep-checkpoints'
ML_LESSON = '--lesson'
ML_LOAD = '--load'
ML_RUN_ID = '--run-id'
ML_NUM_RUNS = '--num-runs'
ML_SAVE_FREQUENCY = '--save-freq'
ML_SEED = '--seed'
ML_SLOW = '--slow'
ML_TRAIN = '--train'
ML_BASE_PORT = '--base-port'
ML_NUM_ENVS = '--num-envs'
ML_DOCKER_TARGET = '--docker-target-name'
ML_NO_GRAPHICS = '--no-graphics'
ML_DEBUG = '--debug'
ML_MULTI_GPU = '--multi-gpu'

# Hyperparameters
HP_TRAINER = 'trainer'
HP_BATCH_SIZE = 'batch_size'
HP_BETA = 'beta'
HP_BUFFER_SIZE = 'buffer_size'
HP_EPSILON = 'epsilon'
HP_HIDDEN_UNITS = 'hidden_units'
HP_LAMBD = 'lambd'
HP_LEARNING_RATE = 'learning_rate'
HP_LEARNING_RATE_SCHEDULE = 'learning_rate_schedule'
HP_MAX_STEPS = 'max_steps'
HP_MEMORY_SIZE = 'memory_size'
HP_NORMALIZE = 'normalize'
HP_NUM_EPOCH = 'num_epoch'
HP_NUM_LAYERS = 'num_layers'
HP_TIME_HORIZON = 'time_horizon'
HP_SEQUENCE_LENGTH = 'sequence_length'
HP_SUMMARY_FREQ = 'summary_freq'
HP_USE_RECURRENT = 'use_recurrent'
HP_VIS_ENCODE_TYPE = 'vis_encode_type'

# Reward signals
HP_REWARD_SIGNALS = 'reward_signals'
HP_EXTRINSIC = 'extrinsic'
HP_CURIOSITY = 'curiosity'
HP_GAIL = 'gail'

HP_STRENGTH = 'strength'
HP_GAMMA = 'gamma'
HP_ENCODING_SIZE = 'encoding_size'
HP_USE_ACTIONS = 'use_actions'
HP_USE_VAIL = 'use_vail'

# Trainer Configuration
TC_DEFAULT = 'default'

# Curriculum
CU_MEASURE = 'measure'
CU_PROGRESS = 'progress'
CU_REWARD = 'reward'
CU_THRESHOLDS = 'thresholds'
CU_MIN_LESSON_LENGTH = 'min_lesson_length'
CU_SIGNAL_SMOOTHING = 'signal_smoothing'
CU_PARAMETERS = 'parameters'

# Grimagents
GA_EXPORT_PATH = '--export-path'
GA_TIMESTAMP = '--timestamp'
GA_INFERENCE = '--inference'
GA_ADDITIONAL_ARGS = 'additional-args'

# Grimsearch
GS_SEARCH = 'search'
GS_BRAIN = 'brain'
GS_NAME = 'name'
GS_HYPERPARAMS = 'hyperparameters'
GS_BUFFER_SIZE_MULTIPLE = 'buffer_size_multiple'
