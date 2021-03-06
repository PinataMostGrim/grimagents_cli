# grimagents
**grimagents** is collection of command line applications that wrap the [Unity Machine Learning Agents](1) toolkit with more automation.

**grimagents** CLI features include:
- Initiate training using arguments loaded from a configuration file
- *(Optional)* Automatically add time-stamp to training run-ids
- *(Optional)* Override loaded configuration values with command line arguments

**grimsearch** CLI features include:
- Search for optimal hyperparameter settings using Grid, Random, and Bayesian strategies

**grimwrapper** CLI features include:
- Display estimated time remaining
- *(Optional)* Automatically copy trained models to another location after training finishes (for example, into a Unity project)


## Requirements
- Pipenv accessible through the PATH environment variable
- A virtual environment setup for the `MLAgents` project folder using Pipenv
- ML-Agents 0.16.1
- Tensorflow 2.2.0 (See [Notes](#notes))
- Scikit-learn 0.22.2 (See [Notes](#notes))


## Installation
- Copy or clone this repository in the `MLAgents` project into a folder named `grim-agents`
- Install `grimagents` with `pipenv install grim-agents/` or in "editable" mode with `pipenv install -e grim-agents/`
- *(Optional)* Give the `grim-agents` folder another name and update `settings.py` accordingly


## Usage
Once installed, `grimagents` can be executed from the `MLAgents` project root folder several ways:

Using console script entry points:
```
pipenv run grimagents -h
pipenv run grimsearch -h
pipenv run grimwrapper -h
```

Using the modules directly:
```
pipenv run python -m grimagents -h
pipenv run python -m grimagents.search -h
pipenv run python -m grimagents.training_wrapper -h
```

Using the batch files in the `grim-agents` folder:
```
grim-agents\grimagents.bat -h
grim-agents\grimsearch.bat -h
grim-agents\grimwrapper.bat -h
```


### grimagents
```
usage: grimagents [-h] [--list] [--edit-config <file>]
                  [--edit-trainer-config <file>] [--tensorboard-start]
                  [--resume] [--dry-run] [--trainer-config TRAINER_CONFIG]
                  [--env ENV] [--run-id RUN_ID] [--base-port BASE_PORT]
                  [--num-envs NUM_ENVS] [--inference]
                  [--graphics | --no-graphics] [--timestamp | --no-timestamp]
                  [--multi-gpu | --no-multi-gpu]
                  configuration_file ...

CLI application that wraps Unity ML-Agents with more automation.

positional arguments:
  configuration_file    Configuration file to extract training arguments from
  args                  Additional arguments applied to training (ex. --debug,
                        --load)

optional arguments:
  -h, --help            show this help message and exit
  --list, -l            List mlagents-learn training options
  --edit-config <file>  Open a grimagents configuration file for editing
  --edit-trainer-config <file>
                        Open a trainer configuration file for editing
  --tensorboard-start, -s
                        Start tensorboard server
  --resume, -r          Resume the training run specified by --run-id
  --dry-run, -n         Print command without executing
  --trainer-config TRAINER_CONFIG
                        Overrides configuration setting
  --env ENV             Overrides configuration setting
  --run-id RUN_ID       Overrides configuration setting
  --base-port BASE_PORT
                        Overrides configuration setting
  --num-envs NUM_ENVS   Overrides configuration setting
  --inference           Overrides configuration setting
  --graphics            Overrides configuration setting
  --no-graphics         Overrides configuration setting
  --timestamp           Append timestamp to run-id. Overrides configuration
                        setting.
  --no-timestamp        Do not append timestamp to run-id. Overrides
                        configuration setting.
  --multi-gpu           Use multi-gpu if supported. Overrides configuration
                        setting.
  --no-multi-gpu        Do not use multi-gpu. Overrides configuration setting.
```

#### Example usage
Create and edit a new `grimagents` configuration file or edit an existing one:
```
grimagents --edit-config grim-agents\etc\3DBall_grimagents.json
```

Initiate training with the `3DBall_grimagents.json` configuration file:
```
grimagents grim-agents\etc\3DBall.json
```

Initiate training with the `3DBall_grimagents.json` configuration file, but override the configuration value for `run-id` (in this case, to resume an earlier training run):
```
grimagents grim-agents\etc\3DBall_grimagents.json --run-id 3DBall-2019-06-20_19-23-58 --resume
```


### grimsearch
```
usage: grimsearch [-h] [--edit-config <file>] [--search-count]
                  [--resume <search index>] [--export-index <search index>]
                  [--random <n>]
                  [--bayesian <exploration_steps> <optimization_steps>]
                  [--bayes-save] [--bayes-load]
                  configuration_file

CLI application that performs a hyperparameter search

positional arguments:
  configuration_file    A grimagents configuration file containing search
                        parameters

optional arguments:
  -h, --help            show this help message and exit
  --edit-config <file>  Open a grimagents configuration file for editing. Adds
                        a default search entry if one is not present.
  --search-count        Output the total number of grid searches a grimagents
                        configuration file will attempt
  --resume <search index>
                        Resume grid search from <search index> (counting from
                        zero)
  --export-index <search index>
                        Export trainer configuration for grid search <index>
  --random <n>, -r <n>  Execute <n> random searches instead of performing a
                        grid search
  --bayesian <exploration_steps> <optimization_steps>, -b <exploration_steps> <optimization_steps>
                        Execute Bayesian Search using a number of exploration
                        steps and optimization steps
  --bayes-save, -s      Save Bayesian optimization progress log to folder
  --bayes-load, -l      Loads Bayesian optimization progress logs from folder
```

#### Example usage
Edit a `grimagents` training configuration file (default search parameters are added automatically):
```
grimsearch --edit-config grim-agents\etc\3DBall_grimagents.json
```

Initiate grid search with the `3DBall_grimagents.json` configuration file:
```
grimsearch grim-agents\etc\3DBall_grimagents.json
```

Initiate 5 random searches with the `3DBall_grimagents.json` configuration file:
```
grimsearch grim-agents\etc\3DBall_grimagents.json --random 5
```

Initiate a Bayesian search with the `3DBall_grimagents.json` configuration file using 5 exploration steps and 10 optimization steps:
```
grimsearch grim-agents\etc\3DBall_grimagents.json --bayesian 5 10
```


### grimwrapper
```
usage: grimwrapper [-h] [--run-id <run-id>] [--export-path EXPORT_PATH]
                   trainer_config_path ...

CLI application that wraps mlagents-learn with automatic exporting of trained
policies and exposes more training information in the console.

positional arguments:
  trainer_config_path   Configuration file that holds brain hyperparameters
  args                  Additional arguments passed on to mlagents-learn (ex.
                        --debug, --load)

optional arguments:
  -h, --help            show this help message and exit
  --run-id <run-id>     Run id for the training session
  --export-path EXPORT_PATH
                        Export trained policies to this path
```


## Configuration
#### grimagents Configuration
`--trainer-config-path` and `--run-id` are the only mandatory configuration values. Configuration values left empty will not be passed on to `mlagents-learn`. Override arguments sent to `grimagents` from the command line will be sent to `mlagents-learn` instead of those loaded from the configuration file.

All paths stored in configuration files should be relative paths from the current working directory. It is advisable to run `grimagents` modules from the `MLAgents` project root folder and configure paths accordingly.

`--timestamp` and `--inference` configuration values are consumed by the `grimagents` module and not passed on to `grimwrapper` or `mlagents-learn`.

Example configuration files can be found in the `etc` folder.


Example `grimagents` configuration:
```json
{
    "trainer-config-path": "grim-agents/etc/3DBall.yaml",
    "--env": "builds/3DBall/3DBall.exe",
    "--export-path": "UnitySDK/Assets/ML-Agents/Examples/3DBall/TFModels",
    "--run-id": "3DBall",
    "--timestamp": true
}
```

#### grimsearch Configuration
Grid Search is the default strategy used by `grimsearch`. Each hyperparameter value added to the search configuration will dramatically increase the number of training runs executed during a Grid Search. Often it can be helpful to run a limited grid search with hyperparameter values bracketing either side of their current value.

Random Search can be applied using the `--random` argument. When used, a random value is chosen between the minimum and maximum values (inclusive) defined for each hyperparameter in the search configuration. A hyperparameter with only one value defined will not be randomized.

When the `--bayesian` argument is present, [Bayesian Optimization](2) will be used to search for optimal hyperparameters. Two values are required for each hyperparameter specified for the search; a minimum and maximum.

`grimsearch` only supports searching hyperparameters for one behaviour at a time. `grimsearch` will respect `--num-envs` while running searches and will also export the trained policy for every search if `--export-path` is present in the configuration file. This may not be desirable as each successive search will overwrite the previous policy's file.

Hyperparameters should be defined using period-separated strings to designate nested relationships.

```json
{
    "search": {
        "behavior_name": "3DBall",
        "search_parameters": {
            "hyperparameters.batch_size": [512, 5120],
            "network_settings.hidden_units": [32, 512],
            "time_horizon": [32, 2048],
            "reward_signals.extrinsic.gamma": [0.98, 0.99]
        }
    }
}
```

As `buffer_size` should always be a multiple of the `batch_size`, it impossible to perform searches on one or the other using static values. A special `buffer_size_multiple` value can be defined that allows `grimsearch` to dynamically set the `buffer_size` based directly on the `batch_size`.

```json
{
    "search": {
        "behavior_name": "3DBall",
        "search_parameters": {
            "hyperparameters.batch_size": [512, 5120],
            "hyperparameters.buffer_size_multiple": [4, 10],
        }
    }
}
```

Additionally, `encoding_size` values (such as `hyperparameters.reward_signals.curiosity.encoding_size`) should always be a multiple of 4 and will be forced to the highest valid multiple below the value chosen by Bayesian Optimization.


## Notes
`grimagents`, `grimwrapper`, and `grimsearch` initiate training using a Pipenv subprocess call.

The `grimagents --resume` argument will not remember how far through a curriculum the previous training run progressed but will accept a `--lesson` override argument.

grimagent's log file is written into `grim-agents/logs` by default, but this can be changed in `settings.py`.

Bayesian search will write the best configuration discovered into a yaml file named `<run-id>_bayes.yaml` next to the trainer config file used for the search. If the `--bayes-save` argument is used, an observations log file will be automatically generated with a timestamp in a folder next to the trainer config file. Likewise, the `--bayes-load` argument will load log files from the same folder. The folder name generated will take the form `<run_id>_bayes`. This folder should be cleared or deleted before beginning a new Bayesian search from scratch.

`BayesianOptimization` requires `numpy >=1.19.0` while `Tensorflow 2.3.0` and greater requires `numpy <1.19.0`. `Tensorflow 2.2.0` must be used until current versions are updated to work with higher versions of `numpy` ([source](3)).

Newer versions of `scikit-learn` throw a `ValueError` when using Bayesian search. Until this issue is resolved, use `scikit-learn 0.22.2`. ([source](4)).


[1](https://github.com/Unity-Technologies/ml-agents)
[2](https://github.com/fmfn/BayesianOptimization)
[3](https://github.com/tensorflow/tensorflow/commit/79518facb4b857af9d9d5df2da463fdbf7eb0e3e)
[4](https://github.com/scikit-optimize/scikit-optimize/issues/910)
