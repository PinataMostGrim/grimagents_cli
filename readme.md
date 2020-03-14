# grimagents
**grimagents** is collection of command line applications that wrap [Unity's Machine Learning Agents](https://github.com/Unity-Technologies/ml-agents) toolkit with more automation.

**grimagents** CLI features include:
- Initiate training using arguments loaded from a configuration file
- Easily resume the last training run (`grimagents --resume`)
- *(Optional)* Automatically add time-stamp to training run-ids
- *(Optional)* Override loaded configuration values with command line arguments

**grimsearch** CLI features include:
- Search for optimal hyperparameter settings using Grid, Random, and Bayesian strategies

**grimwrapper** CLI features include:
- Display estimated time remaining
- *(Optional)* Automatically copy trained models to another location after training finishes (for example, into a Unity project)


## Requirements
- Pipenv accessible through the PATH environment variable
- A virtual environment setup for the `MLAgents` project folder using pipenv and Python 3.6
- ML-Agents 0.10.1


## Installation
- Copy or clone this repository into the `MLAgents` project in a folder named `grim-agents`
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
                  [--edit-trainer-config <file>] [--edit-curriculum <file>]
                  [--tensorboard-start] [--resume] [--dry-run]
                  [--trainer-config TRAINER_CONFIG] [--env ENV]
                  [--sampler SAMPLER] [--lesson LESSON] [--run-id RUN_ID]
                  [--base-port BASE_PORT] [--num-envs NUM_ENVS] [--inference]
                  [--graphics | --no-graphics] [--timestamp | --no-timestamp]
                  [--multi-gpu | --no-multi-gpu]
                  configuration_file ...

CLI application that wraps Unity ML-Agents with more automation.

positional arguments:
  configuration_file    Configuration file to extract training arguments from
  args                  Additional arguments applied to training (ex. --slow,
                        --debug, --load)

optional arguments:
  -h, --help            show this help message and exit
  --list, -l            List mlagents-learn training options
  --edit-config <file>  Open a grimagents configuration file for editing
  --edit-trainer-config <file>
                        Open a trainer configuration file for editing
  --edit-curriculum <file>
                        Open a curriculum file for editing
  --tensorboard-start, -s
                        Start tensorboard server
  --resume, -r          Resume the last training run
  --dry-run, -n         Print command without executing
  --trainer-config TRAINER_CONFIG
                        Overrides configuration setting
  --env ENV             Overrides configuration setting
  --sampler SAMPLER     Overrides configuration setting
  --lesson LESSON       Overrides configuration setting
  --run-id RUN_ID       Overrides configuration setting
  --base-port BASE_PORT
                        Overrides configuration setting
  --num-envs NUM_ENVS   Overrides configuration setting
  --inference           Load environment in inference mode instead of training
  --graphics            Overrides configuration setting
  --no-graphics         Overrides configuration setting
  --timestamp           Append timestamp to run-id. Overrides configuration
                        setting.
  --no-timestamp        Do not append timestamp to run-id. Overrides
                        configuration setting.
  --multi-gpu           Use multi-gpu if supported. Overrides configuration
                        setting.
  --no-multi-gpu        Do not use multi-gpu. Overrides
                        configuration setting.
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

Initiate training with the `3DBall_grimagents.json` configuration file, but override several configuration values (in this case, to manually resume an earlier training run):
```
grimagents grim-agents\etc\3DBall_grimagents.json --run-id 3DBall-2019-06-20_19-23-58 --no-timestamp --load
```

Resume the last training run:
```
grimagents --resume
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
Create or edit a `grimagents` training configuration file (default search parameters are added automatically):
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
                        --slow, --debug, --load)

optional arguments:
  -h, --help            show this help message and exit
  --run-id <run-id>     Run id for the training session
  --export-path EXPORT_PATH
                        Export trained policies to this path
```


## Configuration
#### grimagents Configuration
Values that are not present in a configuration file or left empty will not be passed on to `mlagents-learn`. `--trainer-config-path` and `--run-id` are the only mandatory configuration values. Override arguments sent to `grimagents` from the command line will be sent to `mlagents-learn` instead of those loaded from the configuration file.

All paths stored in configuration files should be relative paths from the current working directory. It is advisable to run `grimagents` modules from the `MLAgents` project root folder and configure paths accordingly.

`--timestamp` and `--inference` configuration values are consumed by the `grimagents` module and not passed on to `grimwrapper` or `mlagents-learn`.

Example configuration files can be found in the `etc` folder.


Example `grimagents` configuration:
```json
{
    "trainer-config-path": "grim-agents/etc/3DBall_config.yaml",
    "--env": "builds/3DBall/3DBall.exe",
    "--sampler": "grim-agents/etc/3dball_generalize.yaml",
    "--export-path": "UnitySDK/Assets/ML-Agents/Examples/3DBall/TFModels",
    "--run-id": "3DBall",
    "--timestamp": true
}
```

#### grimsearch Configuration
Grid Search is the default strategy used by `grimsearch`. Each hyperparameter value added to the search configuration will dramatically increase the number of training runs executed during a Grid Search. Often it can be helpful to run a limited grid search with hyperparameter values bracketing either side of their current value.

Random Search can be applied using the `--random` argument. When used, a random value is chosen between the minimum and maximum values defined for each hyperparameter in the search configuration. A hyperparameter with only one value defined will not be randomized.

When the `--bayesian` argument is present, [Bayesian optimization](https://github.com/fmfn/BayesianOptimization) will be used to search for optimal hyperparameters. Two values are required for each hyperparameter specified for the search; a minimum and maximum.

`grimsearch` only supports searching hyperparamters for one brain at a time. `grimsearch` will respect `--num-envs` and `--num-runs` while running searches and will also export the trained policy for every search if `--export-path` is present in the configuration file. This may not be desirable as each successive search will overwrite the previous policy's file.

Reward Signals can be included in hyperparameter searches by using a period-separated string in search configuration keys.

```json
{

  ...

  "search": {
      "brain": {
          "name": "3DBallLearning",
          "hyperparameters":
          {
            "reward_signals.extrinsic.gamma" : [0.98 , 0.99],
            "reward_signals.curiosity.strength" : [0.001, 0.1],
            "reward_signals.curiosity.encoding_size" : [64, 256]
          }
      }
  }
```

As `buffer_size` should always be a multiple of the `batch_size`, it impossible to perform searches on one or the other using static values. A special `buffer_size_multiple` value can be defined that allows `grimsearch` to dynamically set the `buffer_size` based directly on the `batch_size`.

```json
{

  ...

  "search": {
      "brain": {
          "name": "3DBallLearning",
          "hyperparameters":
          {
              "beta": [1e-4, 1e-2],
              "buffer_size_multiple": [4, 10]
          }
      }
  }
```


## Notes
`grimagents`, `grimwrapper`, and `grimsearch` initiate training using a pipenv subprocess call.

The `grimagents --resume` argument will not remember how far through a curriculum the previous training run progressed but will accept a `--lesson` override argument.

grimagent's log file is written into `grim-agents/logs` by default, but this can be changed in `settings.py`.

Bayesian search will write the best configuration discovered into a yaml file named `<run-id>_bayes.yaml` next to the trainer config file used for the search. If the `--bayes-save` argument is used, an observations log file will be automatically generated with a timestamp in a folder next to the trainer config file. Likewise, the `--bayes-load` argument will load log files form the same folder. The folder name generated will take the form `<run_id>_bayes`. This folder should be cleared or deleted before beginning a new Bayesian search from scratch.
