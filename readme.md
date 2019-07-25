# grimagents
**grimagents** is collection of command line applications that wrap [Unity MLAgents](https://github.com/Unity-Technologies/ml-agents) `mlagents-learn` with several quality of life improvements. Features include:
- Load training arguments from a configuration file
- Hyperparameter grid search
- Log `mlagents-learn` output to file
- Quickly resume the last training run
- *(Optional)* override loaded configuration values with command line arguments
- *(Optional)* time-stamp training run-ids
- *(Optional)* launch training in a new console window
- *(Optional)* export trained models to another location after training finishes (for example, into a Unity project)


## Requirements
- Windows
- Pipenv accessible through the PATH environment variable
- A virtual environment setup for the MLAgents project folder using Python 3.6


## Installation
- Copy or clone this repository into the MLAgents project in a folder named `grim-agents`
- Copy or move `grimagents.bat`, `training_wrapper.bat`, and `search.bat` into the MLAgents project root folder. These batch files automatically add the `grimagents` package folder to `PYTHONPATH` and execute their module using Pipenv.
- *(Optional)* Give the grim-agents folder another name and update `grimagents.bat`, `training_wrapper.bat`, `search.bat`, and `settings.py` accordingly


## Usage
Training can be initiated several ways from the MLAgents project root folder:
- Execute `grimagents.bat` file
- Execute the module in python using `python -m grimagents`
- Execute `training_wrapper.py` in python directly

Grid search can be initiated by executing `search.bat` from the MLAgents project root folder.

### grimagents
```
usage: grimagents [-h] [--list] [--edit-config <file>]
                  [--edit-trainer-config <file>] [--edit-curriculum <file>]
                  [--new-window] [--tensorboard-start] [--resume] [--dry-run]
                  [--trainer-config TRAINER_CONFIG] [--env ENV]
                  [--lesson LESSON] [--run-id RUN_ID] [--num-envs NUM_ENVS]
                  [--inference] [--graphics | --no-graphics]
                  [--timestamp | --no-timestamp]
                  configuration_file ...

CLI application that wraps Unity ML-Agents with quality of life
improvements.

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
  --new-window, -w      Run process in a new console window
  --tensorboard-start, -s
                        Start tensorboard server
  --resume, -r          Resume the last run
  --dry-run, -n         Print command without executing
  --trainer-config TRAINER_CONFIG
  --env ENV
  --lesson LESSON
  --run-id RUN_ID
  --num-envs NUM_ENVS
  --inference           Load environment in inference mode instead of training
  --graphics
  --no-graphics
  --timestamp           Append timestamp to run-id. Overrides configuration
                        setting.
  --no-timestamp        Do not append timestamp to run-id. Overrides
                        configuration setting.
```

#### Example usages
Create and edit a new grimagents training configuration file or edit an existing one:
```
grimagents --edit-config grim-agents\config\3DBall_grimagents.json
```

Initiate training with the `3DBall_grimagents.json` configuration file:
```
grimagents grim-agents\config\3DBall.json
```

Initiate training with the `3DBall_grimagents.json` configuration file, but override several configuration values (in this case, to manually resume an earlier training run):
```
grimagents grim-agents\config\3DBall_grimagents.json --run-id 3DBall-2019-06-20_19-23-58 --no-timestamp --load
```

Resume the last training run:
```
grimagents --resume
```


### training_wrapper
```
usage: training_wrapper [-h] [--run-id <run-id>] [--export-path EXPORT_PATH]
                        [--log-filename LOG_FILENAME]
                        trainer_config_path ...

CLI application that wraps mlagents-learn with logging to file and automatic
exporting of trained policy.

positional arguments:
  trainer_config_path   Configuration file that holds brain hyperparameters
  args                  Additional arguments passed on to mlagents-learn (ex.
                        --slow, --debug, --load)

optional arguments:
  -h, --help            show this help message and exit
  --run-id <run-id>     Run id for the training session
  --export-path EXPORT_PATH
                        Export trained models to this path
  --log-filename LOG_FILENAME
                        Write log output to this file. Defaults to run-id.
```


### search
```
usage: search [-h] [--edit-config <file>] configuration_file

CLI application that performs a hyperparameter grid search

positional arguments:
  configuration_file    grimagents configuration file with search parameters

optional arguments:
  -h, --help            show this help message and exit
  --edit-config <file>  Open a grimagents configuration file for editing. Adds
                        a search entry if one is not present.
```

#### Example usages
Create or edit a grimagents training configuration file (default search parameters are added automatically):
```
search --edit-config grim-agents\config\3DBall_grimagents.json
```

Initiate grid search with the `3DBall_grimagents.json` configuration file:
```
search grim-agents\config\3DBall.json
```


## Configuration
#### grimagents Configuration
Values that are not present in a configuration file or left empty will not be passed on to `mlagents-learn`. `trainer-config-path` and `run-id` are the only mandatory configuration values. Override arguments sent to `grimagents` from the command line will be sent to `mlagents-learn` instead of those loaded from the configuration file.

All paths stored in configuration files should be relative paths from the current working directory. It is advisable to run grimagents modules from the MLAgents project root folder and configure paths accordingly.

`--timestamp` and `--inference` configuration values are consumed by the main module and not passed on to `training_wrapper` or `mlagents-learn`.

An example configuration file is included at `config\3DBall_grimagents.json`.


Example grimagents configuration:
```json
{
    "trainer-config-path": "config\\3DBall_config.yaml",
    "--env": "builds\\3DBall\\3DBall.exe",
    "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
    "--run-id": "3DBall",
    "--timestamp": true
}
```

#### search Configuration
Each hyperparameter value added to the search configuration will dramatically increase the number of training runs executed. Often it can be helpful to run a limited grid search with hyperparameter values bracketing either side of their current value.

`search` only supports grid searching one brain at a time. `search` will respect `--num-envs` and `--num-runs` while running a grid search, and will export the trained policy for every search if `--export-path` is present in the configuration file. This may not be desirable as each successive search will overwrite the previous policy's file.

```json
{
    "trainer-config-path": "config\\3DBall_config.yaml",
    "--env": "builds\\3DBall\\3DBall.exe",
    "--export-path": "",
    "--run-id": "3DBall",
    "--timestamp": true,
    "search": {
        "brain": {
            "name": "3DBallLearning",
            "hyperparameters":
            {
                "beta": [1e-4, 1e-2]
            }
        }
    }
}
```


## Notes
Both `grimagents` and `training_wrapper` initiate training using a Pipenv process call and both initiate training with the current working directory set to the project's root folder. `training_wrapper` potentially works with Linux but is untested while `grimagents` requires Windows.

The `grimagents' '--resume` argument will not remember how far through a curriculum the previous training run progressed but will accept a `--lesson` override argument.

Log files are written into `grim-agents\logs` by default, but this can be changed in `settings.py`. A very limited amount of `mlagent-learn`'s output is sent to `stdout` and only that portion will be captured by the log file.
