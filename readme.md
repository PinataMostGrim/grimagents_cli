# grimagents
**grimagents** is collection of command line applications that wrap [Unity MLAgents](https://github.com/Unity-Technologies/ml-agents) `mlagents-learn` with several quality of life improvements. Features include:
- Load training arguments from a configuration file
- Optionally override loaded configuration arguments with command line arguments
- Optionally time-stamp the training run-id
- Optionally launch training in a new console window
- Logs `mlagents-learn` output to file
- Optionally exports trained models to another location after training finishes (for example, into a Unity project)


## Requirements
- Windows
- Pipenv accessible through the PATH environment variable
- A virtual environment setup for the MLAgents project folder using Python 3.6


## Installation
- Copy or clone this repository into the MLAgents project in a folder named `grim-agents`
- Copy or move `grimagents.bat` into the MLAgents project root folder
- *(Optional)* Give the folder another name but update `grimagents.bat` and `settings.py` accordingly


## Usage
Training can be initiated several ways:
- Execute the provided `grimagents.bat` file
- Execute the module in python using `python -m grimagents`
- Execute `training_wrapper.py` in python directly


### grimagents
```
usage: grim-agents [-h] [--list] [--edit-config FILE]
                   [--edit-trainer-config FILE] [--edit-curriculum FILE]
                   [--new-window] [--tensorboard-start] [--resume] [--env ENV]
                   [--lesson LESSON] [--run-id RUN_ID] [--num-envs NUM_ENVS]
                   [--graphics | --no-graphics] [--timestamp | --no-timestamp]
                   configuration_file ...

CLI application that wraps Unity ML-Agents with some quality of life
improvements.

positional arguments:
  configuration_file    Configuration file to extract training arguments from
  args                  Additional arguments applied to training (ex. --slow,
                        --debug, --load)

optional arguments:
  -h, --help            show this help message and exit
  --list                List mlagents-learn training options
  --edit-config FILE    Open a grimagents configuration file for editing
  --edit-trainer-config FILE
                        Open a trainer configuration file for editing
  --edit-curriculum FILE
                        Open a curriculum file for editing
  --new-window          Run training process in a new console window
  --tensorboard-start   Start tensorboard server
  --resume              Resume the last training run
  --env ENV
  --lesson LESSON
  --run-id RUN_ID
  --num-envs NUM_ENVS
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
grimagents --edit-config grim-agents\config\3DBall.json
```

Initiate training with the `3DBall.json` configuration file:
```
grimagents grim-agents\config\3DBall.json
```

Initiate training with the `3DBall.json` configuration file, but override several configuration values (in this case, to resume an earlier training run):
```
grimagents grim-agents\config\3DBall.json --run-id 3DBall-2019-06-20_19-23-58 --no-timestamp --load
```


### training_wrapper
```
usage: training_wrapper [-h] [--run-id <run-id>] [--export-path EXPORT_PATH]
                        [--log-filename LOG_FILENAME]
                        trainer_config_path ...

CLI application that wraps mlagents-learn with quality of life improvements.

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


## Configuration
Values that are not present in a configuration file or left empty will not be passed on to `mlagents-learn`. `trainer-config-path` and `run-id` are the only mandatory configuration values. Override arguments sent to `grimagents` will be sent to `mlagents-learn` instead of those loaded from the configuration file.

All paths stored in configuration files should be relative paths from the MLAgents project root folder to the target asset or folder. This example configuration file is included at `config\3DBall.json`.

The `--timestamp` and `--log-filename` values are consumed by the main module and not passed on to `training_wrapper` or `mlagents-learn`.

#### Example configuration file
```json
{
    "trainer-config-path": "config\\3DBall.yaml",
    "--env": "builds\\3DBall\\Unity Environment.exe",
    "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
    "--curriculum": "",
    "--keep-checkpoints": "",
    "--lesson": "",
    "--run-id": "3DBall",
    "--num-runs": "",
    "--save-freq": "",
    "--seed": "",
    "--base-port": "",
    "--num-envs": "",
    "--no-graphics": false,
    "--timestamp": true
}
```


## Notes
Both `grimagents` and `training_wrapper` initiate training using a Pipenv process call and both initiate training with the project's root folder set as the current working directory. `training_wrapper` potentially works with Linux but is untested while `grimagents` requires Windows.

Log files are written into `grim-agents\logs` by default, but this can be changed in `settings.py`. A limited amount of `mlagent-learn`'s output is sent to stdout so only that portion will be captured in the log file.
