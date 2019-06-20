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
usage: grim-agents [-h] [--list] [--edit-config FILE] [--new-window]
                   [--tensorboard-start] [--env ENV] [--lesson LESSON]
                   [--run-id RUN_ID] [--num-envs NUM_ENVS]
                   [--graphics | --no-graphics] [--timestamp | --no-timestamp]
                   configuration_file ...

CLI application that wraps Unity ML-Agents with some quality of life
improvements.

positional arguments:
  configuration_file   Configuration file to load training arguments from
  args                 Additional arguments applied to training (ex. --slow,
                       --debug, --load)

optional arguments:
  -h, --help           show this help message and exit
  --list               List mlagents-learn training options
  --edit-config FILE   Open a configuration file for editing
  --new-window         Run training process in a new console window
  --tensorboard-start  Start tensorboard server
  --env ENV
  --lesson LESSON
  --run-id RUN_ID
  --num-envs NUM_ENVS
  --graphics
  --no-graphics
  --timestamp          Append timestamp to run-id. Overrides configuration
                       setting.
  --no-timestamp       Do not append timestamp to run-id. Overrides
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
usage: training_wrapper [-h] [--run-id <run-id>] [--timestamp]
                        [--export-path EXPORT_PATH]
                        trainer_config_path ...

CLI application that wraps mlagents-learn with quality of life improvements.

positional arguments:
  trainer_config_path   Configuration file that holds brain hyperparameters
  args                  Additional arguments passed on to mlagents-learn (ex.
                        --slow, --debug, --load)

optional arguments:
  -h, --help            show this help message and exit
  --run-id <run-id>     Run id for the training session
  --timestamp           Append a timestamp to the run-id. Timestamp will not
                        be applied to log file name.
  --export-path EXPORT_PATH
                        Export trained models to this path
```


## Notes
Both `grimagents` and `training_wrapper` initiate training using a Pipenv process call and both initiate training with the project's root folder set as the current working directory. `training_wrapper` potentially works with Linux but is untested while `grimagents` requires Windows.

All paths stored in configuration files should be relative paths from the MLAgents' project root folder to the target asset or folder. An example configuration file is provided at `config\3DBall.json`.

Log files are written into `grim-agents\logs` by default, but this can be changed in `settings.py`. A limited amount of mlagent-learn's output is sent to stdout so only that portion will be captured in the log file.
