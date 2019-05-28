#!/usr/bin/env python3


from grimagents.command import execute_command
from grimagents.command import execute_command_shell

# start cmd /K pipenv run mlagents-learn config\mkv_config.yaml --run-id=MKV_04 --train
# /C carry out command and close
# /K carry out command and remain


def main():
    # command = ['python', 'mock_trainer.py']
    command = ['pipenv', 'run', 'python', 'mock_trainer.py']
    execute_command(command)


if __name__ == '__main__':
    main()
