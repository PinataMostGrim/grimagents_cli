#!/usr/bin/env python3

import logging
import grimagents.settings as settings

from grimagents.command_util import execute_command

# start cmd /K pipenv run mlagents-learn config\mkv_config.yaml --run-id=MKV_04 --train
# /C carry out command and close
# /K carry out command and remain


def main():

    # example:
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # root.addHandler(handler)

    # handler = logging.StreamHandler(sys.stdout)

    # handler = logging.FileHandler('file.log')
    # handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(levelname)s: %(message)s')
    # handler.setFormatter(formatter)

    # trainer_logger = logging.getLogger('grimagents.mocktrainer')
    # trainer_logger.addHandler(handler)

    cwd = settings.get_project_folder_absolute()
    command = ['pipenv', 'run', 'python', 'grim-agents\mock_trainer.py']
    output = execute_command(command, cwd)

    print('')
    print('MOCK WRAPPER'.center(40, '='))
    print(output)


if __name__ == '__main__':
    main()
