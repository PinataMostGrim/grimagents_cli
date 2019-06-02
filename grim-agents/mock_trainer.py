#!/usr/bin/env python3

import logging
import os
import sys
import time


# I know I can send logging info into the std out stream by adjusting the basic config
# Ideally I want to do this using a handler though.
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
trainer_logger = logging.getLogger('grimagents.mocktrainer')

# handler = logging.FileHandler('file.log')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(levelname)s: %(message)s')
# handler.setFormatter(formatter)

# trainer_logger.addHandler(handler)

print('')
print('Mock Training')
print(f'Python executable: {os.path.abspath(sys.executable)}')
print(f'Python version: {sys.version}')
print('')

print(f'CWD: {os.getcwd()}')
print(f'Arguments: {sys.argv[1:]}')
print('')

trainer_logger.info('Starting mock training...')

for i in range(5):
    trainer_logger.info('Training...')
    sys.stdout.flush()
    time.sleep(1)

trainer_logger.info('Mock training complete')
