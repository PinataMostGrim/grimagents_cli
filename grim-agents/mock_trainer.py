#!/usr/bin/env python3

import os
import sys
import time


print('')
print('Mock Training')
print(f'Python executable: {os.path.abspath(sys.executable)}')
print(f'Python version: {sys.version}')
print('')

print('Starting mock training...')
for i in range(5):
    print('Training...')
    sys.stdout.flush()
    time.sleep(1)

print('Mock training complete')
