### 2.0.0
- Added Linux support
- Removed support for running commands in a new terminal window. This functionality complicated adding Linux support.

### 1.1.0
- Added Bayesian Search strategy to grimsearch
- Added support for saving and loading Bayesian Search progress
- Added "Estimated Time Remaining" to grimwrapper's output

### 1.0.0
- Added grimsearch command line application
- Added Grid Search strategy to grimsearch
- Added Random Search strategy to grimsearch
- Added a command for adding or editing search configuration values in a grimagents configuration file
- Added setup.py to convert grimagents into a package and allow installation
- Remove support for loading multiple trainer configuration files. This functionality has been replaced with grimsearch.
- Numerous bug fixes

### 0.3.0
- Added support for loading multiple trainer configuration files to train in parallel
- Added "dry run" argument which shows the resulting command without executing it
- Added shortcut flags for many arguments
- Added an lesson override argument for the resume command
- Numerous bug fixes

### 0.2.0
- Added a command for resuming training
- Added a command for creating or editing a trainer configuration yaml file
- Added a command for creating or editing a curriculum configuration file
- Numerous bug fixes

### 0.1.0
- Initial version
- Added grimwrapper and grimagents command line applications
