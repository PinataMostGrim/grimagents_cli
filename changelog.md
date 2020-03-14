### 2.2.0
- Added support for ML-Agents 0.10.1
- Added configuration support for '--env-args'

### 2.1.2
- Added configuration support for '--docker-target-name' and '--debug'
- Fixed exception caused by performing searches using the 'encoding_size' reward signal
- Improved clarity of readme and example configuration files
- Moved tests inside the 'grimagents' package

### 2.1.1
- Update readme.md and configuration files

### 2.1.0
- Added support for ML-Agents 0.9.3
- Added support for performing hyperparameter searches on reward signals
- Added support for sampler files (generalized agents)
- Added support for the '--multi-gpu' argument
- Bayes search cleans up search configuration files when finished
- The best configuration found using Bayes search is now saved to `<run-id_bayes>.yaml`

### 2.0.1
- Fixed exception caused by bayesian-optimization update
- Fixed failing github action workflow

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
