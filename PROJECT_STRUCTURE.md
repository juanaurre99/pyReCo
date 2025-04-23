# pyReCo Project Structure Documentation

## Overview
pyReCo (Python Reservoir Computing) is a library developed by the Cyber-Physical Systems in Mechanical Engineering (CPSME) research group at TU Berlin. It focuses on time series forecasting and research using Reservoir Computing techniques.

## Project Structure

### Root Directory
- `src/` - Main source code directory
- `tests/` - Test files and test data
- `docs/` - Documentation files
- `examples/` - Example usage and tutorials
- `setup.py` - Package installation configuration
- `setup.cfg` - Additional package configuration
- `pyproject.toml` - Modern Python project configuration
- `tox.ini` - Test automation configuration
- `README.md` - Project overview and basic usage
- `CONTRIBUTING.rst` - Guidelines for contributing
- `LICENSE.txt` - Project license
- `CHANGELOG.md` - Version history and changes

### Source Code (`src/pyreco/`)

#### Core Modules (`core/`)

1. **custom_models.py**
   - Implements the CustomModel API
   - Provides more flexible RC model configurations
   - Allows custom layer stacking and configuration

2. **layers.py**
   - Defines different types of network layers:
     - InputLayer
     - RandomReservoirLayer
     - ReadoutLayer
   - Implements layer-specific functionality and properties

3. **optimizers.py**
   - Contains optimization algorithms for training
   - Implements Ridge regression and other optimization methods

4. **initializer.py**
   - Network initialization utilities
   - Weight initialization methods
   - Parameter initialization

5. **pruning.py**
   - Implements network pruning algorithms
   - Node removal and network optimization
   - Performance optimization tools

6. **network_prop_extractor.py**
   - Network property extraction
   - Feature calculation
   - Property analysis tools

#### Utility Modules (`utils/`)

1. **utils_data.py**
   - Data preprocessing and manipulation utilities
   - Time series handling functions
   - Data format conversion utilities

2. **utils_networks.py**
   - Network-related utility functions
   - Network property calculations
   - Network manipulation tools

3. **plotting.py**
   - Visualization utilities
   - Result plotting functions
   - Network visualization tools

#### Analysis Modules (`analysis/`)

1. **graph_analyzer.py**
   - Network graph analysis tools
   - Graph property calculations
   - Network structure analysis

2. **node_analyzer.py**
   - Individual node analysis tools
   - Node importance assessment
   - Node contribution analysis

3. **node_selector.py**
   - Node selection algorithms
   - Important node identification
   - Node subset selection

4. **remove_transients.py**
   - Transient state removal
   - Initial condition handling
   - State stabilization

#### Metrics Module (`metrics/`)

1. **metrics.py**
   - Defines performance evaluation metrics
   - Includes MSE, MAE, and other relevant metrics
   - Implements metric calculation utilities

#### Validation Module (`validation/`)

1. **cross_validation.py**
   - Implements k-fold cross-validation
   - Model evaluation utilities
   - Performance assessment tools

## Key Classes and Their Purposes

### CustomModel (core/custom_models.py)
- Flexible RC implementation
- Custom layer stacking
- Advanced configuration options

### Layer Classes (core/layers.py)
1. **InputLayer**
   - Handles input data
   - Input preprocessing
   - Input dimension management

2. **RandomReservoirLayer**
   - Implements reservoir dynamics
   - Node connections
   - State updates

3. **ReadoutLayer**
   - Output generation
   - Weight training
   - Prediction computation

## API Structure

### Model API
- Follows scikit-learn syntax
- Methods:
  - `.fit()` - Model training
  - `.predict()` - Making predictions
  - `.evaluate()` - Performance evaluation

### CustomModel API
- Follows TensorFlow Sequential API style
- Methods:
  - `.add()` - Layer addition
  - `.compile()` - Model compilation
  - `.fit()` - Training
  - `.predict()` - Prediction

## Data Format Requirements

### Input Data Shape
- Format: `[n_batch, n_time, n_states]`
  - `n_batch`: Number of samples
  - `n_time`: Time steps per sample
  - `n_states`: Number of states/channels

## Dependencies
- NumPy
- SciPy
- scikit-learn
- TensorFlow (optional)
- Matplotlib (for plotting)

## Testing Structure
- Unit tests in `tests/`
- Test automation via tox
- Coverage reporting via .coveragerc

## Documentation
- Official documentation in `docs/`
- ReadTheDocs configuration in `.readthedocs.yml`
- Example usage in `examples/` 