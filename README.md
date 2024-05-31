# Conciliator steering

The project is a code implementation for the TCML paper "Conciliator steering: Imposing user preference in MORL decision-making problems". The code enables the user to interactively or pre-definedly impose a priority weighting over rewards in a DeepSeaTreasure v1 benchmark presented by Cassimon et al., producing policies resulting in the user's preferred outcome.

## Getting started

### Prerequisites

* For Python, follow the insctructions from [Python's official website.](https://www.python.org/downloads/)

* For Git on Windows, follow the instructions from [Git's official website.](https://gitforwindows.org/)

Note: the code is designed in a Windows environment using the PyPI package installer due to the pip install commands included in the command line script. Using this script in other environments may induce errors.

### Usage

1. Make sure the prerequisites are installed
2. Clone or download the repository
3. Run `sh commands.sh` in the command line in the root directory
4. Success!

## Structure

The repository is partiotioned into the following files:

* The `Pipeline` folder contains the proposed algorithm along with all the required Python scripts for it to run. The scripts contain their own documentation.
  * The `Results` sub-folder contains the outputs of the proposed algorithm.
* The `requirements.txt` contains the required libraries to run the demo.
* The `commands.sh` contains the command line script used to run the testing suite as a whole: it first install the libraries, then executes the testing and finally aggregates the outputs into their own directory.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
