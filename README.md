
# Escape-Theory-ABM

An Agent-Based Model (ABM) of a small community, simulating the onset of suicidal thoughts with state-based interactions.


## Modules
```
src/
├── errors/                        # Custom error classes
|
├── model/
│   ├── agents/                    # Contains agent classes with unique parameter settings
|   |   ├── StandardAgent.py       # Default agent class with main agent action definitions
|   |   └── VolatileAgent.py       # Agent class with higher volatility
|   |
│   ├── parameters/
|   |   ├── sets/                  # Record classes to contain parameters per update equation
|   |   ├── Parameters.py          # Abstract parameters superclass with getters/setters for all equation sets
|   |   ├── DefaultParameters.py   # Extension of Parameters that initializes with default values for all update equations
|   |   ├── VolatileParameters.py  # Extension of DefaultParameters that sets higher stress sd, and lower suicidal thought and escape behavior thresholds
|   |   └── StateParameters.py     # Class containing parameters required for calculation of state effects and duration
|   |
│   ├── system_updates/            # Location state representations, AgentUpdater with evolution functions
│   └── SuicideModel.py            # Model class that initializes the environment
|
├── output/                        # Files containing output from runs
├── Constants.py                   # Constants used in the model
├── run_model.py                   # Runs the model with input for number of agents and length of simulation
└── requirements.txt               # Python library requirements for this model
.gitignore
README.md
```
## Run Locally

Clone the project

```bash
  git clone https://github.com/MikkoBra/Social-Suicide-ABM
```

Go to the project directory

```bash
  cd Social-Suicide-ABM/src
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the model

```bash
  cd src
  python run_model.py
```
If you want to generate a plot, make sure that the filename of the dataset is correct in run_model.py

## Authors

- [Mikko Brandon](https://www.github.com/MikkoBra)

