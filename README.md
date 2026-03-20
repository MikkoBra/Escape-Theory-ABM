
# ABM of Suicidality

An Agent-Based Model (ABM) of suicidality. The core dynamics are based on Wang et al. and Engels.
Agent types and model settings were defined for two types of analysis: the effects of sleep deprivation on suicidality,
and social contagion through social media. The first was developed for a paper submitted to the International Conference
of Computational Science 2026, the latter for my Master's thesis project at the University of Amsterdam.


Extensions are currently in development. For the code used to write the paper submitted to ICCS 2026,
you may use the scripts contained in src. The model itself is contained in src/iccs_model.


## Modules
```
src/
├── errors/                                         # Custom error classes
|
├── model/                                          # Module containing all model logic
│   ├── dynamics/                                   # Defines the core dynamics of the models
|   |   ├── parameters/                             # Contains everything related to parameter
|   |   |   |                                         settings
|   |   |   ├── sets/                               # Contains classes representing specific
|   |   |   |                                         parameters and their coefficient settings
|   |   |   ├── AbstractParameters.py               # Abstract class contining all parameters'
|   |   |   |                                         Set classes
|   |   |   ├── DefaultParameters.py                # Implementation of AbstractParameters
|   |   |   |                                         defining default parameter values
|   |   |   ├── DefaultParameterFactory.py          # Class that enables modification of
|   |   |   |                                         specified default coefficient values
|   |   |   └── StateParameters.py                  # Class containing state-related variables
|   |   |                                             that need to be stored between states
|   |   ├── states/                                 # Contains classes defining agent states
|   |   ├── AgentUpdater.py                         # Contains all parameter update rules
|   |   ├── state_registry.py                       # Initializes state objects before simulation
|   |   └── StateManager.py                         # Handles switching between agent states
|   |
|   ├── sleep_deprivation/                          # Contains agents and models specific to
|   |   |                                             analysis of the effects of sleep
|   |   |                                             deprivation on suicidality
|   │   ├── agents/
|   |   |   ├── AgentFactory.py                     # Interface for generating agents
|   |   |   ├── StandardAgent.py                    # Agent with all model extensions
|   |   |   ├── BaselineAgent.py                    # Agent with dynamics from Wang et al.
|   |   |   ├── SleepAgent.py                       # Agent with consistent 8h sleep
|   |   |   └── BadSleepAgent.py                    # Agent with consistent 6h sleep     
|   │   └── models/
|   |       ├── DefaultModel.py                     # Model with 15 StandardAgents and 1 Baseline
|   |       └── SleepModel.py                       # Model with 1 SleepAgent and 1 BadSleepAgent
|   |
│   └── social_contagion/                           # Contains agents and models specific to
|       |                                             analysis of the effects of social media
|       |                                             on social contagion of suicidality
|       ├── agents/                  
|       └── models/
|
├── output/                        # Files containing simulation output
├── Constants.py                   # Constants used in the model
├── run_model.py                   # Runs the default sleep deprivation model based on input
├── sleep_experiment.py            # Runs the SleepModel and generates results for analysis
└── requirements.txt               # Python library requirements for this model
.gitignore
README.md
```
## Run Locally

Clone the project

```bash
  git clone https://github.com/MikkoBra/Escape-Theory-ABM
```

Go to the project directory

```bash
  cd Escape-Theory-ABM/src
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

## Authors

- [Mikko Brandon](https://www.github.com/MikkoBra)

