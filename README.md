
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
├── iccs_model/                                     # Module containing all model logic for the ICCS 2026 model
|   |
|   ├── agents/                                     # Contains agents specifically defined for
|   |                                                 analysis in the ICCS 2026 paper.
|   ├── models/
|   |   ├── DefaultModel.py                         # Model with 15 StandardAgents and 1 Baseline
|   |   └── SleepModel.py                           # Model with 1 SleepAgent and 1 BadSleepAgent
|   |
│   └── dynamics/                                   # Contains logic for state switching and agent updating
|
├── networked_model/                                # Module containing all model logic for the final thesis model
|   |
|   ├── agents/                                     # Contains parameter settings for each agent type.
|   |                                               # MODIFY/ADD HERE TO FINETUNE COEFFICIENTS AND RUN DIFFERENT SCENARIOS
|   |
|   ├── models/                                     # Contains models for different scenarios
|   |   |                                           # MODIFY/ADD HERE TO CHANGE DATA COLLECTION AND COMPONENTS USED
|   |   |
|   |   ├── BaselineModel.py                        # Model with only disconnected baseline Wang et al. agents (no daily schedule)
|   |   └── NetworkedModel.py                       # Model with network effects and daily schedules, can also run baseline agents
|   |
│   └── dynamics/                                   # Contains logic for state switching and agent updating
|                                                   # MODIFY/ADD HERE TO CHANGE AGENT DYNAMICS AND SCHEDULES
|
├── output/                        # Files containing simulation output (make dir manually if you encounter related error)
├── Constants.py                   # Constants used in the model
├── run_baseline_model.py          # Runs the improved model with only baseline (Wang et al.) agents without schedule
├── run_model.py                   # Runs the default ICCS sleep deprivation model based on input
├── run_network_model.py           # Contains code for running and analyzing the improved model with network effects
├── sleep_experiment.py            # Runs the ICCS SleepModel, used for the submitted paper
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

To run the model as it was developed for ICCS 2026:

```bash
  cd src
  python run_model.py
```

To run an optimized version of the baseline Wang et al. model:

```bash
  cd src
  python run_baseline_model.py
```

```run_network_model.py``` contains many functions that are useful for running and testing the full extended model, but is still in development. Run it and copy from it at your own discretion.

## Authors

- [Mikko Brandon](https://www.github.com/MikkoBra)

