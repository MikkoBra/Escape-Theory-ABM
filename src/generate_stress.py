from model.models.SocialEffectsModel import SuicideModel
import numpy as np
from tqdm import trange
from pathlib import Path


if __name__=="__main__":
    agent_type = input("Input agent type\n> ")
    T = 0
    # Timestep size
    dt = 1/(24*60)
    model = SuicideModel(dt=dt, n=1, stress_gen=True, parameters={
        "type": agent_type,
    })
    # Days to model
    T = int(input("Enter number of days to model\n> "))
    N_steps = int(T/dt)
    t = np.linspace(0, T, N_steps+1)
    for _ in trange(1, N_steps + 1, desc="Running simulation"):
        model.step()

    # Ensure folder exists
    data_folder = Path("output")
    data_folder.mkdir(parents=True, exist_ok=True)

    agent_df = model.datacollector.get_agent_vars_dataframe()
    filename = f"{agent_type}_stress.csv"

    file_path = data_folder / filename 
    agent_df.to_csv(file_path, index=True)