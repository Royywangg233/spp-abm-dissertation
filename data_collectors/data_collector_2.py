import pandas as pd
from tqdm import tqdm
from model import SSPModel


# Configuration
strategies = ["SSP", "HSP", "RANDOM"]
#strategies = ["RANDOM"]
seeds = range(10)  # You can increase this to 10 later
steps_per_run = 500
results = []

# Run simulation for each strategy and seed
for strategy in strategies:
    for seed in tqdm(seeds, desc=f"Running {strategy} strategy"):
        model = SSPModel(
            width=100,
            height=100,
            N_criminals=100,
            N_police=20,
            strategy_type=strategy,
            ssp_bias_for_minor=0.8,
            ssp_investigate_prob=0.9,
            ssp_success_prob=0.8,
            random_seed=seed
        )

        for step in range(steps_per_run):
            model.step()

            # Collect target metrics per step
            active_serious = len([a for a in model.criminal_agents if a.is_active and a.is_serious_offender])

            results.append({
                "Strategy": strategy,
                "Seed": seed,
                "Step": step,
                "Detected_Serious": model.detected_serious,
                "Detected_Criminals": model.detected_criminals,
                "False_Positives": model.false_positives,
                "Active_Serious_Remaining": active_serious,
                "Total_Serious_Offences": model.total_serious_offences,
                "Total_Minor_Offences": model.total_minor_offences,
                "Total_Offences": model.total_offences
            })


# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV for later analysis
df_results.to_csv("data/percesion_recall_f1.csv", index=False)


