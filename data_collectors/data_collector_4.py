import pandas as pd
from tqdm import tqdm
from model import SSPModel


# Configuration
strategies = ["SSP_Low", "SSP_High", "SSP", "HSP", "RANDOM"]
#strategies = ["SSP_Low", "SSP_High"]
#strategies = ["RANDOM"]
seeds = range(50)  # You can increase this to 10 later
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
            random_seed=seed
        )

        for step in range(steps_per_run):
            model.step()

            results.append({
            "Strategy": strategy,
            "Seed": seed,
            "Step": step,
            "Total_Investigations": model.total_investigations,
            "Detection_Precision": (
                model.detected_serious / model.detected_criminals if model.detected_criminals else 0
            ), 
            "Avg_Detection_Delay": (
                sum(a.detected_step - a.offence_steps[0][1]
                    for a in model.criminal_agents if a.detected_step and a.offence_steps)
                / model.detected_criminals if model.detected_criminals else 0
            ),
            "Avg_Offences_per_Detected": (
                sum(len(a.offence_steps) for a in model.criminal_agents if a.detected_step)
                / model.detected_criminals if model.detected_criminals else 0
            ),
            "Avg_Risk": (
                sum(a.perceived_risk for a in model.criminal_agents if a.is_active)
                / len([a for a in model.criminal_agents if a.is_active])
                if len([a for a in model.criminal_agents if a.is_active]) > 0 else 0
            ),
            "Avg_Escapes": (
                sum(a.escape_count for a in model.criminal_agents if a.is_active)
                / len([a for a in model.criminal_agents if a.is_active])
                if len([a for a in model.criminal_agents if a.is_active]) > 0 else 0
            ),
            "Police_Efficiency": (
                model.detected_serious / model.total_police if model.total_police else 0
            ),
            "Active_Criminals" : (len([a for a in model.criminal_agents if a.is_active])
            ),
            "Detected_Serious": model.detected_serious,
            "Detected_Criminals": model.detected_criminals,
            "Avg_Cost_per_Detection": model.total_investigation_cost / model.detected_serious if model.detected_serious else 0,
            "False_Positives": model.false_positives,
            "Active_Serious_Remaining": len([a for a in model.criminal_agents if a.is_active and a.is_serious_offender]),
            "Total_Serious_Offences": model.total_serious_offences,
            "Total_Minor_Offences": model.total_minor_offences,
            "Total_Offences": model.total_offences,
            "Detections_per_10_Investigations": (model.detected_serious / model.total_investigations * 10) if model.total_investigations else 0,
            "SSP_Resolved_Serious": model.ssp_resolved_serious,
            "Investigation_Success_Rate": model.successful_investigations / model.total_investigations if model.total_investigations else 0,
        })



# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV for later analysis
df_results.to_csv("data/ssp_results.csv", index=False)


