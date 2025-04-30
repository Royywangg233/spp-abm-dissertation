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
            detected_serious = model.detected_serious
            precision = model.detected_serious / model.detected_criminals if model.detected_criminals else 0
            detections_per_10_inv = (model.detected_serious / model.total_investigations * 10) if model.total_investigations else 0
            police_efficiency = (model.detected_serious / model.total_police) if model.total_police else 0
            ssp_resolved = model.ssp_resolved_serious

            results.append({
                "Strategy": strategy,
                "Seed": seed,
                "Step": step,
                "Detected_Serious": detected_serious,
                "Detection_Precision": precision,
                "Detections_per_10_Investigations": detections_per_10_inv,
                "Police_Efficiency": police_efficiency,
                "SSP_Resolved_Serious": ssp_resolved,
                "Detected_Criminals": model.detected_criminals,
                "False_Positives": model.false_positives,
                "Total_Investigations": model.total_investigations,
                "Investigation_Success_Rate": model.successful_investigations / model.total_investigations if model.total_investigations else 0,
                "Avg_Cost_per_Detection": model.total_investigation_cost / model.detected_serious if model.detected_serious else 0,
                "Total_Serious_Offences": model.total_serious_offences,
                "Total_Minor_Offences": model.total_minor_offences,
                "Total_Offences": model.total_offences
            })


# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV for later analysis
df_results.to_csv("data/strategy_comparison.csv", index=False)


