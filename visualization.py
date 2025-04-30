from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from agents import CriminalAgent, PoliceAgent, HotspotCellAgent
from model import SSPModel

COST_MINOR_INVESTIGATION = 1
COST_SERIOUS_INVESTIGATION = 3

class SummaryElement(TextElement):
    def render(self, model):
        #serious_ids = [a.unique_id for a in model.criminal_agents if a.is_serious_offender]
        serious_ids = [a.unique_id for a in model.criminal_agents if a.is_active and a.is_serious_offender]
        active_criminals = len([a for a in model.criminal_agents if a.is_active])
        serious_offenders_count = len(model.serious_offenders[-1]) if model.serious_offenders else 0
        #minor_offenders_count = len(model.minor_offenders[-1]) if model.minor_offenders else 0 
        minor_offenders_count = active_criminals - len(serious_ids)
        inactive_criminals = model.total_criminals - active_criminals
        precision = (model.detected_serious / model.detected_criminals) if model.detected_criminals else 0
        serious_detected_ratio = (model.detected_serious / model.total_serious) if model.total_serious else 0
        investigation_success_rate = (model.successful_investigations / model.total_investigations) if model.total_investigations else 0
        detection_delays = [a.detected_step - a.offence_steps[0][1] for a in model.criminal_agents if a.detected_step and a.offence_steps]
        avg_detection_delay = sum(detection_delays) / len(detection_delays) if detection_delays else 0
        avg_offences_per_detected = sum(len(a.offence_steps) for a in model.criminal_agents if a.detected_step) / model.detected_criminals if model.detected_criminals else 0
        detections_per_10_inv = (model.detected_serious / model.total_investigations * 10) if model.total_investigations else 0
        avg_cost_per_detection = (model.total_investigation_cost / model.detected_serious) if model.detected_serious else 0
        police_efficiency = (model.detected_serious / model.total_police) if model.total_police else 0
        missed_serious = len([a for a in model.criminal_agents if a.is_active and a.is_serious_offender and len(a.offence_steps) == 0])
        avg_risk = sum(a.perceived_risk for a in model.criminal_agents if a.is_active) / active_criminals if active_criminals else 0
        avg_escapes = sum(a.escape_count for a in model.criminal_agents if a.is_active) / active_criminals if active_criminals else 0
        serious_this_step = len(model.serious_offences)
        minor_this_step = len(model.minor_offences)
        dynamic_total_serious = len([a for a in model.criminal_agents if a.is_serious_offender])
        #remaining_serious = len([a for a in model.criminal_agents if a.is_active and a.is_serious_offender])
        serious_detected_ratio = model.detected_serious / dynamic_total_serious if dynamic_total_serious else 0
        # clear step-based offence lists AFTER rendering
        model.minor_offences.clear()
        model.serious_offences.clear()





        table_html = f"""
        <table border='1' style='border-collapse: collapse; font-size: 12px;'>
            <tr><th>Indicator</th><th>Value</th></tr>
            <tr><td>Total Serious Offenders</td><td>{len(serious_ids)}</td></tr>
            <tr><td>Total Minor Offenders</td><td>{minor_offenders_count}</td></tr>
            <tr><td>Active Criminals</td><td>{active_criminals}</td></tr>
            <tr><td>Detected Serious</td><td>{model.detected_serious} ({serious_detected_ratio:.1%})</td></tr>
            <tr><td>Detected Criminals</td><td>{inactive_criminals} ({model.detected_criminals / model.total_criminals:.1%})</td></tr>
            <tr><td>Detection Precision</td><td>{precision:.1%}</td></tr>
            <tr><td>SSP Resolved Serious</td><td>{model.ssp_resolved_serious}</td></tr>
            <tr><td>False Positives</td><td>{model.false_positives}</td></tr>
            <tr><td>Avg Offences / Detected</td><td>{avg_offences_per_detected:.2f}</td></tr>
            <tr><td>Avg Detection Delay (steps)</td><td>{avg_detection_delay:.1f}</td></tr>
            <tr><td>Total Investigations</td><td>{model.total_investigations}</td></tr>
            <tr><td>Investigation Success Rate</td><td>{investigation_success_rate:.1%}</td></tr>
            <tr><td>Detections per 10 Investigations</td><td>{detections_per_10_inv:.2f}</td></tr>
            <tr><td>Avg Cost per Detection</td><td>{avg_cost_per_detection:.2f}</td></tr>
            <tr><td>Cost per Minor Investigation</td><td>{COST_MINOR_INVESTIGATION}</td></tr>
            <tr><td>Cost per Serious Investigation</td><td>{COST_SERIOUS_INVESTIGATION}</td></tr>
            <tr><td>Police Efficiency</td><td>{police_efficiency:.2f}</td></tr>
            <tr><td>Unobservable Serious Offenders</td><td>{missed_serious}</td></tr>
            <tr><td>Avg Risk Perception</td><td>{avg_risk:.2f}</td></tr>
            <tr><td>Avg Escapes per Active Criminal</td><td>{avg_escapes:.2f}</td></tr>
            <tr><td>Minor Offences (This Step)</td><td>{minor_this_step}</td></tr>
            <tr><td>Serious Offences (This Step)</td><td>{serious_this_step}</td></tr>
            <tr><td>Total Minor Offences</td><td>{model.total_minor_offences}</td></tr>
            <tr><td>Total Serious Offences</td><td>{model.total_serious_offences}</td></tr>
            <tr><td>Total Offences (All)</td><td>{model.total_offences}</td></tr>

        </table>
        """
        return table_html

            

def agent_portrayal(agent):
    portrayal = {}

    def is_hotspot(pos):
        try:
            return pos in server.runner.model.hotspot_cells
        except:
            return False

    # Show hotspot background
    if isinstance(agent, HotspotCellAgent):
        pos = agent.pos
        count = agent.model.hotspot_counts.get(pos, 0)
        max_count = max(agent.model.hotspot_counts.values(), default=1)
        intensity = min(255, int(255 * (count / max_count)))
        hex_color = f"#ff{255 - intensity:02x}{204 - intensity // 2:02x}"
        portrayal = {
            "Shape": "rect",
            "Color": hex_color,
            "Filled": "true",
            "Layer": 0,
            "w": 0.85,
            "h": 0.85
        }
        return portrayal

    
    elif isinstance(agent, PoliceAgent):
        color = {
            "SSP": "blue",
            "HSP": "purple",
            "RANDOM": "green"
        }.get(agent.strategy_type, "blue")

        portrayal = {
            "Shape": "rect",
            "Color": color,
            "w": 0.4,
            "h": 0.4,
            "xAlign": 0.25,
            "yAlign": 0.25,
            "Layer": 3,
            "text": f"{agent.strategy_type[:3]}",
            "text_color": "white"
        }
        return portrayal

    # Show criminals
    if isinstance(agent, CriminalAgent):
        color = "red" if agent.is_serious_offender else "gray"
        label = "Serious" if agent.is_serious_offender else "Minor"
        portrayal = {
            "Shape": "circle",
            "Color": color,
            "r": 0.35,
            "Layer": 1,
            "text": label,
            "text_color": "white"
        }
        return portrayal

    

grid = CanvasGrid(agent_portrayal, 100, 100, 500, 500)
chart = ChartModule([
    {"Label": "Detected_Serious", "Color": "Black"},
    {"Label": "Minor_Offences", "Color": "Orange"}
])
summary = SummaryElement()

server = ModularServer(
    SSPModel,
    [summary, grid, chart],
    "SSP Simulation",
    {
        "width": 100,
        "height": 100,
        "N_criminals": 100,
        "N_police": 20,
        "strategy_type": UserSettableParameter(
            "choice",
            "Patrol Strategy",
            value="SSP",
            choices=["SSP", "HSP", "RANDOM"]
        ),
        "ssp_bias_for_minor": UserSettableParameter(
            "slider",
            "SSP Bias for Minor Offences",
            value=0.8,
            min_value=0.0,
            max_value=1.0,
            step=0.05
        ),
        "random_seed": UserSettableParameter(
            "slider",
            "Random Seed (for reproducibility)",
            value=42,
            min_value=0,
            max_value=1000,
            step=1
        ),
        "width": UserSettableParameter(
            "slider",
            "Grid Width",
            value=100,
            min_value=5,
            max_value=250,
            step=1
        ),
        "height": UserSettableParameter(
            "slider",
            "Grid Height",
            value=100,
            min_value=5,
            max_value=250,
            step=1
        ),
        "N_criminals": UserSettableParameter(
            "slider",
            "Number of Criminals",
            value=100,
            min_value=10,
            max_value=120,
            step=5
        ),
        "N_police": UserSettableParameter(
            "slider",
            "Number of Police",
            value=20,
            min_value=1,
            max_value=50,
            step=1
        )
    }
)

# server.launch()
