from mesa import Model
#from mesa.time import RandomActivation
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random

from agents import CriminalAgent, PoliceAgent, HotspotCellAgent


class SSPModel(Model):
    #def __init__(self, width, height, N_criminals, N_police, strategy_type="SSP"):
    def __init__(self, width, height, N_criminals, N_police, strategy_type="SSP", ssp_bias_for_minor=0.8, ssp_investigate_prob=0.9, ssp_success_prob=0.8, random_seed=None, ssp_memory_window=2):  
        
        #print("Random seed used:", random_seed)

        if random_seed is not None:
          print(f"random_seed={random_seed}")
          random.seed(random_seed)
          
        self.num_agents = N_criminals
        self.grid = MultiGrid(width, height, True)
        #self.schedule = RandomActivation(self)
        self.schedule = BaseScheduler(self)
        self.detected_serious = 0
        self.detected_criminals = 0
        self.false_positives = 0
        self.minor_offences = []
        self.serious_offences = []
        self.serious_offenders = []  
        self.minor_offenders = []    
        self.recent_offenders = {}  # Dictionary to track recent offenders with timestamps
        self.running = True
        self.strategy_type = strategy_type
        self.total_criminals = 0
        self.total_serious = 0
        self.total_police = N_police
        self.total_offences = 0
        self.total_minor_offences = 0
        self.total_serious_offences = 0
        self.total_investigations = 0
        self.successful_investigations = 0
        self.ssp_resolved_serious = 0
        self.ssp_bias_for_minor = ssp_bias_for_minor
        self.criminal_agents = []
        self.total_investigation_cost = 0.0
        self.ssp_investigate_prob = ssp_investigate_prob
        self.ssp_success_prob = ssp_success_prob
        self.ssp_memory_window = ssp_memory_window 

        # Adjust parameters based on strategy type, overriding default values.
        if strategy_type == "SSP_High":
            self.ssp_investigate_prob = 0.95
            self.ssp_success_prob = 0.9
            self.ssp_bias_for_minor = 0.6
            self.ssp_memory_window = 10  

        elif strategy_type == "SSP_Low":
            self.ssp_investigate_prob = 0.7
            self.ssp_success_prob = 0.7
            self.ssp_bias_for_minor = 0.9
            self.ssp_memory_window = 1  


        # Initialize the opportunity structure map.
        self.opportunity_map = {
            (x, y): random.uniform(0.5, 1.5)  # The default is set to 1.0. Higher values facilitate offending, whereas lower values inhibit it.
            for x in range(self.grid.width)
            for y in range(self.grid.height)
        }
        self.hotspot_cells = [] 

        for i in range(self.num_agents):
            #is_serious = random.random() < 0.2
            cover_prob = random.uniform(0.2, 0.6)
            #a = CriminalAgent(i, self, is_serious, cover_prob)
            a = CriminalAgent(i, self, cover_prob)
            self.schedule.add(a)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            self.criminal_agents.append(a)
            self.total_criminals += 1
            a.adjust_propensity_from_history()

            # Initialization: Assign each agent three historical behavior records, with each record independently set as blank, minor, or serious.
            for j in range(3):
                behaviour = random.choices(
                    population=["blank", "minor", "serious"],
                    weights=[0.4, 0.4, 0.2],  
                    k=1
                )[0]
        
                if behaviour == "blank":
                    continue  
        
                t = -3 + j  # Time points of historical offences 


                if behaviour == "serious":
                    a.serious_offence_count += 1
                  
                a.offence_steps.append(((x, y), t, behaviour))
                a.last_offence_type = behaviour  # The most recent behavior type is used by SSP for decision-making.
        
                # Add the latest non-blank behavior to recent_offenders for initial use by SSP.
                if j == 2:
                    self.recent_offenders[a] = 0

            a.adjust_propensity_from_history()



        for i in range(N_police):
            p = PoliceAgent(i + N_criminals, self, strategy_type=self.strategy_type)
            self.schedule.add(p)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(p, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "Detected_Serious": lambda m: m.detected_serious,
                "Minor_Offences": lambda m: len(m.minor_offences)
            }
        )

    def step(self):
        self.schedule.step()
        # At each step, record all current serious offenders.
        serious_offenders = [a for a in self.criminal_agents if a.is_active and a.is_serious_offender]
    
        minor_offenders = [a for a in self.criminal_agents if a.is_active and not a.is_serious_offender]
    
        self.serious_offenders.append(serious_offenders)
        self.minor_offenders.append(minor_offenders)
        self.total_minor_offences += len(self.minor_offences)
        self.total_serious_offences += len(self.serious_offences)
        self.total_offences += len(self.minor_offences) + len(self.serious_offences)
        
        if self.strategy_type == "HSP":
            self.update_hotspots()
        
        self.minor_offences.clear()
        self.serious_offences.clear()


    def update_hotspots(self):
        # Remove old hotspot background agents
        if hasattr(self, 'hotspot_bg_agents'):
            for a in self.hotspot_bg_agents:
                self.grid.remove_agent(a)
            self.hotspot_bg_agents.clear()

        # Count offences per cell based on agent history
        offence_counts = {}
        for agent in self.criminal_agents:
            if agent.is_active:
                for pos, t, _ in agent.offence_steps:
                   if self.schedule.time - t <= 5:
                      offence_counts[pos] = offence_counts.get(pos, 0) + 1

        sorted_cells = sorted(offence_counts.items(), key=lambda item: item[1], reverse=True)
        self.hotspot_counts = dict(sorted_cells)
        self.hotspot_cells = [cell for cell, count in sorted_cells[:9]] if sorted_cells else []
        print(f"[STEP {self.schedule.time}] Hotspot Cells: {self.hotspot_cells}")

        # Add new invisible agents to render background
        self.hotspot_bg_agents = []
        print(f"[STEP {self.schedule.time}] Hotspot Agent Count: {len(self.hotspot_cells)}")
        for i, cell in enumerate(self.hotspot_cells):
            bg = HotspotCellAgent(f"hotspot_{i}_{self.schedule.time}", self, cell)
            self.grid.place_agent(bg, cell)
            bg.pos = cell
            self.schedule.add(bg)
            self.hotspot_bg_agents.append(bg)
        print(f"HOTSPOT CREATED: {[a.pos for a in self.hotspot_bg_agents]}")