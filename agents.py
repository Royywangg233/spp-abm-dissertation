from mesa import Agent
import random


# HotspotCellAgent: Invisible agent used to render hotspot backgrounds
class HotspotCellAgent(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos

    def step(self):
        pass 

# CriminalAgent: Represents potential offenders in the environment
class CriminalAgent(Agent):
    def __init__(self, unique_id, model, cover_prob):
        super().__init__(unique_id, model)
        self.cover_prob = cover_prob
        self.offending_propensity = random.betavariate(2, 5)
        self.activity_level = random.uniform(0.4, 0.9)
        self.is_active = True
        self.offence_steps = []
        self.detected_step = None
        self.serious_prob = random.uniform(0.3, 0.7)
        self.minor_reward = 1.0
        self.serious_reward = 3.0
        self.risk_sensitivity = random.uniform(0.5, 1.5)
        self.perceived_risk = 0.5
        self.escape_count = 0
        self.cooldown_steps = 0  
        self.serious_offence_count = 0

    def adjust_propensity_from_history(self):
        if self.is_serious_offender:
            # Serious Criminals, more active
            self.activity_level = random.uniform(0.7, 1.0)
            self.offending_propensity = random.betavariate(3, 2)  # Skewed to the right, indicating higher values
            self.serious_prob = random.uniform(0.6, 0.9)

        elif self.serious_offence_count == 1:
            # Some degree of risk exists
            self.activity_level = random.uniform(0.5, 0.8)
            self.offending_propensity = random.betavariate(2, 2)
            self.serious_prob = random.uniform(0.4, 0.6)
        else:
            # they behave more cautiously
            self.activity_level = random.uniform(0.3, 0.6)
            self.offending_propensity = random.betavariate(2, 5)
            self.serious_prob = random.uniform(0.2, 0.5)
          
    def evade_police(self):
        self.escape_count += 1

    @property
    def is_serious_offender(self):
        return self.serious_offence_count >= 2


    def step(self):
        if not self.is_active:
            return

        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1
            return

        #print(f"Agent {self.unique_id} is moving.")
          
        possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = random.choice(possible_moves)
        self.model.grid.move_agent(self, new_position)

        # Detect police officers within a certain range and adjust risk levels based on distance-weighted influence.
        max_radius = 10  
        risk_boost = 0.0
        
        for dx in range(-max_radius, max_radius + 1):
            for dy in range(-max_radius, max_radius + 1):
                new_x, new_y = self.pos[0] + dx, self.pos[1] + dy
                if self.model.grid.out_of_bounds((new_x, new_y)):
                    continue
                distance = abs(dx) + abs(dy)
                if distance == 0 or distance > max_radius:
                    continue
                cell_agents = self.model.grid.get_cell_list_contents([(new_x, new_y)])
                for agent in cell_agents:
                    if isinstance(agent, PoliceAgent):
                        risk_boost += 0.1 / distance  # The closer the distance, the higher the risk
         
        # Slight decay with distance, weighted by proximity to police officers. 
        self.perceived_risk = min(1.0, max(0.1, self.perceived_risk + risk_boost - 0.02)) 
        if random.random() < self.activity_level:
          opp_factor = self.model.opportunity_map.get(self.pos, 1.0)
          risk_factor = self.perceived_risk * self.risk_sensitivity
          expected_serious = self.serious_reward * opp_factor - risk_factor * 3.0
          expected_minor = self.minor_reward * opp_factor - risk_factor * 0.5
          #expected_serious = self.serious_reward - risk_factor * 3.0  
          #expected_minor = self.minor_reward - risk_factor * 0.5
         
          # offender decision model 
          if (expected_serious - expected_minor) > 2.5 and random.random() < self.serious_prob:
              self.model.serious_offences.append(self)
              offence_type = "serious"
              self.cooldown_steps = 20
          elif expected_minor > 1.3:
              self.model.minor_offences.append(self)
              offence_type = "minor"
              self.cooldown_steps = 10
          else:
              return  # No offence
          
          # Recording behaviours
          self.last_offence_type = offence_type  # "minor" or "serious"
          self.model.recent_offenders[self] = self.model.schedule.time
          #self.offence_steps.append((self.pos, self.model.schedule.time))
          self.offence_steps.append((self.pos, self.model.schedule.time, offence_type))

          if offence_type == "serious":
             self.serious_offence_count += 1



# PoliceAgent: Represents law enforcement with configurable patrol strategies
COST_MINOR_INVESTIGATION = 1
COST_SERIOUS_INVESTIGATION = 3

class PoliceAgent(Agent):
    def __init__(self, unique_id, model, strategy_type):
        super().__init__(unique_id, model)
        self.strategy_type = strategy_type
        #self.investigate_prob = 0.7
        #self.success_prob = 0.6
        self.vision_radius = 5
        self.cooldown_steps = 0  
        self.target_pos = None
        #if strategy_type == "SSP":
        if strategy_type.startswith("SSP"):
            self.investigate_prob = model.ssp_investigate_prob
            self.success_prob = model.ssp_success_prob
        else:
            self.investigate_prob = 0.7  # HSP and RANDOM default value
            self.success_prob = 0.6


    def step(self):
        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1
            return
          
        possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = random.choice(possible_moves)
        self.model.grid.move_agent(self, new_position)

        #if self.strategy_type == "SSP":
        if self.strategy_type.startswith("SSP"):
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=self.vision_radius)
            for agent in neighbors:
                #if isinstance(agent, CriminalAgent) and agent.is_active and agent in self.model.recent_offenders and self.model.schedule.time - self.model.recent_offenders[agent] <= 2:
                if isinstance(agent, CriminalAgent) and agent.is_active and agent in self.model.recent_offenders:
                    time_since = self.model.schedule.time - self.model.recent_offenders[agent]
                    if time_since <= self.model.ssp_memory_window:
                        self.model.total_investigations += 1

                        # Using the investigation bias parameter
                        last_type = getattr(agent, "last_offence_type", None)
                        bias_prob = self.model.ssp_bias_for_minor if last_type == "minor" else (1 - self.model.ssp_bias_for_minor)
                        if random.random() < bias_prob:
                        
                            if random.random() < self.investigate_prob:
                                self.model.successful_investigations += 1
                                if random.random() < self.success_prob:
                                    if random.random() > agent.cover_prob:
                                        agent.is_active = False
                                        self.model.grid.remove_agent(agent)
                                        self.model.detected_criminals += 1
                                        agent.detected_step = self.model.schedule.time

                                        # Cost tracking combined with cooldown time integration.
                                        offence_type = getattr(agent, "last_offence_type", None)
                                        
                                        if offence_type == "minor":
                                            self.model.total_investigation_cost += COST_MINOR_INVESTIGATION
                                            self.cooldown_steps = 1
                                        elif offence_type == "serious":
                                            self.model.total_investigation_cost += COST_SERIOUS_INVESTIGATION
                                            self.cooldown_steps = 4
                                        else:
                                            self.model.total_investigation_cost += 1.0
                                            self.cooldown_steps = 1  

                                    
                                        if agent.is_serious_offender:
                                            self.model.detected_serious += 1
                                            if self.strategy_type.startswith("SSP") and getattr(agent, "last_offence_type", None) == "minor":
                                                self.model.ssp_resolved_serious += 1
                                    else:
                                        self.model.false_positives += 1
                                        agent.evade_police()
                        else:
                            continue


        elif self.strategy_type == "HSP":

            if not self.target_pos or self.target_pos not in self.model.hotspot_cells:
                if self.model.hotspot_cells:
                    self.target_pos = random.choice(self.model.hotspot_cells)

            # Move one step toward the target. 
            if self.target_pos:
                current_x, current_y = self.pos
                target_x, target_y = self.target_pos

                dx = target_x - current_x
                dy = target_y - current_y
                step_x = (1 if dx > 0 else -1) if dx != 0 else 0
                step_y = (1 if dy > 0 else -1) if dy != 0 else 0

                # Select the next position, preferring horizontal or vertical moves.
                options = []
                if step_x != 0:
                    options.append((current_x + step_x, current_y))
                if step_y != 0:
                    options.append((current_x, current_y + step_y))
                options = [p for p in options if not self.model.grid.out_of_bounds(p)]

                if options:
                    new_position = random.choice(options)
                    self.model.grid.move_agent(self, new_position)

            else:
                # In the absence of a target, take a random step.
                possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
                new_position = random.choice(possible_moves)
                self.model.grid.move_agent(self, new_position)

            # Investigate nearby offenders.
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=self.vision_radius)
            for agent in neighbors:
                if isinstance(agent, CriminalAgent) and agent.is_active:
                    self.model.total_investigations += 1
                    if random.random() < self.investigate_prob:
                        self.model.successful_investigations += 1
                        if random.random() < self.success_prob:
                            adjusted_cover_prob = min(1.0, agent.cover_prob + 0.2)
                            if random.random() > adjusted_cover_prob:
                                agent.is_active = False
                                self.model.grid.remove_agent(agent)
                                self.model.detected_criminals += 1
                                agent.detected_step = self.model.schedule.time

                                offence_type = getattr(agent, "last_offence_type", None)
                                if offence_type == "minor":
                                    self.model.total_investigation_cost += COST_MINOR_INVESTIGATION
                                    self.cooldown_steps = 1
                                elif offence_type == "serious":
                                    self.model.total_investigation_cost += COST_SERIOUS_INVESTIGATION
                                    self.cooldown_steps = 4
                                else:
                                    self.model.total_investigation_cost += 1.0
                                    self.cooldown_steps = 1
                                
                                if agent.is_serious_offender:
                                    self.model.detected_serious += 1
                                else:
                                    self.model.false_positives += 1
                            else:
                                agent.evade_police()

        elif self.strategy_type == "RANDOM":
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = random.choice(possible_moves)
            self.model.grid.move_agent(self, new_position)
        
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=self.vision_radius)
            for agent in neighbors:
                if isinstance(agent, CriminalAgent) and agent.is_active:
                    if random.random() > 0.5:  # The investigation continues with a 50% chance.
                        continue
                    self.model.total_investigations += 1
                    if random.random() < self.investigate_prob:
                        self.model.successful_investigations += 1
                        if random.random() < self.success_prob:
                            if random.random() > agent.cover_prob:
                                agent.is_active = False
                                self.model.grid.remove_agent(agent)
                                agent.detected_step = self.model.schedule.time
                                self.model.detected_criminals += 1
        
                                offence_type = getattr(agent, "last_offence_type", None)
                                if offence_type == "minor":
                                    self.model.total_investigation_cost += COST_MINOR_INVESTIGATION
                                    self.cooldown_steps = 1
                                elif offence_type == "serious":
                                    self.model.total_investigation_cost += COST_SERIOUS_INVESTIGATION
                                    self.cooldown_steps = 4
                                else:
                                    self.model.total_investigation_cost += 1.0
                                    self.cooldown_steps = 1
                                
                                if agent.is_serious_offender:
                                    self.model.detected_serious += 1
                            else:
                                self.model.false_positives += 1
                                agent.evade_police()
                    break