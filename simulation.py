import sys
import pickle
import random
import pygame
import numpy as np
from typing import List, Dict, Tuple

from agent import Agent
from resource import Resource
import settings as s
import genetics as gen

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((s.WIDTH, s.HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.fast_mode = False
        
        self.font_ui = pygame.font.SysFont("Consolas", 18)
        self.font_loot = pygame.font.SysFont("Arial", 30, bold=True)
        self.font_info = pygame.font.SysFont("Arial", 18)

        self.agents: List[Agent] = []
        self.resources: List[Resource] = []
        self.stockpiles: Dict[Tuple[int, int, int], int] = {
            s.GREEN: 0,
            s.BLUE: 0
        }

        self.generation = 1
        self.frame_count = 0
        
        self.init_agents()
        self.init_resources()

    def init_agents(self) -> None:
        teams_config = [
            {"color": s.GREEN, "res_color": s.ORANGE, "base": (100, s.HEIGHT // 2)},
            {"color": s.BLUE, "res_color": s.LIGHT_BLUE, "base": (s.WIDTH - 100, s.HEIGHT // 2)}
        ]

        for config in teams_config:
            base_pos = config["base"]
            for _ in range(s.NUM_AGENTS // 2):
                spawn_x = base_pos[0] + random.uniform(-40, 40)
                spawn_y = base_pos[1] + random.uniform(-40, 40)
                self.agents.append(Agent(spawn_x, spawn_y, config["color"], config["res_color"], base_pos))

    def init_resources(self) -> None:
        self.resources = [Resource() for _ in range(s.NUM_RESOURCES)]

    def check_collisions(self) -> None:
        for agent in self.agents:
            if agent.carrying_resource:
                dist_to_base = np.linalg.norm(agent.pos - agent.base_pos)
                if dist_to_base < (s.SAFE_ZONE_BASE_RADIUS + s.AGENT_RADIUS):
                    self.handle_deposit(agent)
                continue

            for res in self.resources:
                if res.active:
                    dist = np.linalg.norm(agent.pos - np.array(res.pos))
                    if dist < (s.AGENT_RADIUS + res.radius):
                        res.active = False
                        agent.carrying_resource = True
                        agent.color = agent.color_resource
                        break

    def handle_deposit(self, agent: Agent) -> None:
        self.stockpiles[agent.team] += 1
        agent.carrying_resource = False
        agent.color = agent.team
        agent.reward_deposit()

    def check_raids(self) -> None:
        base_green_pos = np.array((100, s.HEIGHT // 2))
        base_blue_pos = np.array((s.WIDTH - 100, s.HEIGHT // 2))

        for agent in self.agents:
            if not agent.active or agent.carrying_resource:
                continue

            enemy_team = s.BLUE if agent.team == s.GREEN else s.GREEN
            target_base_pos = base_blue_pos if agent.team == s.GREEN else base_green_pos
            
            dist = np.linalg.norm(agent.pos - target_base_pos)

            if dist < s.SAFE_ZONE_BASE_RADIUS and self.stockpiles[enemy_team] > 0:
                self.stockpiles[enemy_team] -= 1
                agent.carrying_resource = True
                agent.color = agent.color_resource
                agent.fitness += s.RAID_REWARD
                agent.raids_successful += 1
                agent.energy = min(agent.energy + 50.0, s.INITIAL_ENERGY)

    def resolve_agent_collisions(self) -> None:
        n = len(self.agents)
        min_dist_sq = (s.AGENT_RADIUS * 2) ** 2
        
        for i in range(n):
            a1 = self.agents[i]
            if not a1.active: continue

            for j in range(i + 1, n):
                a2 = self.agents[j]
                if not a2.active: continue

                delta = a1.pos - a2.pos
                dist_sq = np.dot(delta, delta)

                if 0 < dist_sq < min_dist_sq:
                    dist = np.sqrt(dist_sq)
                    overlap = (s.AGENT_RADIUS * 2) - dist
                    correction = (delta / dist) * (overlap * 0.5)
                    
                    a1.pos += correction
                    a2.pos -= correction

    def update(self) -> None:
        self.frame_count += 1
        if self.frame_count >= s.EPOCH_DURATION:
            self.next_generation()
            return

        team_green = [a for a in self.agents if a.team == s.GREEN and a.active]
        team_blue = [a for a in self.agents if a.team == s.BLUE and a.active]

        for agent in self.agents:
            if not agent.active: continue
            
            enemies = team_blue if agent.team == s.GREEN else team_green
            neighbors = [a for a in self.agents if a is not agent and a.active]
            
            agent.update(self.resources, enemies, neighbors)
        
        self.resolve_agent_collisions()
        self.check_collisions()
        self.respawn_resources()
        self.check_raids()

    def respawn_resources(self) -> None:
        self.resources = [r for r in self.resources if r.active]
        if len(self.resources) < s.NUM_RESOURCES:
            if random.random() < s.RESOURCE_RESPAWN_RATE:
                self.resources.append(Resource())

    def next_generation(self) -> None:
        print(f"--- FINE GENERAZIONE {self.generation} ---")
        
        team_green = [a for a in self.agents if a.team == s.GREEN]
        team_blue = [a for a in self.agents if a.team == s.BLUE]
        
        avg_fit_g = np.mean([a.fitness for a in team_green]) if team_green else 0
        avg_fit_b = np.mean([a.fitness for a in team_blue]) if team_blue else 0
        print(f"Fitness Media -> VERDI: {avg_fit_g:.2f} | BLU: {avg_fit_b:.2f}")

        brains_green = gen.evolve_population(team_green)
        brains_blue = gen.evolve_population(team_blue)
        
        self.agents.clear()
        self.resources.clear()
        
        base_green_pos = (100, s.HEIGHT // 2)
        base_blue_pos = (s.WIDTH - 100, s.HEIGHT // 2)
        
        self.repopulate(brains_green, s.GREEN, s.ORANGE, base_green_pos)
        self.repopulate(brains_blue, s.BLUE, s.LIGHT_BLUE, base_blue_pos)
        
        self.init_resources()
        self.frame_count = 0
        self.generation += 1
        self.stockpiles = {s.GREEN: 0, s.BLUE: 0}

    def repopulate(self, brains: list, team_color, res_color, base_pos) -> None:
        for brain in brains:
            spawn_x = base_pos[0] + random.uniform(-40, 40)
            spawn_y = base_pos[1] + random.uniform(-40, 40)
            new_agent = Agent(spawn_x, spawn_y, team_color, res_color, base_pos)
            new_agent.brain = brain
            self.agents.append(new_agent)

    def draw_controls_gui(self) -> None:
        controls = [
            "COMANDI:",
            "[S] Salva Simulazione",
            "[L] Carica Simulazione",
            "[K] Esporta Cervelli",
            "[R] Riavvia con Cervelli",
            "[TAB] Turbo Mode",
            "[ESC] Esci"
        ]
        
        start_x, start_y = 10, s.HEIGHT - 150
        line_height = 18
        padding = 5
        
        max_width = max([self.font_ui.size(line)[0] for line in controls])
        total_height = len(controls) * line_height
        
        bg_surface = pygame.Surface((max_width + padding * 2, total_height + padding * 2), pygame.SRCALPHA)
        bg_surface.fill((240, 240, 240, 200))
        self.screen.blit(bg_surface, (start_x - padding, start_y - padding))
        
        for i, line in enumerate(controls):
            color = (50, 50, 150) if i == 0 else (20, 20, 20)
            text_surf = self.font_ui.render(line, True, color)
            self.screen.blit(text_surf, (start_x, start_y + i * line_height))

    def draw(self) -> None:
        self.screen.fill(s.WHITE)
        
        base_green = (100, s.HEIGHT // 2)
        base_blue = (s.WIDTH - 100, s.HEIGHT // 2)
        
        pygame.draw.circle(self.screen, (200, 200, 200), base_green, s.SAFE_ZONE_BASE_RADIUS, 1)
        pygame.draw.circle(self.screen, (200, 200, 200), base_blue, s.SAFE_ZONE_BASE_RADIUS, 1)
        
        text_g = self.font_loot.render(str(self.stockpiles[s.GREEN]), True, s.GREEN)
        self.screen.blit(text_g, (base_green[0] - 10, base_green[1] - 15))
        
        text_b = self.font_loot.render(str(self.stockpiles[s.BLUE]), True, s.BLUE)
        self.screen.blit(text_b, (base_blue[0] - 10, base_blue[1] - 15))

        for res in self.resources:
            res.draw(self.screen)

        for agent in self.agents:
            agent.draw(self.screen)

        info_text = f"Gen: {self.generation} | Frame: {self.frame_count}/{s.EPOCH_DURATION}"
        self.screen.blit(self.font_info.render(info_text, True, (0, 0, 0)), (10, 10))
        
        self.draw_controls_gui()
        
        if self.fast_mode:
            turbo_text = self.font_ui.render(">>> TURBO MODE <<<", True, (255, 0, 0))
            self.screen.blit(turbo_text, (s.WIDTH // 2 - 80, 10))
            
        pygame.display.flip()

    def events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: self.running = False
                elif event.key == pygame.K_s: self.save_simulation()
                elif event.key == pygame.K_l: self.load_simulation()
                elif event.key in [pygame.K_t, pygame.K_TAB]:
                    self.fast_mode = not self.fast_mode
                    print(f"Turbo Mode: {self.fast_mode}")

    def save_simulation(self, filename="checkpoint.pkl") -> None:
        try:
            state = {
                'generation': self.generation,
                'frame_count': self.frame_count,
                'agents': self.agents,
                'resources': self.resources,
                'stockpiles': self.stockpiles
            }
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
            print(f"--- Salvato in {filename} ---")
        except Exception as e:
            print(f"Errore salvataggio: {e}")

    def load_simulation(self, filename="checkpoint.pkl") -> None:
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            self.generation = state['generation']
            self.frame_count = state['frame_count']
            self.agents = state['agents']
            self.resources = state['resources']
            self.stockpiles = state.get('stockpiles', {s.GREEN: 0, s.BLUE: 0})
            print(f"--- Caricato stato Gen {self.generation} ---")
        except Exception as e:
            print(f"Errore caricamento: {e}")

    def run(self) -> None:
        while self.running:
            self.events()
            loops = s.STEP_PER_FRAME_TURBO if self.fast_mode else s.STEP_PER_FRAME
            for _ in range(loops):
                self.update()
            self.draw()
            if not self.fast_mode:
                self.clock.tick(s.FPS)

if __name__ == "__main__":
    sim = Simulation()
    sim.run()