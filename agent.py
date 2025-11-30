import pygame
import numpy as np
import settings as c
from brain import Brain

class Agent:
    def __init__(self, x: float, y: float, team_color: tuple, team_color_resource: tuple, base_pos: tuple):
        self.pos = np.array([x, y], dtype='float64')
        
        self.vel = np.random.randn(2)
        norm_vel = np.linalg.norm(self.vel)
        if norm_vel > 0:
            self.vel = (self.vel / norm_vel) * c.MAX_SPEED_LIMIT
            
        self.acc = np.zeros(2)
        
        self.team = team_color
        self.color = team_color
        self.color_resource = team_color_resource
        self.base_pos = np.array(base_pos, dtype='float64')
        
        self.health = c.HEALTH
        self.energy = c.INITIAL_ENERGY
        self.carrying_resource = False
        
        self.is_attacking = False
        self.attack_cooldown = 0
        self.attack_damage = c.ATTACK_DAMAGE
        self.attack_range = c.ATTACK_RANGE 

        self.fitness = 0.0
        self.resources_delivered = 0
        self.damage_dealt = 0.0
        self.raids_successful = 0

        self.brain = Brain(c.INPUT_SIZE, c.HIDDEN_SIZE, c.OUTPUT_SIZE)
        self.debug_target = None 

    @property
    def active(self) -> bool:
        return self.health > 0
    
    @property
    def is_home(self) -> bool:
        return np.linalg.norm(self.pos - self.base_pos) < c.SAFE_ZONE_BASE_RADIUS

    def get_state(self, resources: list, enemies: list) -> np.ndarray:
        state_carrying = 1.0 if self.carrying_resource else -1.0
        
        # Wall sensing
        d_left = self.pos[0]
        d_right = c.WIDTH - self.pos[0]
        d_top = self.pos[1]
        d_bottom = c.HEIGHT - self.pos[1]
        
        wall_range = c.FOV_RADIUS
        s_left = max(0.0, 1.0 - (d_left / wall_range))
        s_right = max(0.0, 1.0 - (d_right / wall_range))
        s_top = max(0.0, 1.0 - (d_top / wall_range))
        s_bottom = max(0.0, 1.0 - (d_bottom / wall_range))
        
        # Resource sensing
        dist_res, sin_res, cos_res = 0.0, 0.0, 0.0
        self.debug_target = None
        if not self.carrying_resource:
            closest_res = self._find_closest(resources)
            if closest_res:
                dist_res, sin_res, cos_res = self._calculate_relative_vector(closest_res.pos)
                self.debug_target = closest_res.pos 

        # Base sensing
        dist_base, sin_base, cos_base = self._calculate_relative_vector(self.base_pos)
        
        # Enemy sensing
        dist_enemy, sin_enemy, cos_enemy = 0.0, 0.0, 0.0
        closest_enemy = self._find_closest(enemies) 
        if closest_enemy:
            dist_enemy, sin_enemy, cos_enemy = self._calculate_relative_vector(closest_enemy.pos)
        
        # Enemy Base sensing
        enemy_base_x = c.WIDTH - 100 if self.base_pos[0] < c.WIDTH // 2 else 100
        enemy_base_pos = np.array([enemy_base_x, c.HEIGHT // 2])
        dist_eb, sin_eb, cos_eb = self._calculate_relative_vector(enemy_base_pos)

        return np.array([
                state_carrying,
                dist_res, sin_res, cos_res,
                dist_base, sin_base, cos_base,
                dist_enemy, sin_enemy, cos_enemy,
                dist_eb, sin_eb, cos_eb,
                s_left, s_right, s_top, s_bottom
            ])

    def _find_closest(self, entities: list):
        closest = None
        min_dist = c.FOV_RADIUS 
        
        for entity in entities:
            if not entity.active or entity is self: continue

            d = np.linalg.norm(entity.pos - self.pos)
            if d < min_dist:
                min_dist = d
                closest = entity
        return closest
    
    def reward_deposit(self) -> None:
        self.resources_delivered += 1
        self.fitness += c.DEPOSIT_REWARD
        self.energy = min(self.energy + c.ENERGY_ON_DEPOSIT, c.INITIAL_ENERGY)

    def _calculate_relative_vector(self, target_pos: np.ndarray) -> tuple:
        delta = target_pos - self.pos
        dist = np.linalg.norm(delta)
        
        if dist > c.FOV_RADIUS:
            return 0.0, 0.0, 0.0
            
        proximity = 1.0 - (dist / c.FOV_RADIUS)
        rad_target = np.arctan2(delta[1], delta[0])
        rad_self = np.arctan2(self.vel[1], self.vel[0])
        rad_diff = rad_target - rad_self
        return proximity, np.sin(rad_diff), np.cos(rad_diff)

    def update(self, resources: list, enemies: list, neighbors: list) -> None:
        if not self.active: return

        movement_cost = (np.linalg.norm(self.vel) / c.MAX_SPEED_LIMIT) * c.MOVE_COST_FACTOR
        self.energy -= (c.ENERGY_DECAY_RATE + movement_cost)

        if self.energy <= 0:
            self.die("starvation")
            return
        if self.health <= 0:
            self.die("combat")
            return

        sensors = self.get_state(resources, enemies)
        outputs = self.brain.forward(sensors)
        thrust, turn, attack_trigger = outputs[0], outputs[1], outputs[2]
        
        self.apply_force(thrust, turn)
        
        self.vel += self.acc
        speed = np.linalg.norm(self.vel)
        if speed > c.MAX_SPEED_LIMIT:
            self.vel = (self.vel / speed) * c.MAX_SPEED_LIMIT
        self.pos += self.vel
        self.acc *= 0 
        
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        self.is_attacking = False
        
        if attack_trigger > 0.5:
            self.attack(neighbors)

        self.handle_boundaries()

    def die(self, cause: str) -> None:
        self.health = 0
        self.energy = 0
        if cause == "starvation":
            self.fitness += c.STARVATION_PENALTY
        elif cause == "combat":
            self.fitness += c.DEATH_PENALTY

    def handle_boundaries(self) -> None:
        if self.pos[0] < 0: 
            self.pos[0] = 0; self.vel[0] *= -1
        elif self.pos[0] > c.WIDTH: 
            self.pos[0] = c.WIDTH; self.vel[0] *= -1
            
        if self.pos[1] < 0: 
            self.pos[1] = 0; self.vel[1] *= -1
        elif self.pos[1] > c.HEIGHT: 
            self.pos[1] = c.HEIGHT; self.vel[1] *= -1

    def apply_force(self, thrust: float, turn: float) -> None:
        angle_adj = turn * 0.2 
        c_th, s_th = np.cos(angle_adj), np.sin(angle_adj)
        rotation_matrix = np.array([[c_th, -s_th], [s_th, c_th]])
        
        if np.linalg.norm(self.vel) < 0.1:
             self.vel = np.array([1.0, 0.0])
             
        self.vel = np.dot(self.vel, rotation_matrix)
        
        thrust_magnitude = (thrust + 1) / 2 * 0.5 
        direction = self.vel / (np.linalg.norm(self.vel) + 1e-6)
        self.acc += direction * thrust_magnitude

    def attack(self, neighbors: list) -> None:
        if self.energy < c.ATTACK_ENERGY_COST: return 

        self.energy -= c.ATTACK_ENERGY_COST 
        self.attack_cooldown = c.ATTACK_COOLDOWN
        self.is_attacking = True 
        hit_someone = False

        for target in neighbors:
            dist = np.linalg.norm(self.pos - target.pos)
            if dist < self.attack_range:
                if target.is_home: continue

                if target.team == self.team:
                    self.fitness += c.FRIENDLY_FIRE_PENALTY
                else:
                    target.health -= self.attack_damage
                    self.damage_dealt += self.attack_damage
                    self.fitness += c.ATTACK_REWARD
                    
                    if target.health <= 0 and target.energy > 0:
                        loot = target.energy * 0.5
                        self.energy = min(self.energy + loot, c.INITIAL_ENERGY)
                        self.fitness += c.KILL_REWARD
                hit_someone = True

        if not hit_someone:
            self.fitness -= 1.0

    def draw(self, screen: pygame.Surface) -> None:
        if not self.active:
            pygame.draw.circle(screen, c.BLACK, self.pos.astype(int), c.AGENT_RADIUS)
            return
        
        body_color = self.color_resource if self.carrying_resource else self.color
        pygame.draw.circle(screen, body_color, self.pos.astype(int), c.AGENT_RADIUS)
        
        if self.carrying_resource:
             pygame.draw.circle(screen, (0,0,0), self.pos.astype(int), 2)

        if self.is_attacking:
            pygame.draw.circle(screen, (255, 50, 50), self.pos.astype(int), int(self.attack_range), 5)

        # Health bar
        bar_w, bar_h = 20, 4
        bar_x = self.pos[0] - bar_w // 2
        bar_y = self.pos[1] - c.AGENT_RADIUS - 8 

        pygame.draw.rect(screen, (0, 0, 0), (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2))
        pygame.draw.rect(screen, (200, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        pct = max(0, self.health / c.HEALTH)
        pygame.draw.rect(screen, (50, 200, 50), (bar_x, bar_y, bar_w * pct, bar_h))
        
        if self.debug_target is not None:
             pygame.draw.line(screen, (50, 255, 50), self.pos, self.debug_target, 1)

        if self.is_home:
            pygame.draw.circle(screen, (100, 200, 255), self.pos.astype(int), c.AGENT_RADIUS + 4, 1)