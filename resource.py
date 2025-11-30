import pygame
import random
import settings as c

class Resource:
    def __init__(self):
        self.x = random.uniform(10, c.WIDTH - 10)
        self.y = random.uniform(10, c.HEIGHT - 10)
        self.pos = (self.x, self.y)
        self.radius = 5
        self.color = c.YELLOW
        self.active = True

    def draw(self, screen: pygame.Surface) -> None:
        if self.active:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)