# Author: Porter Zach
# Python 3.9

import pygame
import math
import numpy as np

class Car(pygame.sprite.Sprite):
    def __init__(self, pos: tuple, img: str):
        super().__init__()
        self.image_original = pygame.image.load(img).convert_alpha()
        self.image = self.image_original.copy()
        self.rect = self.image.get_rect(center = pos)

        self.accel_rate = .5
        self.max_speed = 8
        self.max_rot_speed = .08
        self.friction = .99
        
        self.reset(np.array(pos))

    def reset(self, pos):
        self.rotation = 0
        self.turn(0)
        self.position = np.array(pos, np.float64)
        self.velocity = np.array([0, 0])

    def clamp_vec(self, vector, max_mag):
        mag = np.linalg.norm(vector)
        if mag <= max_mag:
            return vector
        else: return vector / mag * max_mag

    def accel(self, amount: float):
        amount += 0.5

        amount *= self.accel_rate
        self.velocity = self.clamp_vec(self.velocity + np.array([amount * math.cos(self.rotation), amount * -math.sin(self.rotation)]), self.max_speed)
        self.position += self.velocity
        self.rect.center = tuple(self.position)

    def turn(self, angle: float):
        self.rotation += angle * self.max_rot_speed
        self.image = pygame.transform.rotate(self.image_original, math.degrees(self.rotation))
        self.rect = self.image.get_rect(center = self.rect.center)

    def update(self):
        self.velocity *= self.friction

    def dir(self):
        return np.array([math.cos(self.rotation), -math.sin(self.rotation)])