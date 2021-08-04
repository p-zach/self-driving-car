# Author: Porter Zach
# Python 3.9

from math import cos, sin, pi
import numpy as np
import pygame
from pygame.locals import QUIT, K_LEFT, K_RIGHT, K_UP, K_DOWN
import courses
from ray import cast
from car import Car

class Environment:
    def __init__(self, course_name):
        pygame.display.init()
        self.fps = 60
        self.clock = pygame.time.Clock()

        self.course, self.course_img, self.start = courses.import_course(course_name)

        self.display = pygame.display.set_mode(self.course_img.get_rect().size)

        self.car = Car(self.start, "car.png")

        self.entities = pygame.sprite.Group()
        self.entities.add(self.car)

        # max_detect_distance = 200

    def rotate(self, vec, angle):
        return np.array([vec[0] * cos(angle) - vec[1] * sin(angle), vec[0] * sin(angle) + vec[1] * cos(angle)])

    def get_ray_dirs(self, dir):
        return [
            self.rotate(dir, pi / 2),
            self.rotate(dir, pi / 4),
            dir,
            self.rotate(dir, -pi / 4),
            self.rotate(dir, -pi / 2)
        ]    

    def get_observation(self):
        dir = self.car.dir()
        rays = self.get_ray_dirs(dir)
        # returns velocity, look direction, and distance to 5 ray intersections
        return np.array([self.car.velocity[0], self.car.velocity[1],
            dir[0], dir[1],
            cast(self.car.position, rays[0], self.course),
            cast(self.car.position, rays[1], self.course),
            cast(self.car.position, rays[2], self.course),
            cast(self.car.position, rays[3], self.course),
            cast(self.car.position, rays[4], self.course)])

    def reset(self):
        self.car.reset(self.start)
        return self.get_observation()

    def step(self, actions):
        prev_position = self.car.position.copy()
        self.car.turn(actions[1])
        self.car.accel(actions[0])
        self.car.update()

        observation = self.get_observation()
        reward = np.linalg.norm(self.car.position - prev_position)
        done = any([True if result < 1 else False for result in observation[4:]])

        return observation, reward, done

    def draw(self):
        self.display.fill((0,0,0))
        self.display.blit(self.course_img, (0, 0))

        self.entities.draw(self.display)

        pygame.display.update()

        self.clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == QUIT:
                quit()


# while True:

#     keys = pygame.key.get_pressed()
#     turn, forward = 0, 0
#     if keys[K_UP]:
#         forward += 1
#     if keys[K_DOWN]:
#         forward -= 1
#     if keys[K_LEFT]:
#         turn += 1
#     if keys[K_RIGHT]:
#         turn -= 1

#     car.accel(forward)
#     car.turn(turn)
#     car.update()

#     ray_dirs = get_ray_dirs(car.dir())
#     cast_results = [cast(np.array(car.position), ray_dirs[i], course, True) for i in range(len(ray_dirs))]
#     for result in cast_results:
#         if result[0] < 1:
#             car.reset(start)
#         pygame.draw.line(display, (255, 0, 0), car.position, result[1], 1)