# Author: Porter Zach
# Python 3.9

from math import cos, sin, pi
import numpy as np
import pygame
from pygame.locals import QUIT, K_LEFT, K_RIGHT, K_UP, K_DOWN
import courses
from ray import cast
from car import Car

def rotate(vec, angle):
    return np.array([vec[0] * cos(angle) - vec[1] * sin(angle), vec[0] * sin(angle) + vec[1] * cos(angle)])

def get_ray_dirs(dir):
    return [
        rotate(dir, pi / 2),
        rotate(dir, pi / 4),
        dir,
        rotate(dir, -pi / 4),
        rotate(dir, -pi / 2)
    ]    

pygame.display.init()
fps = 60
clock = pygame.time.Clock()

course, course_img, start = courses.import_course("course1.png")

display = pygame.display.set_mode(course_img.get_rect().size)

car = Car(start, "car.png")

entities = pygame.sprite.Group()
entities.add(car)

max_detect_distance = 200

while True:
    display.fill((0,0,0))

    display.blit(course_img, (0, 0))

    keys = pygame.key.get_pressed()
    turn, forward = 0, 0
    if keys[K_UP]:
        forward += 1
    if keys[K_DOWN]:
        forward -= 1
    if keys[K_LEFT]:
        turn += 1
    if keys[K_RIGHT]:
        turn -= 1

    car.accel(forward)
    car.turn(turn)
    car.update()

    ray_dirs = get_ray_dirs(car.dir())
    cast_results = [cast(np.array(car.position), ray_dirs[i], course, True) for i in range(len(ray_dirs))]
    for result in cast_results:
        if result[0] < 1:
            car.reset(start)
        pygame.draw.line(display, (255, 0, 0), car.position, result[1], 1)

    entities.draw(display)

    pygame.display.update()

    clock.tick(fps)

    for event in pygame.event.get():
        if event.type == QUIT:
            quit(), 1