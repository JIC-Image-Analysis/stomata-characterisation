"""Basic viewer for scrolling through sorted png images."""

import os
import re

import pygame
pygame.init()
w = 500
h = 500
size=(w,h)
screen = pygame.display.set_mode(size) 
c = pygame.time.Clock() # create a clock object for timing

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

fpaths = [f for f in sorted_nicely(os.listdir('.')) if f.endswith('.png')]


location = 0
img=pygame.image.load(fpaths[location]) 
screen.blit(img,(0,0))
pygame.display.flip() # update the display
print("Now displaying: {}.".format(fpaths[location]))

DONE = False
while not DONE:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                location -= 1
                if location == -1:
                    location = len(fpaths) - 1
                img=pygame.image.load(fpaths[location]) 
                screen.blit(img,(0,0))
                pygame.display.flip() # update the display
                print("Now displaying: {}.".format(fpaths[location]))
            if event.key == pygame.K_RIGHT:
                location += 1
                if location == len(fpaths):
                    location=0
                img=pygame.image.load(fpaths[location]) 
                screen.blit(img,(0,0))
                pygame.display.flip() # update the display
                print("Now displaying: {}.".format(fpaths[location]))
