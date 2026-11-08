import os
import time
import copy
import pygame

import threading
class ThalamusJoystick:
    def __init__(self):
        self.isActive = False
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        pygame.init()

        t1 = threading.Thread(target=self.worker, daemon=True)
        t1.start()

        self.button_status = []
        self.hat_status = [0,0]
        self.analog_status = []

    def get_status(self):
        return  self.hat_status, self.button_status, self.analog_status

    def worker(self):
        self.joy = None
        self.jid = None
        self.name = None
        self.guid = None
        clock = pygame.time.Clock()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

                if event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                    print("Joystick button change.")
                    buttons = self.joy.get_numbuttons()

                    button_status = []
                    for i in range(buttons):
                        button_status.append(self.joy.get_button(i))

                    print("button_status:", button_status)
                    self.button_status = copy.deepcopy(button_status)


                if event.type == pygame.JOYAXISMOTION:
                    axis_status = []
                    for axis_num in range(self.joy.get_numaxes()):
                        axis_status.append(self.joy.get_axis(axis_num))

                    print("axis:", " ".join(f"{v:6.2f}" for v in axis_status))
                    self.analog_status = copy.deepcopy(axis_status)

                if event.type == pygame.JOYHATMOTION:
                    hat_status = self.joy.get_hat(0)
                    print("pad chagne:", hat_status)
                    self.hat_status[0] = hat_status[0]
                    self.hat_status[1] = hat_status[1]

                if event.type == pygame.JOYDEVICEADDED:
                    self.joy = pygame.joystick.Joystick(event.device_index)
                    self.jid = self.joy.get_instance_id()
                    self.name = self.joy.get_name()
                    self.guid = self.joy.get_guid()
                    print(f"Joystick #{self.joy.get_instance_id()} connencted")

                if event.type == pygame.JOYDEVICEREMOVED:
                    print(f"Joystick #{event.instance_id} disconnected")
            clock.tick(30)


