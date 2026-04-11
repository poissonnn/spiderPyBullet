import pybullet as p
import numpy as np
import threading


def simulate():
    p.connect(p.DIRECT)
    # Create a simulation
    p.createMultiBody(1, p.createCollisionShape(p.GEOM_SPHERE, radius=0.5))
    # Simulate for 100 steps
    for i in range(100):
        p.stepSimulation()
    # Disconnect from the physics server
    p.disconnect()


threads = []
for i in range(10):
    thread = threading.Thread(target=simulate)
    threads.append(thread)
    thread.start()


for thread in threads:
    thread.join()