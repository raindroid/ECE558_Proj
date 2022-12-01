# %% [markdown]
# # ECE558 Project
# By Tuo and Yucan

# %% [markdown]
# ## Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt

# Other utilities
import time
from IPython.utils import io

with io.capture_output() as captured:
    import meep as mp
    from meep import mpb

from simulation import *

# %%
## For each setting
# [ name: str,  blocks: int, w_range: [range of w], ws_range: [range of ws], geometry: lambda (w, ws, blocks): [mp.GeometricObject]]
sc_y = 4
sc_z = 4
h = 0.150  # Si height (um) # 150 nm as in Bilodeau paper

settings = [
    ("S1", 1, np.linspace(0.1, 1.6, 76), [0], 
     lambda w, ws, blocks: [mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(), material=Simulation.Si)], ),
    ("S1+SiO2", 1, np.linspace(0.1, 1.6, 76), [0],
     lambda w, ws, blocks: [mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(),  material=Simulation.Si), 
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]),
    
    ("S2", 2, np.linspace(0.1, 0.91, 28),np.linspace(0.02, 0.98, 25),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w /2, 0), 
                                        material=Simulation.Si) for x in range(blocks)] ]),
    ("S2+SiO2_b", 2, np.linspace(0.1, 0.91, 28),np.linspace(0.02, 0.98, 25),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]),
    ("S2+SiO2_m", 2, np.linspace(0.1, 0.91, 28),np.linspace(0.02, 0.98, 25),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            *[mp.Block(size=mp.Vector3(mp.inf, ws, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w + ws / 2, 0), 
                                       material=Simulation.SiO2) for x in range(blocks - 1)] ]),
    ("S2+SiO2_bm", 2, np.linspace(0.1, 0.91, 28),np.linspace(0.02, 0.98, 25),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            *[mp.Block(size=mp.Vector3(mp.inf, ws, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w + ws / 2, 0), 
                                       material=Simulation.SiO2) for x in range(blocks - 1)],
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]),
    
    ("S3", 3, np.linspace(0.1, 0.91, 28), np.linspace(0.02, 0.47, 16),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w /2, 0), 
                                        material=Simulation.Si) for x in range(blocks)] ]),
    ("S3+SiO2_b", 3, np.linspace(0.1, 0.91, 28), np.linspace(0.02, 0.47, 16),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]),
    ("S3+SiO2_m", 3, np.linspace(0.1, 0.91, 28), np.linspace(0.02, 0.47, 16),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            *[mp.Block(size=mp.Vector3(mp.inf, ws, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w + ws / 2, 0), 
                                       material=Simulation.SiO2) for x in range(blocks - 1)] ]),
    ("S3+SiO2_bm", 3, np.linspace(0.1, 0.91, 28), np.linspace(0.02, 0.47, 16),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            *[mp.Block(size=mp.Vector3(mp.inf, ws, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w + ws / 2, 0), 
                                       material=Simulation.SiO2) for x in range(blocks - 1)],
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]),
    
    ("S4", 4, np.linspace(0.1, 0.7, 25), np.linspace(0.02, 0.38, 13),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w /2, 0), 
                                material=Simulation.Si) for x in range(blocks)] ]),
    ("S4+SiO2_b", 4, np.linspace(0.1, 0.7, 25), np.linspace(0.02, 0.38, 13),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]),
    ("S4+SiO2_m", 4, np.linspace(0.1, 0.7, 25), np.linspace(0.02, 0.38, 13),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            *[mp.Block(size=mp.Vector3(mp.inf, ws, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w + ws / 2, 0), 
                                       material=Simulation.SiO2) for x in range(blocks - 1)] ]),
    ("S4+SiO2_bm", 4, np.linspace(0.1, 0.7, 25), np.linspace(0.02, 0.38, 13),
     lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                        material=Simulation.Si) for x in range(blocks)],
                            *[mp.Block(size=mp.Vector3(mp.inf, ws, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w + ws / 2, 0), 
                                       material=Simulation.SiO2) for x in range(blocks - 1)],
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]),
]


# %% [markdown]
# ### Execute the settings

# %%
import asyncio
def simulation_execute(setting, spinner=None):
    sim = Simulation(sc_y=sc_y, sc_z=sc_z)
    (setting_name, blocks, w_range, ws_range, geometry_lambda) = setting
    neffs = np.zeros([len(w_range), len(ws_range), 2, 4])
    for w_i, w in enumerate(w_range):
        for ws_i, ws in enumerate(ws_range):
            geometry = geometry_lambda(w, ws, blocks)
            sim.init_mode_solver(geometry, default_material=Simulation.PC1)
            neff = sim.run()
            neffs[w_i][ws_i][0] = neff
            
            sim.init_mode_solver(geometry, default_material=Simulation.PC2)
            neff = sim.run()
            neffs[w_i][ws_i][1] = neff
    
    path = f"./data/sim_{setting_name}.npy"
    np.save(path, neffs)
    if spinner: 
        spinner.succeed(f"Finished simulation with setting {setting_name}, saved to {path}")

from halo import Halo
with Halo(text='Simulating', spinner='dots') as spinner:
    for i, setting in enumerate(settings[1:]):
        spinner.info(f"Simulating {setting[0]} ...")
        simulation_execute(setting, spinner)
                        


