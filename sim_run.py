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
h = 0.220  # Si height (um) # 150 nm as in Bilodeau paper

## For each setting
# [ name: str,  blocks: int, w_range: [range of w], ws_range: [range of ws], geometry: lambda (w, ws, blocks): [mp.GeometricObject]]
sc_y = 4
sc_z = 4
h = 0.220  # Si height (um) # 150 nm as in Bilodeau paper
singleblock_geometry = lambda w, _, __: [mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(),  material=Simulation.Si), 
                            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]

multiblock_geometry = lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), 
                                                         center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                                         material=Simulation.Si) for x in range(blocks)],
                                             mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),
                                                      center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2)]
multiblock_m_geometry = lambda w, ws, blocks: [ *[mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0 - (blocks * w + (blocks - 1) * ws) / 2 + x * (w + ws) + w / 2, 0), 
                                                           material=Simulation.Si) for x in range(blocks)],
                                               mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,0.25*sc_z+0.25*h), material=Simulation.SiO2),
                                               mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*sc_z-0.5*h),center=mp.Vector3(0,0,-0.25*sc_z-0.25*h), material=Simulation.SiO2),
                                               mp.Block(size=mp.Vector3(mp.inf, 0.5*sc_y-0.5*(blocks*w+blocks*ws-ws), mp.inf),center=mp.Vector3(0,-0.25*sc_y-0.25*(blocks*w+blocks*ws-ws),0), material=Simulation.SiO2),
                                               mp.Block(size=mp.Vector3(mp.inf, 0.5*sc_y-0.5*(blocks*w+blocks*ws-ws), mp.inf),center=mp.Vector3(0,0.25*sc_y+0.25*(blocks*w+blocks*ws-ws),0), material=Simulation.SiO2)]

w2_range = np.linspace(0.06, 0.3, 25)
ws2_range = np.linspace(0.06, 1.5, 73)

w3_range = np.linspace(0.06, 0.3, 25)
ws3_range = np.linspace(0.06, 1.2, 58)

w4_range = np.linspace(0.06, 0.3, 25)
ws4_range = np.linspace(0.06, 0.8, 38)

settings = [
    ("S1", 1, np.linspace(0.06, 1.6, 78), [0], singleblock_geometry),
     
    ("S2", 2, w2_range, ws2_range, multiblock_geometry),
    ("S2_p", 2, w2_range, np.linspace(0.06, 0.40, 35), multiblock_m_geometry),
    
    ("S3", 3, w3_range, ws3_range, multiblock_geometry),
    ("S3_p", 3, w3_range, np.linspace(0.06, 0.35, 30), multiblock_m_geometry),
    
    ("S4", 4, w4_range, ws4_range, multiblock_geometry),
    ("S4_p", 4, w4_range, np.linspace(0.06, 0.35, 30), multiblock_m_geometry)
]

# %% [markdown]
# ### Execute the settings

# %%
import sys, os
print('Starting simulation with setting index', sys.argv[1])
start = int(sys.argv[1]) if sys.argv[1] and int(sys.argv[1]) >= 0 else 0

def get_path(name):
    return f"./data/sim_{name}.npy"

def simulation_execute(setting, spinner=None):
    # get the start time
    st = time.time()
    sim = Simulation(sc_y=sc_y, sc_z=sc_z)
    (setting_name, blocks, w_range, ws_range, geometry_lambda) = setting
    neffs = np.zeros([len(w_range), len(ws_range), 2, 4])
    total = len(w_range) * len(ws_range)
    curr = 0
    for w_i, w in enumerate(w_range):
        for ws_i, ws in enumerate(ws_range):
            curr += 1
            print(f"Progress {curr} / {total} iterations ... time: {time.time() - st:0.1f}s")
            sys.stdout.flush()
            geometry = geometry_lambda(w, ws, blocks)
            sim.init_mode_solver(geometry, default_material=Simulation.PC1)
            neff = sim.run()
            neffs[w_i][ws_i][0] = neff
            
            sim.init_mode_solver(geometry, default_material=Simulation.PC2)
            neff = sim.run()
            neffs[w_i][ws_i][1] = neff
    
    path = get_path(setting_name)
    np.save(path, neffs)
    if spinner: 
        spinner.succeed(f"Finished simulation with setting {setting_name}, saved to {path}")

from halo import Halo
with Halo(text='Simulating', spinner='dots') as spinner:
    for i, setting in enumerate(settings[start:]):
        spinner.info(f"Simulating {i}. {setting[0]} ...")
        simulation_execute(setting, spinner)
                        
                        


