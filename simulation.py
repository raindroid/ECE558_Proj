import numpy as np
import matplotlib.pyplot as plt

# Other utilities
import time
from IPython.utils import io

with io.capture_output() as captured:
    import meep as mp
    from meep import mpb


class Block(object):
    def __init__(self, w=mp.inf, h=mp.inf, n=1, center=mp.Vector3()):
        """General block class to store parameters
        
        Parameters:
        w (float) : width of the object (microns)
        h (float) : heigth of the object (microns)
        n (float) : refractive index of the object
        """
        self.w = w
        self.h = h
        self.n = n
        self.center = center
    
    def __str__(self):
        return f"{self.__class__.__name__}(w={self.w}, h={self.h}, index={self.n})"
    
class Simulation(object):
    
    Si = mp.Medium(index=3.47343) # Silicon index @ 1550 nm
    PC1 = mp.Medium(index=1.552) # Photochromic State 1 index
    PC2 = mp.Medium(index=1.598) # Photochromic State 2 index
    SiO2 = mp.Medium(index=1.444) # SiO2 bottom cladding index
    
    def __init__(self, N=4, wl=1.55, sc_y=4, sc_z=4, res=64):
        """Script returning the first n effective indices of a silicon photonic strip waveguide
    Accounts for material dispersion of the silicon using a Lorentz model

        Args:
            N (int, optional): number of modes to return. Defaults to 1.
            w (float, optional): width of the waveguide. Defaults to 0.5.
            h (float, optional): heigth of the waveguide. Defaults to 0.22.
            wl (float, optional): wavelength of the simulation (microns). Defaults to 1.55.
            sc_y (int, optional): width of the simulation (microns). Defaults to 4.
            sc_z (int, optional): heigth of the simulation (microns). Defaults to 4.
            res (int, optional): resolution of the simulation (pixels/micron). Defaults to 64.
        """
        self.supercell = Block(w=sc_y, h=sc_z)
        self.N = N
        self.wl = wl
        self.resolution = res
        self.ms = None
        
    def init_mode_solver(self, geometry=[], default_material=mp.Medium(index=1.552)):
        # Simulation parameters
        geometry_lattice = mp.Lattice(size=mp.Vector3(0, self.supercell.w, self.supercell.h))
    
        with io.capture_output() as _:
            ms = mpb.ModeSolver(
                geometry_lattice=geometry_lattice,
                geometry=geometry,
                resolution=self.resolution,
                default_material=default_material) # PC is fully covering the waveguide.
            self.ms = ms
        return ms

    def plot(self, figsize=(6,4), dpi=100, cmap='Blues'):
        assert self.ms, "Geometry not setup properly"
        
        with io.capture_output() as _:
            self.ms.init_params(mp.NO_PARITY, True)

        # Waveguide Geometry Visualization
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        n = np.sqrt(self.ms.get_epsilon())

        pos = ax.imshow(n.T, cmap=cmap, interpolation='spline36', 
                        extent=[-self.supercell.w/2,self.supercell.w/2,
                                -self.supercell.h/2,self.supercell.h/2] )
        cbar = fig.colorbar(pos, ax=ax)
        
        cbar.set_label('n')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        plt.show()
        
    def sim_E_field(self, geometry, sources=None, default_material=mp.Medium(index=1.552), amplitude=1.0, component=mp.Ey,
                store_path="./data/ey.npy", cell_x=0):
        cell = mp.Vector3(cell_x, self.supercell.w, self.supercell.h) # just work in 2D for this
        freq = 1 / self.wl

        sources = sources if sources is not None else [
            mp.Source(src=mp.ContinuousSource(freq), center=mp.Vector3(x=0, y=0, z=0),
                      component=component, amplitude=amplitude)] # 1mA source

        pml_layers = [mp.PML(1.0)]

        sim = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            geometry=geometry,
                            default_material=default_material,
                            sources=sources,
                            resolution=self.resolution,
                            force_complex_fields=True)

        # If we want to save the E-field z-components every 0.1 units of time,
        # then, instead of the above, we can run:
        # sim.run(mp.to_appended("ez", mp.at_every(0.1, mp.output_efield_z)),
        #         until=35)

        sim.init_sim()
        sim.solve_cw()
        
        return sim
    
    def run(self):
        f_mode = 1/self.wl   # frequency corresponding to desired wavelength (in um) 
        band_min = 1 
        band_max = self.N
        kdir = mp.Vector3(1) # Our waveguide is along the x direction
        tol = 1e-6 # Iterative solver stop condition
        kmag_guess = f_mode*3.45 # Initial guess
        kmag_min = f_mode*0.1; kmag_max = f_mode*4.0 # Some search range

        # Do the simulation
        with io.capture_output() as _:
            k_fmode = self.ms.find_k(mp.NO_PARITY, f_mode, band_min, band_max, 
                                    kdir, tol, kmag_guess, kmag_min, kmag_max)

        # Compute effective index
        neff = []
        for i in range(self.N):
            neff.append(k_fmode[i] / f_mode)
        
        return neff
    

if __name__ == '__main__':
    sim = Simulation()
    
    h = 0.150  # Si height (um) # 150 nm as in Bilodeau paper
    w = 0.175 # waveguide width (um) # 175 nm as in Bilodeau paper 
    geometry = [mp.Block(size=mp.Vector3(mp.inf, w, h), center=mp.Vector3(), 
                         material=Simulation.Si)]
    
    sim.init_mode_solver(geometry, default_material=Simulation.PC1)
    neff = sim.run()
    print(neff)