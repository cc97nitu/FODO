import math
import json
import typing

import torch
import torch.nn as nn

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps
from ThinLens.Models import Model


class F0D0Model_single(Model):
    
    def __init__(self, k1f: float = 0.0331, k1d: float = -0.0331, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4,):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity
    
        self.cells = list()
        beamline = list()
        for i in range(1):
            # define elements
            dr = Elements.Drift(1.191, **self.generalProperties)
            drb = Elements.Drift(2.618, **self.generalProperties)
            qf = Elements.Quadrupole(4.0, k1f, **self.generalProperties)
            qd = Elements.Quadrupole(4.0, k1d, **self.generalProperties)
            
            cell = [qf, dr, drb, dr,qd, dr, drb, dr]
            beamline.append(cell)
            
        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return
        
class F0D0Model6(Model):
    
    def __init__(self, k1f: float = 0.0331, k1d: float = -0.0331, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order,)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity
    
        self.cells = list()
        beamline = list()
        for i in range(6):
            # define elements
            dr = Elements.Drift(1.191, **self.generalProperties)
            drb = Elements.Drift(2.618, **self.generalProperties)
            qf = Elements.Quadrupole(4.0, k1f, **self.generalProperties)
            qd = Elements.Quadrupole(4.0, k1d, **self.generalProperties)
            
            cell = [qf, dr, drb, dr,qd, dr, drb, dr]
            beamline.append(cell)
            
        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return
        
class F0D0Model12(Model):
    
    def __init__(self, k1f: float = 0.0331, k1d: float = -0.0331, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order,)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity
    
        self.cells = list()
        beamline = list()
        for i in range(12):
            # define elements
            dr = Elements.Drift(1.191, **self.generalProperties)
            drb = Elements.Drift(2.618, **self.generalProperties)
            qf = Elements.Quadrupole(4.0, k1f, **self.generalProperties)
            qd = Elements.Quadrupole(4.0, k1d, **self.generalProperties)
            
            cell = [qf, dr, drb, dr,qd, dr, drb, dr]
            beamline.append(cell)
            
        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


class RBendLine(Model):
    def __init__(self, angle: float, e1: float, e2: float, dim: int = 4, slices: int = 1, order: int = 2,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # define beam line
        d1 = Elements.Drift(1, **self.generalProperties)
        rb1 = Elements.RBen(0.1, angle, e1=e1, e2=e2, **self.generalProperties)
        d2 = Elements.Drift(1, **self.generalProperties)

        # beam line
        self.elements = nn.ModuleList([d1, rb1, d2])
        self.logElementPositions()
        return


class SIS18_Cell_minimal(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # specify beam line elements
        rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)
        rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)

        d1 = Elements.Drift(0.645, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell_minimal_noDipoles(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # specify beam line elements
        d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, k2f: float = 0, k2d: float = 0,
                 dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # define beam line elements
        rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                             **self.generalProperties)
        rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                             **self.generalProperties)
        rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                             **self.generalProperties)
        rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                             **self.generalProperties)

        # sextupoles
        ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
        ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

        # one day there will be correctors
        hKick1 = Elements.Dummy(0, **self.generalProperties)
        hKick2 = Elements.Dummy(0, **self.generalProperties)
        vKick = Elements.Dummy(0, **self.generalProperties)

        hMon = Elements.Monitor(0.13275, **self.generalProperties)
        vMon = Elements.Monitor(0.13275, **self.generalProperties)

        d1 = Elements.Drift(0.2, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3a = Elements.Drift(6.345, **self.generalProperties)
        d3b = Elements.Drift(0.175, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5a = Elements.Drift(0.195, **self.generalProperties)
        d5b = Elements.Drift(0.195, **self.generalProperties)
        d6a = Elements.Drift(0.3485, **self.generalProperties)
        d6b = Elements.Drift(0.3308, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMon, d6b]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return
        
class SIS18_Lattice_minimal_sa(Model):
    def __init__(self, k1f: float = 0.28339, k1d: float = -0.494471, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
        beamline = list()
        for i in range(12):
            # specify beam line elements
            ALPHA = 15 * 1/57.2958
            LL = 150 * 1/57.2958
            PFR = 0.0#7.3 * 1/57.2958
            rb1 = Elements.RBen(length=2.61799, angle=ALPHA, e1=PFR, e2=PFR,
                                **self.generalProperties)
            rb2 = Elements.RBen(length=2.617993878, angle=ALPHA, e1=PFR, e2=PFR,
                                **self.generalProperties)
            # if i == 0 or i == 1 or i == 2:# or i == 3: # or i==1 or i==2: #or i == 5 or i == 9:
            #     qs1f = Elements.QuadrupoleTripplet(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            # else:
            #     qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            # if i == 0 or i == 1 or i == 2: #or i == 3:
            #     #or i==2 :# or i == 1 or i == 2:
            #     qs2d = Elements.QuadrupoleTripplet(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            # else:
            #     qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            # if i == 0 or i == 1 or i == 2: #or i == 5 or i == 9 :
            #     qs3t = Elements.QuadrupoleTripplet(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
            # else:
            #     qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
                
            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.QuadrupoleTripplet(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
            #qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
            
            
            d1 = Elements.Drift(0.645, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3 = Elements.Drift(6.8390117, **self.generalProperties)
            d4 = Elements.Drift(0.6000000, **self.generalProperties)
            d5 = Elements.Drift(0.7098000, **self.generalProperties)
            d6 = Elements.Drift(0.4998000, **self.generalProperties)

            cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]
            #cell = [d3, qs1f, d4, qs2d, d5, qs3t, d6, d1, rb1, d2, rb2]
            
            self.cells.append(cell)
            beamline.append(cell)

        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


class SIS18_Lattice_minimal(Model):
    def __init__(self, k1f: float = 2.82632e-01, k1d: float = -4.92e-01, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
        beamline = list()
        for i in range(12):
            # specify beam line elements
            rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                                **self.generalProperties)
            rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                                **self.generalProperties)
            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)

            d1 = Elements.Drift(0.645, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
            d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
            d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
            d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

            cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]
            self.cells.append(cell)
            beamline.append(cell)

        # flatten beamlinem
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return
        

#QS1F: QUADRUPOLE, L=1.04, K1=0.282632;
#QS2D: QUADRUPOLE, L=1.04, K1=-0.492;
#QS3T: QUADRUPOLE, L = 0.4804, K1 = 0.656;

class SIS18_Lattice(Model):
    def __init__(self, k1f: float = 2.82632e-01, k1d: float = -4.92e-01, k2f: float = 0, k2d: float = 0,
                 dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32,
                 cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
        beamline = list()
        if cellsIdentical:
            # specify beam line elements
            rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                 **self.generalProperties)
            rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                 **self.generalProperties)
            rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                 **self.generalProperties)
            rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                 **self.generalProperties)

            # sextupoles
            ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
            ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

            # one day there will be correctors
            hKick1 = Elements.Drift(0, **self.generalProperties)
            hKick2 = Elements.Drift(0, **self.generalProperties)
            vKick = Elements.Drift(0, **self.generalProperties)

            hMon = Elements.Monitor(0.13275, **self.generalProperties)
            vMon = Elements.Monitor(0.13275, **self.generalProperties)

            d1 = Elements.Drift(0.2, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3a = Elements.Drift(6.345, **self.generalProperties)
            d3b = Elements.Drift(0.175, **self.generalProperties)
            d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
            d5a = Elements.Drift(0.195, **self.generalProperties)
            d5b = Elements.Drift(0.195, **self.generalProperties)
            d6a = Elements.Drift(0.3485, **self.generalProperties)
            d6b = Elements.Drift(0.3308, **self.generalProperties)

            # quadrupoles shall be sliced more due to their strong influence on tunes
            quadrupoleGeneralProperties = dict(self.generalProperties)
            quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)

            for i in range(12):
                cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a,
                        ks3c,
                        d5b,
                        qs3t, d6a, hMon, vMon, d6b]

                self.cells.append(cell)
                beamline.append(cell)

        else:
            for i in range(12):
                # specify beam line elements
                rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                     **self.generalProperties)
                rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                     **self.generalProperties)
                rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                     **self.generalProperties)
                rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                     **self.generalProperties)

                # sextupoles
                ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
                ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

                # one day there will be correctors
                hKick1 = Elements.Drift(0, **self.generalProperties)
                hKick2 = Elements.Drift(0, **self.generalProperties)
                vKick = Elements.Drift(0, **self.generalProperties)

                hMon = Elements.Monitor(0.13275, **self.generalProperties)
                vMon = Elements.Monitor(0.13275, **self.generalProperties)

                d1 = Elements.Drift(0.2, **self.generalProperties)
                d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
                d3a = Elements.Drift(6.345, **self.generalProperties)
                d3b = Elements.Drift(0.175, **self.generalProperties)
                d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
                d5a = Elements.Drift(0.195, **self.generalProperties)
                d5b = Elements.Drift(0.195, **self.generalProperties)
                d6a = Elements.Drift(0.3485, **self.generalProperties)
                d6b = Elements.Drift(0.3308, **self.generalProperties)

                # quadrupoles shall be sliced more due to their strong influence on tunes
                quadrupoleGeneralProperties = dict(self.generalProperties)
                quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

                qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
                qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
                qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)

                cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a,
                        ks3c,
                        d5b,
                        qs3t, d6a, hMon, vMon, d6b]

                self.cells.append(cell)
                beamline.append(cell)

        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import ThinLens.Maps

    torch.set_printoptions(precision=4, sci_mode=True)

    dtype = torch.double
    dim = 6

    # set up models
    mod1 = SIS18_Cell(dtype=dtype, dim=dim, slices=4, quadSliceMultiplicity=4)

    # show initial twiss
    print("initial twiss")
    twissX0, twissY0 = mod1.getInitialTwiss()
    print(twissX0, twissY0)

    # get tunes
    print("tunes: {}".format(mod1.getTunes()))

    # show twiss
    twiss = mod1.getTwiss()

    plt.plot(twiss["s"], twiss["betx"])
    plt.show()
    plt.close()

    # dump to string
    modelDescription = mod1.toJSON()
    mod1.fromJSON(modelDescription)

    # dump to file
    with open("/dev/shm/modelDump.json", "w") as f:
        mod1.dumpJSON(f)

    with open("/dev/shm/modelDump.json", "r") as f:
        mod1.loadJSON(f)
