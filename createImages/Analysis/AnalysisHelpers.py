import math
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

"""These helper functions implement three commonly used functionalities:
The Object Selection Helpers represent standard object selections that serve as a starting point for
self defined object selection strategies.
The selectAndSortContainer function can be used to do selecting and sorting in a one liner.
The StandardEventCuts function implements three standard cuts used in essentially all analyses.
"""


# Object Selection Helpers
def isGoodLepton(Lepton):
    if (abs(Lepton.pdgId()) == 11 and isGoodElectron(Lepton)): return True;
    if (abs(Lepton.pdgId()) == 13 and isGoodMuon(Lepton)): return True;
    return False;
    
def isGoodElectron(Lepton):
    if not Lepton.isTight(): return False
    if not Lepton.pt() > 25: return False
    if not Lepton.isoetconerel20() < 0.15: return False
    if not Lepton.isoptconerel30() < 0.15: return False
    return True;
    
def isGoodMuon(Lepton):
    if not Lepton.isTight(): return False
    if not Lepton.pt() > 25: return False
    if not Lepton.isoetconerel20() < 0.15: return False
    if not Lepton.isoptconerel30() < 0.15: return False
    return True;
    
def isGoodJet(jet):
    if jet.pt() < 25: return False
    if abs(jet.eta() > 2.5): return False
    if jet.pt() < 50 and abs(jet.eta() < 2.4) and jet.jvf() < 0.5: return False
    return True

def mlb(jet,lepton):
    pstop = jet.tlv() + lepton.tlv()
    return pstop.M()/10.

# Utility function
def selectAndSortContainer(container, selectingFunction, sortingFunction):
    selectedContainer = [particle for particle in container if selectingFunction(particle)]
    return sorted(selectedContainer, key=sortingFunction, reverse=True)

# Event Selection Helpers
def StandardEventCuts(eventinfo):
    if not (eventinfo.triggeredByElectron() or eventinfo.triggeredByMuon()): return False
    if not eventinfo.passGRL(): return False
    if not eventinfo.hasGoodVertex(): return False
    return True;
    
    
# Variable Definitions:
def WTransverseMass(lepton, etmiss):
    return math.sqrt(2*lepton.pt()*etmiss.et()*(1-math.cos(lepton.tlv().DeltaPhi(etmiss.tlv()))));


def DrawObject(particle, pt, ptype) :
    
    import math

    ecolors = {"lepton" : "g", "bjet" : (1, 0, 0), "jet": (1,125./255.,125./255.), "etmiss" : (0,0,0), "top" : "b"}
    fcolors = {"lepton" : "g", "bjet" : (1, 0, 0), "jet": (1,125./255.,125./255.), "etmiss" : (0,0,0), "top" : "None"}

    phiAxis = np.array([pt*2.*math.pi/299.]) # Ellypse axis
    etaAxis = np.array([pt*6./299.])
            
    try :
        center = np.array([particle.eta(),particle.phi()])
    except:
        center = np.array([0.,particle.phi()])

#    Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = "None", edgecolor= ecolors[ptype], lw = 3, alpha=0.70)        
    Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = fcolors[ptype], edgecolor= ecolors[ptype], lw = 3, alpha=0.70)

    
    return Object