import math
from ROOT import TEllipse, TColor
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

"""These helper functions implement three commonly used functionalities:
The Object Selection Helpers represent standard object selections that serve as a starting point for
self defined object selection strategies.
The selectAndSortContainer function can be used to do selecting and sorting in a one liner.
The StandardEventCuts function implements a standard cut used in essentially all analyses.
"""


# Object Selection Helpers
def isGoodPhoton(Photon):
    if not Photon.isTightID(): return False
    if not Photon.pt() > 25: return False
    if not Photon.isoetconerel20() < 0.15: return False
    if not Photon.isoptconerel30() < 0.15: return False
    return True;

def isGoodLepton(Lepton):
    if (abs(Lepton.pdgId()) == 11 and isGoodElectron(Lepton)): return True;
    if (abs(Lepton.pdgId()) == 13 and isGoodMuon(Lepton)): return True;
    return False;
    
def isGoodElectron(Lepton):
    if not Lepton.isTightID(): return False
    if not Lepton.pt() > 30: return False
    if not Lepton.isoetconerel20() < 0.15: return False
    if not Lepton.isoptconerel30() < 0.15: return False
    return True;
    
def isGoodMuon(Lepton):
    if not Lepton.isTightID(): return False
    if not Lepton.pt() > 30: return False
    if not Lepton.isoetconerel20() < 0.15: return False
    if not Lepton.isoptconerel30() < 0.15: return False
    return True;

#def oneJet(jet):
#    if jet.
    
def isGoodJet(jet):
    if jet.pt() < 30: return False
    if abs(jet.eta()) > 4.5: return False
    if jet.pt() < 60 and abs(jet.eta()) < 2.4 and jet.jvt() < 0.59: return False
    return True

def isGoodFatJet(FatJet):
    if FatJet.pt() < 250: return False
    if abs(FatJet.eta()) > 2: return False
    if FatJet.m() < 40: return False
    return True

def isGoodTau(Tau):
    if Tau.pt() < 25: return False
    if abs(Tau.eta()) > 2.5: return False
    if not Tau.isTight(): return False
    #if Tau.nTracks != 1 or Tau.nTracks != 3: return False
    #if Tau.BDTid < ?: return False
    return True

# Utility function
def selectAndSortContainer(container, selectingFunction, sortingFunction):
    selectedContainer = [particle for particle in container if selectingFunction(particle)]
    return sorted(selectedContainer, key=sortingFunction, reverse=True)

# Event Selection Helpers
def StandardEventCuts(eventinfo):
    if not (eventinfo.triggeredByElectron() or eventinfo.triggeredByMuon() or eventinfo.triggeredByPhoton() or eventinfo.triggeredByTau() or eventinfo.triggeredByDiTau()): return False
    return True;
    
def mlb(jet,lepton):
    pstop = jet.tlv() + lepton.tlv()
    return pstop.M()
    #return pstop.M()/10.

    
# Variable Definitions:
def WTransverseMass(lepton, etmiss):
    return math.sqrt(2*lepton.pt()*etmiss.et()*(1-math.cos(lepton.tlv().DeltaPhi(etmiss.tlv()))));

# Draw Physics Objects as Circles
def DrawObjectROOT(particle, pt, ptype) :
    
    import math

    phiAxis = pt *2.* math.pi / 224. # Ellypse axis
    etaAxis = pt *9. / 224.

    y = particle.phi()
    try  :
        x = particle.eta()
    except:
        x = 0

    ellipse = TEllipse(x,y,etaAxis, phiAxis)
    if 'mltop' in ptype :
        ellipse.SetLineColor(10000+len(ptype))
        ellipse.SetLineWidth(2)
#        ellipse.SetFillColorAlpha(20000+len(ptype), 0.0) #Este provoca solapamiento, mejor no usar
        ellipse.SetFillStyle(0) # Mejor usar este (circunferencias)
    else :
#        ellipse.SetLineWidth(2)  #Empty
        ellipse.SetLineWidth(0)
#        ellipse.SetLineColor(10000+len(ptype))  #Empty
        ellipse.SetFillColor(20000+len(ptype))
#        ellipse.SetFillStyle(0)  #Empty
        ellipse.SetFillStyle(1001)
        
    return ellipse

## Draw Physics Objects as Circles
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
