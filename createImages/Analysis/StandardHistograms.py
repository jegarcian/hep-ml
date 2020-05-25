"""This file defines standard histograms which can be reused in various analyses.
The ranges of these histograms should accomodate most analyses.
"""

import ROOT
from array import array

def getStandardHistogram(name):

    pt_bins = [0.,10.,20.,30.,50.,100.,250.,500.,1000.,1500.,2000.]
    pt_bins = [0.,20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,250.,300.,350.,400.,450.,500.]


    if (name == "etmiss"):          return ROOT.TH1D("etmiss",          "Missing Transverse Momentum;p_{T,Miss} [GeV];Events", len(pt_bins)-1, array('d', pt_bins))

    if (name == "n_jets"):          return ROOT.TH1D("n_jets",          "Number of Jets;N_{jets};Events", 10, -0.5, 9.5)
    if (name == "jet_pt"):          return ROOT.TH1D("jet_pt",          "Jet Transverse Momentum;p_{T}^{jet} [GeV];Jets", len(pt_bins)-1, array('d', pt_bins))
    if (name == "jet_m"):           return ROOT.TH1D("jet_m",           "Jet Mass; m^{jet} [MeV]; Jets", 20, 0, 50)
    if (name == "jet_jvt"):         return ROOT.TH1D("jet_jvt",         "Jet Vertex Tagger; JVT ; Jets", 20, 0, 1)
    if (name == "jet_eta"):         return ROOT.TH1D("jet_eta",         "Jet Pseudorapidity; #eta^{jet}; Jets", 30, -3, 3)
    if (name == "jet_MV2c10"):      return ROOT.TH1D("jet_MV2c10",      "Jet MV2c10; MV2c10 weight ; Jets", 20, -1, 1)
    if (name == "lep_n"):           return ROOT.TH1D("lep_n",           "Number of Leptons; N_{lep} ;Events", 10, -0.5, 9.5)

    if (name == "lep_pt"):          return ROOT.TH1D("lep_pt",          "Lepton Transverse Momentum;p_{T}^{lep} [GeV];Leptons", len(pt_bins)-1, array('d', pt_bins))
    if (name == "lep_eta"):         return ROOT.TH1D("lep_eta",         "Lepton Pseudorapidity; #eta^{lep}; Leptons", 30, -3, 3)
    if (name == "lep_E"):           return ROOT.TH1D("lep_E",           "Lepton Energy; E^{lep} [GeV]; Leptons", 30, 0, 300)
    if (name == "lep_phi"):         return ROOT.TH1D("lep_phi",         "Lepton Azimuthal Angle ; #phi^{lep}; Leptons", 32, -3.2, 3.2)
    if (name == "lep_charge"):      return ROOT.TH1D("lep_charge",      "Lepton Charge; Q^{lep}; Leptons", 7, -1.75, 1.75)
    if (name == "lep_type"):        return ROOT.TH1D("lep_type",        "Lepton PDG ID; PDGID^{lep}; Leptons", 31, -0.5, 30.5)
    if (name == "lep_ptconerel30"): return ROOT.TH1D("lep_ptconerel30", "Lepton Relative Transverse Momentum Isolation; ptconerel30^{lep}; Leptons", 40, 0, 0.2)
    if (name == "lep_etconerel20"): return ROOT.TH1D("lep_etconerel20", "Lepton Relative Transverse Energy Isolation; etconerel20^{lep}; Leptons", 40,  -0.05, 0.2)
    if (name == "lep_z0"):          return ROOT.TH1D("lep_z0",          "Lepton z0 impact parameter; z_{0}^{lep} [mm]; Leptons", 40, -1, 1)
    if (name == "lep_d0"):          return ROOT.TH1D("lep_d0",          "Lepton d0 impact parameter; d_{0}^{lep} [mm]; Leptons", 40, -1, 1)

    if (name == "leadlep_pt"):          return ROOT.TH1D("leadlep_pt",          "Leading Lepton Transverse Momentum;p_{T}^{leadlep} [GeV];Leptons", len(pt_bins)-1, array('d', pt_bins))
    if (name == "leadlep_eta"):         return ROOT.TH1D("leadlep_eta",         "Leading Lepton Pseudorapidity; #eta^{leadlep}; Leptons", 30, -3, 3)
    if (name == "leadlep_E"):           return ROOT.TH1D("leadlep_E",           "Leading Lepton Energy; E^{leadlep} [GeV]; Leptons", 30, 0, 300)
    if (name == "leadlep_phi"):         return ROOT.TH1D("leadlep_phi",         "Leading Lepton Azimuthal Angle ; #phi^{leadlep}; Leptons", 32, -3.2, 3.2)
    if (name == "leadlep_charge"):      return ROOT.TH1D("leadlep_charge",      "Leading Lepton Charge; Q^{leadlep}; Leptons", 7, -1.75, 1.75)
    if (name == "leadlep_type"):        return ROOT.TH1D("leadlep_type",        "Leading Lepton PDG ID; PDGID^{leadlep}; Leptons",  31, -0.5, 30.5)
    if (name == "leadlep_ptconerel30"): return ROOT.TH1D("leadlep_ptconerel30", "Leading Lepton Relative Transverse Momentum Isolation; ptconerel30^{leadlep}; Leptons", 40, 0, 0.2)
    if (name == "leadlep_etconerel20"): return ROOT.TH1D("leadlep_etconerel20", "Leading Lepton Relative Transverse Energy Isolation; etconerel20^{leadlep}; Leptons", 40, -0.05, 0.2)
    if (name == "leadlep_z0"):          return ROOT.TH1D("leadlep_z0",          "Leading Lepton z0 impact parameter; z_{0}^{leadlep} [mm]; Leptons", 40, -1, 1)
    if (name == "leadlep_d0"):          return ROOT.TH1D("leadlep_d0",          "Leading Lepton d0 impact parameter; d_{0}^{leadlep} [mm]; Leptons", 40, -1, 1)

    if (name == "traillep_pt"):          return ROOT.TH1D("traillep_pt",          "Trailing Lepton Transverse Momentum;p_{T}^{traillep} [GeV];Leptons", 40, 0, 200)
    if (name == "traillep_eta"):         return ROOT.TH1D("traillep_eta",         "Trailing Lepton Pseudorapidity; #eta^{traillep}; Leptons", 30, -3, 3)
    if (name == "traillep_E"):           return ROOT.TH1D("traillep_E",           "Trailing Lepton Energy; E^{traillep} [GeV]; Leptons", 30, 0, 300)
    if (name == "traillep_phi"):         return ROOT.TH1D("traillep_phi",         "Trailing Lepton Azimuthal Angle ; #phi^{traillep}; Leptons", 32, -3.2, 3.2)
    if (name == "traillep_charge"):      return ROOT.TH1D("traillep_charge",      "Trailing Lepton Charge; Q^{traillep}; Leptons", 7, -1.75, 1.75)
    if (name == "traillep_type"):        return ROOT.TH1D("traillep_type",        "Trailing Lepton PDG ID; PDGID^{traillep}; Leptons",  31, -0.5, 30.5)
    if (name == "traillep_ptconerel30"): return ROOT.TH1D("traillep_ptconerel30", "Trailing Lepton Relative Transverse Momentum Isolation; ptconerel30^{traillep} [GeV]; Leptons", 40, 0, 0.2)
    if (name == "traillep_etconerel20"): return ROOT.TH1D("traillep_etconerel20", "Trailing Lepton Relative Transverse Energy Isolation; etconerel20^{traillep} [GeV]; Leptons", 40, -0.05, 0.2)
    if (name == "traillep_z0"):          return ROOT.TH1D("traillep_z0",          "Trailing Lepton z0 impact parameter; z_{0}^{traillep} [mm]; Leptons", 40, -1, 1)
    if (name == "traillep_d0"):          return ROOT.TH1D("traillep_d0",          "Trailing Lepton d0 impact parameter; d_{0}^{traillep} [mm]; Leptons", 40, -1, 1)

    if (name == "photon_n"):                return ROOT.TH1D("photon_n",                "Number of Photons; N_{photon} ;Events", 10, -0.5, 9.5)
    if (name == "leadphoton_pt"):           return ROOT.TH1D("leadphoton_pt",           "Leading Photon Transverse Momentum;p_{T}^{leadphoton} [GeV];Photons", 40, 0, 200)
    if (name == "leadphoton_eta"):          return ROOT.TH1D("leadphoton_eta",          "Leading Photon Pseudorapidity; #eta^{leadphoton}; Photons", 30, -3, 3)
    if (name == "leadphoton_E"):            return ROOT.TH1D("leadphoton_E",            "Leading Photon Energy; E^{leadphoton} [GeV]; Photons", 30, 0, 300)
    if (name == "leadphoton_phi"):          return ROOT.TH1D("leadphoton_phi",          "Leading Photon Azimuthal Angle ; #phi^{leadphoton}; Photons", 32, -3.2, 3.2)
    if (name == "leadphoton_ptconerel30"):  return ROOT.TH1D("leadphoton_ptconerel30",  "Leading Photon Relative Transverse Momentum Isolation; ptconerel30^{leadphoton} [GeV]; Photons", 40, -0.05, 0.2)
    if (name == "leadphoton_etconerel20"):  return ROOT.TH1D("leadphoton_etconerel20",  "Leading Photon Relative Transverse Energy Isolation; etconerel20^{trailphoton} [GeV]; Photons", 40, -0.05, 0.2)
    if (name == "trailphoton_pt"):          return ROOT.TH1D("trailphoton_pt",          "Trailing Photon Transverse Momentum;p_{T}^{trailphoton} [GeV];Photons", 40, 0, 200)
    if (name == "trailphoton_eta"):         return ROOT.TH1D("trailphoton_eta",         "Trailing Photon Pseudorapidity; #eta^{trailphoton}; Photons", 30, -3, 3)
    if (name == "trailphoton_E"):           return ROOT.TH1D("trailphoton_E",           "Trailing Photon Energy; E^{trailphoton} [GeV]; Photons", 30,0, 300)
    if (name == "trailphoton_phi"):         return ROOT.TH1D("trailphoton_phi",         "Trailing Photon Azimuthal Angle ; #phi^{trailphoton}; Photons", 32, -3.2, 3.2)
    if (name == "trailphoton_ptconerel30"): return ROOT.TH1D("trailphoton_ptconerel30", "Trailing Photon Relative Transverse Momentum Isolation; ptconerel30^{trailphoton} [GeV]; Photons", 40, -0.05, 0.2)
    if (name == "trailphoton_etconerel20"): return ROOT.TH1D("trailphoton_etconerel20", "Trailing Photon Relative Transverse Energy Isolation; etconerel20^{trailphoton} [GeV]; Photons", 40, -0.05, 0.2)

    if (name == "fatjet_n"):     return ROOT.TH1D("fatjet_n",     "Number of Fat Jets;N_{fatjets};Events", 10, -0.5, 9.5)
    if (name == "fatjet_pt"):    return ROOT.TH1D("fatjet_pt",    "Fat Jet Transverse Momentum;p_{T}^{fatjet} [GeV];FatJets", 40, 0, 300)
    if (name == "fatjet_m"):     return ROOT.TH1D("fatjet_m",     "Fat Jet Mass; m^{fatjet} [GeV]; FatJets", 30, 0, 300)
    if (name == "fatjet_eta"):   return ROOT.TH1D("fatjet_eta",   "Fat Jet Pseudorapidity; #eta^{fatjet}; FatJets", 30, -3, 3)
    if (name == "fatjet_D2"):    return ROOT.TH1D("fatjet_D2",    "Fat Jet D2; D2 weight ; FatJets", 40, 0, 20)
    if (name == "fatjet_tau32"): return ROOT.TH1D("fatjet_tau32", "Fat Jet tau32; tau32 weight ; FatJets", 20, 0, 1)

    if (name == "tau_n"):            return ROOT.TH1D("tau_n",            "Number of Taus; N_{tau} ;Events", 10, -0.5, 9.5)
    if (name == "tau_pt"):           return ROOT.TH1D("tau_pt",           "Tau Transverse Momentum;p_{T}^{tau} [GeV];Taus", 40, 0, 200)
    if (name == "tau_eta"):          return ROOT.TH1D("tau_eta",          "Tau Pseudorapidity; #eta^{tau}; Taus", 30, -3, 3)
    if (name == "tau_E"):            return ROOT.TH1D("tau_E",            "Tau Energy; E^{tau} [GeV]; Taus", 30, 0, 300)
    if (name == "tau_phi"):          return ROOT.TH1D("tau_phi",          "Tau Azimuthal Angle ; #phi^{tau}; Taus", 32, -3.2, 3.2)
    if (name == "tau_nTracks"):      return ROOT.TH1D("tau_nTracks",      "Tau Number of Tracks; N_{tautracks}; Taus", 5, -0.5, 4.5)
    if (name == "tau_BDTid"):        return ROOT.TH1D("tau_BDTid",        "Tau BDT score; tau BDT score; Taus", 40, 0, 1)
    if (name == "ditau_m"):          return ROOT.TH1D("ditau_m",          "Ditau Mass; M_{#tau#tau} [GeV]; Events", 40, 0, 300)
    if (name == "leadtau_pt"):       return ROOT.TH1D("leadtau_pt",       "Leading Tau Transverse Momentum;p_{T}^{leadtau} [GeV];Taus", 40, 0, 200)
    if (name == "leadtau_eta"):      return ROOT.TH1D("leadtau_eta",      "Leading Tau Pseudorapidity; #eta^{leadtau}; Taus", 30, -3, 3)
    if (name == "leadtau_E"):        return ROOT.TH1D("leadtau_E",        "Leading Tau Energy; E^{leadtau} [GeV]; Taus", 30, 0, 300)
    if (name == "leadtau_phi"):      return ROOT.TH1D("leadtau_phi",      "Leading Tau Azimuthal Angle ; #phi^{leadtau}; Taus", 32, -3.2, 3.2)
    if (name == "leadtau_nTracks"):  return ROOT.TH1D("leadtau_nTracks",  "Leading Tau Number of Tracks; N_{leadtautracks}; Taus", 5, -0.5, 4.5)
    if (name == "leadtau_BDTid"):    return ROOT.TH1D("leadtau_BDTid",    "Leading Tau BDT score; leadtau BDT score; Taus", 40, 0, 1)
    if (name == "trailtau_pt"):      return ROOT.TH1D("trailtau_pt",      "Trailing Tau Transverse Momentum;p_{T}^{trailtau} [GeV];Taus", 40, 0, 200)
    if (name == "trailtau_eta"):     return ROOT.TH1D("trailtau_eta",     "Trailing Tau Pseudorapidity; #eta^{trailtau}; Taus", 30, -3, 3)
    if (name == "trailtau_E"):       return ROOT.TH1D("trailtau_E",       "Trailing Tau Energy; E^{trailtau} [GeV]; Taus", 30, 0, 300)
    if (name == "trailtau_phi"):     return ROOT.TH1D("trailtau_phi",     "Trailing Tau Azimuthal Angle ; #phi^{trailtau}; Taus", 32, -3.2, 3.2)
    if (name == "trailtau_nTracks"): return ROOT.TH1D("trailtau_nTracks", "Trailing Tau Number of Tracks; N_{trailtautracks}; Taus", 5, -0.5, 4.5)
    if (name == "trailtau_BDTid"):   return ROOT.TH1D("trailtau_BDTid",   "Trailing Tau BDT score; trailtau BDT score; Taus", 40, 0, 1)

    if (name == "WtMass"):            return ROOT.TH1D("WtMass",            "Transverse Mass of the W Candidate; M_{T,W} [GeV]; Events", 40, 40, 150)
    if (name == "invMassZ"):          return ROOT.TH1D("invMassZ",          "Invariant Mass of the Z Candidate;M_{ll} [GeV]; Events", 30, 60, 120)
    if (name == "invMassH"):          return ROOT.TH1D("invMassH",          "Invariant Mass of the H Candidate;M_{#gamma#gamma} [GeV]; Events", 30, 95, 155)

    return None
