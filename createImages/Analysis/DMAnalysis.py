import AnalysisHelpers as AH
import ROOT
import Analysis

#======================================================================
        
class DMAnalysis(Analysis.Analysis):
  """Semileptonic DMAnalysis loosely based on the ATLAS analyses of top pair events where 
  one W boson decays to leptons and one decays to hadrons.
  """
  def __init__(self, store):
    super(DMAnalysis, self).__init__(store)

  def initialize(self):
      self.hist_WtMass      =  self.addStandardHistogram("WtMass")

      self.hist_leptn       =  self.addStandardHistogram("lep_n")
      self.hist_leptpt      =  self.addStandardHistogram("lep_pt")
      self.hist_lepteta     =  self.addStandardHistogram("lep_eta")
      self.hist_leptE       =  self.addStandardHistogram("lep_E")
      self.hist_leptphi     =  self.addStandardHistogram("lep_phi")
      self.hist_leptch      =  self.addStandardHistogram("lep_charge")
      self.hist_leptID      =  self.addStandardHistogram("lep_type")
      self.hist_leptptc     =  self.addStandardHistogram("lep_ptconerel30")
      self.hist_leptetc     =  self.addStandardHistogram("lep_etconerel20")
      self.hist_lepz0       =  self.addStandardHistogram("lep_z0")
      self.hist_lepd0       =  self.addStandardHistogram("lep_d0")

      self.hist_njets       =  self.addStandardHistogram("n_jets")       
      self.hist_jetspt      =  self.addStandardHistogram("jet_pt")       
      self.hist_jetm        =  self.addStandardHistogram("jet_m")        
      self.hist_jetJVF      =  self.addStandardHistogram("jet_jvf")      
      self.hist_jeteta      =  self.addStandardHistogram("jet_eta")      
      self.hist_jetmv1      =  self.addStandardHistogram("jet_MV1")      

      self.hist_etmiss      = self.addStandardHistogram("etmiss")
      self.hist_vxp_z       = self.addStandardHistogram("vxp_z")
      self.hist_pvxp_n      = self.addStandardHistogram("pvxp_n")
  
  
  def analyze(self):

      import math
      import matplotlib.pyplot as plt

      # retrieving objects
      eventinfo = self.Store.getEventInfo()
      weight = eventinfo.scalefactor()*eventinfo.eventWeight() if not self.getIsData() else 1
      self.countEvent("all", weight)

      # apply standard event based selection
      if not AH.StandardEventCuts(eventinfo): return False
      self.countEvent("EventCuts", weight)

      # neutrinos are expected, so cut on missing transverse momentum
      etmiss = self.Store.getEtMiss()
      if not (etmiss.et() > 30.0): return False
      self.countEvent("MET", weight)
      
      # one good lepton from one of the W boson decays is expected, so require exactly one good lepton
      goodLeptons = AH.selectAndSortContainer(self.Store.getLeptons(), AH.isGoodLepton, lambda p: p.pt())
      if not (len(goodLeptons) == 1): return False
      self.countEvent("1 Lepton", weight)

      leadlepton = goodLeptons[0]
      
      # two jets from one of the W boson decays as well as two b-jets from the top pair decays are expected
      goodJets = AH.selectAndSortContainer(self.Store.getJets(), AH.isGoodJet, lambda p: p.pt())
      if not len(goodJets) >= 2: return False
      self.countEvent("Jets", weight)

      # apply the b-tagging requirement using the MV1 algorithm at 80% efficiency
      btags = sum([1 for jet in goodJets if jet.mv1() > 0.7892])
      if not (btags >= 1): return False
      self.countEvent("btags", weight)

      # apply a cut on the transverse mass of the W boson decaying to leptons
      if not (AH.WTransverseMass(leadlepton, etmiss) > 30.0): return False

      # Histograms detailing event information
      self.hist_vxp_z.Fill(eventinfo.primaryVertexPosition(), weight)
      self.hist_pvxp_n.Fill(eventinfo.numberOfVertices(), weight)

      # histograms for the W boson properties
      self.hist_WtMass.Fill(AH.WTransverseMass(leadlepton, etmiss), weight)

      # histograms for missing et
      self.hist_etmiss.Fill(etmiss.et(),weight)  

      # histograms detailing lepton information
      self.hist_leptn.Fill(len(goodLeptons), weight)
      self.hist_leptpt.Fill(leadlepton.pt(), weight)
      self.hist_lepteta.Fill(leadlepton.eta(), weight)
      self.hist_leptE.Fill(leadlepton.e(), weight)
      self.hist_leptphi.Fill(leadlepton.phi(), weight)
      self.hist_leptch.Fill(leadlepton.charge(), weight)
      self.hist_leptID.Fill(leadlepton.pdgId(), weight)
      self.hist_lepz0.Fill(leadlepton.z0(), weight)
      self.hist_lepd0.Fill(leadlepton.d0(), weight)      
      self.hist_leptptc.Fill(leadlepton.isoptconerel30(), weight)
      self.hist_leptetc.Fill(leadlepton.isoetconerel20(), weight)
      
      # histograms detailing jet information
      self.hist_njets.Fill(len(goodJets), weight)
      [self.hist_jetm.Fill(jet.m(), weight) for jet in goodJets]
      [self.hist_jetspt.Fill(jet.pt(), weight) for jet in goodJets]
      [self.hist_jetJVF.Fill(jet.jvf(), weight) for jet in goodJets]
      [self.hist_jeteta.Fill(jet.eta(), weight) for jet in goodJets]
      [self.hist_jetmv1.Fill(jet.mv1(), weight) for jet in goodJets]


      # Create Plot 
 
      # Create the empty figure
      fig = plt.figure(frameon=False)
      fig.set_size_inches(4,4)
      ax = plt.Axes(fig, [0., 0., 1., 1.])
      ax.axis([-3,3,-math.pi,math.pi])
      fig.add_axes(ax)

      # Represent Jets 
      for jet in goodJets :
        scalePt = 3.*self.hist_jetspt.GetXaxis().FindBin(jet.pt())
#        scalePt = math.log(jet.pt(),11./10.)
        if jet.mv1() > 0.7892 :
          
          ax.add_artist(AH.DrawObject(jet, scalePt, "bjet"))
          ax.add_artist(AH.DrawObject(jet, scalePt+AH.mlb(jet,leadlepton), "top"))
        else :
          ax.add_artist(AH.DrawObject(jet, scalePt, "jet"))

      # Represent MET 
      scalePt = 3.*self.hist_etmiss.GetXaxis().FindBin(etmiss.et())
#      scalePt = math.log(etmiss.et(),11./10.)
      ax.add_artist(AH.DrawObject(etmiss, scalePt, "etmiss"))

      # Represent Leptons 
      scalePt = 3.*self.hist_leptpt.GetXaxis().FindBin(leadlepton.pt())
#      scalePt = math.log(leadlepton.pt(),11./10.)        
      if abs(leadlepton.pdgId()) == 11 :
        ax.add_artist(AH.DrawObject(leadlepton, scalePt, "lepton"))
      else :
        ax.add_artist(AH.DrawObject(leadlepton, scalePt, "lepton"))

      #'event_numEvent.jpg'
      import os
      if not os.path.exists(self.getImageDir()): os.makedirs(self.getImageDir())
      image_name = self.getImageDir()+"/"+str(eventinfo.eventNumber())+".jpg"
      fig.savefig(image_name, dpi=56)
      plt.close(fig) 
      del plt, fig

      return True

  def finalize(self):
      pass
