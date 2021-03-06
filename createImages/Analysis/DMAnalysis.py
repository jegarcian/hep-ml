import AnalysisHelpers as AH
import ROOT
from ROOT import TCanvas, TASImage, TColor
import Analysis
import os, datetime, math

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

def sfloat(value) :
    return "%0.4f" % value

#======================================================================

class DMAnalysis(Analysis.Analysis):
  """DM Monotop Selection loose selection of events where
  one W boson decays to leptons and one decays to hadrons.
  """
  def __init__(self, store):
    super(DMAnalysis, self).__init__(store)

  def initialize(self):
      self.hist_WtMass       =  self.addStandardHistogram("WtMass")
      self.hist_mlb          =  self.addStandardHistogram("mtop")
      self.hist_WtMass_max   =  self.addHistogram("WtMass_max", ROOT.TH1D("WtMass_max", "Transverse Mass of the W Candidate Max; M_{T,W} max [GeV]; Events", 40, 0, 200))
      self.hist_WtMass_min   =  self.addHistogram("WtMass_min", ROOT.TH1D("WtMass_min", "Transverse Mass of the W Candidate Min; M_{T,W} min [GeV]; Events", 40, 0, 200))

      self.hist_leptn        =  self.addStandardHistogram("lep_n")
      self.hist_leptpt       =  self.addStandardHistogram("lep_pt")
      self.hist_lepteta      =  self.addStandardHistogram("lep_eta")
      self.hist_leptE        =  self.addStandardHistogram("lep_E")
      self.hist_leptphi      =  self.addStandardHistogram("lep_phi")
      self.hist_leptch       =  self.addStandardHistogram("lep_charge")
      self.hist_leptID       =  self.addStandardHistogram("lep_type")
      self.hist_leptptc      =  self.addStandardHistogram("lep_ptconerel30")
      self.hist_leptetc      =  self.addStandardHistogram("lep_etconerel20")
      self.hist_lepz0        =  self.addStandardHistogram("lep_z0")
      self.hist_lepd0        =  self.addStandardHistogram("lep_d0")
      self.hist_leptpt_max   =  self.addHistogram("lep_pt_max", ROOT.TH1D("lep_pt_max", "Lepton Transverse Momentum Max;p_{T}^{lep} max [GeV];Leptons", 40, 0, 200))
      self.hist_leptpt_min   =  self.addHistogram("lep_pt_min", ROOT.TH1D("lep_pt_min", "Lepton Transverse Momentum Min;p_{T}^{lep} min [GeV];Leptons", 40, 0, 200))
      self.hist_leptptc_max  =  self.addHistogram("lep_ptconerel30_max", ROOT.TH1D("lep_ptconerel30_max", "Lepton Relative Transverse Momentum Isolation Max; ptconerel30^{lep} max; Leptons", 40, 0, 0.2))
      self.hist_leptptc_min  =  self.addHistogram("lep_ptconerel30_min", ROOT.TH1D("lep_ptconerel30_min", "Lepton Relative Transverse Momentum Isolation Min; ptconerel30^{lep} min; Leptons", 40, 0, 0.2))
      self.hist_leptetc_max  =  self.addHistogram("lep_etconerel20_max", ROOT.TH1D("lep_etconerel20_max", "Lepton Relative Transverse Energy Isolation Max; etconerel20^{lep} max; Leptons", 40, -0.05, 0.2))
      self.hist_leptetc_min  =  self.addHistogram("lep_etconerel20_min", ROOT.TH1D("lep_etconerel20_min", "Lepton Relative Transverse Energy Isolation Min; etconerel20^{lep} min; Leptons", 40, -0.05, 0.2))

      self.hist_njets        =  self.addStandardHistogram("n_jets")
      self.hist_jetspt       =  self.addStandardHistogram("jet_pt")
      self.hist_jetJVT       =  self.addStandardHistogram("jet_jvt")
      self.hist_jeteta       =  self.addStandardHistogram("jet_eta")
      self.hist_jetmv2c10    =  self.addStandardHistogram("jet_MV2c10")
      self.hist_jetspt_max   =  self.addHistogram("jet_pt_max", ROOT.TH1D("jet_pt_max", "Jet Transverse Momentum Max;p_{T}^{jet} max [GeV];Jets", 40, 0, 200))
      self.hist_jetspt_min   =  self.addHistogram("jet_pt_min", ROOT.TH1D("jet_pt_min", "Jet Transverse Momentum Min;p_{T}^{jet} min [GeV];Jets", 40, 0, 200))

      self.hist_etmiss       = self.addStandardHistogram("etmiss")
      self.hist_etmiss_max   = self.addHistogram("etmiss_max", ROOT.TH1D("etmiss_max", "Missing Transverse Momentum Max;p_{T,Miss} max [GeV];Events", 20, 0,200))
      self.hist_etmiss_min   = self.addHistogram("etmiss_min", ROOT.TH1D("etmiss_min", "Missing Transverse Momentum Min;p_{T,Miss} min [GeV];Events", 20, 0,200))

  def analyze(self,createImages=True):

      import math
#      import matplotlib.pyplot as plt
      import numpy as np

      # retrieving objects
      eventinfo = self.Store.getEventInfo()
      weight = eventinfo.scalefactor()*eventinfo.eventWeight()*eventinfo.scalefactorBTAG() if not self.getIsData() else 1
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
      if not len(goodJets) >= 1: return False
      self.countEvent("Jets", weight)

      # apply the b-tagging requirement using the MV2c10 algorithm at 80% efficiency
      btags = sum([1 for jet in goodJets if jet.mv2c10() > 0.8244273])
      if not (btags >= 1): return False
      self.countEvent("btags", weight)

      # apply a cut on the transverse mass of the W boson decaying to leptons
      if not (AH.WTransverseMass(leadlepton, etmiss) > 30.0): return False

      # histograms for the W boson properties
      self.hist_WtMass.Fill(AH.WTransverseMass(leadlepton, etmiss), weight)
      self.hist_WtMass_max.Fill(AH.WTransverseMass(leadlepton, etmiss)*(1+math.sqrt((leadlepton.pt_syst()/leadlepton.pt())*(leadlepton.pt_syst()/leadlepton.pt())+(etmiss.et_syst()/etmiss.et())*(etmiss.et_syst()/etmiss.et()))/2), weight)
      self.hist_WtMass_min.Fill(AH.WTransverseMass(leadlepton, etmiss)*(1-math.sqrt((leadlepton.pt_syst()/leadlepton.pt())*(leadlepton.pt_syst()/leadlepton.pt())+(etmiss.et_syst()/etmiss.et())*(etmiss.et_syst()/etmiss.et()))/2), weight)

      # histograms for missing et
      self.hist_etmiss.Fill(etmiss.et(),weight)
      self.hist_etmiss_max.Fill(etmiss.et()+etmiss.et_syst(),weight)
      self.hist_etmiss_min.Fill(etmiss.et()-etmiss.et_syst(),weight)

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
      self.hist_leptpt_max.Fill(leadlepton.pt()+leadlepton.pt_syst(), weight)
      self.hist_leptpt_min.Fill(leadlepton.pt()-leadlepton.pt_syst(), weight)
      self.hist_leptptc_max.Fill(leadlepton.isoptconerel30_max(), weight)
      self.hist_leptptc_min.Fill(leadlepton.isoptconerel30_min(), weight)
      self.hist_leptetc_max.Fill(leadlepton.isoetconerel20_max(), weight)
      self.hist_leptetc_min.Fill(leadlepton.isoetconerel20_min(), weight)

      # histograms detailing jet information
      self.hist_njets.Fill(len(goodJets), weight)
      [self.hist_jetspt.Fill(jet.pt(), weight) for jet in goodJets]
      [self.hist_jetJVT.Fill(jet.jvt(), weight) for jet in goodJets]
      [self.hist_jeteta.Fill(jet.eta(), weight) for jet in goodJets]
      [self.hist_jetmv2c10.Fill(jet.mv2c10(), weight) for jet in goodJets]
      [self.hist_jetspt_max.Fill(jet.pt()+jet.pt_syst(), weight) for jet in goodJets]
      [self.hist_jetspt_min.Fill(jet.pt()-jet.pt_syst(), weight) for jet in goodJets]


      # Fill Histograms
      for jet in goodJets :
          if jet.mv2c10() > 0.8244273 :
              self.hist_mlb.Fill(AH.mlb(jet,leadlepton),weight)

      saveCSV = False

      if saveCSV :
          try :
              events_file = open("results/"+self.Store.filename.replace("root","csv"),"a+")
          except :
              pass

      csv_row = str(self.Store.dsid)+","+str(eventinfo.eventNumber())

      if not createImages :
        return True

      VERBOSE = False
      if VERBOSE :
        f = open("ttbar.log","a+")

      useROOT = True
      # Create Plot

      if useROOT :

        # Create the empty figure
        ecolors = {"leptons" : TColor(10007, 0., 1.0, 0.), "bjet" : TColor(10004,1., 0., 0.), "jet": TColor(10003,1,0.5,0.5), "etmiss" : TColor(10006,0.,0,0.9), "mltop" : TColor(10005,0.5,0.5,0.5)}
        fcolors = {"leptons" : TColor(20007, 0., 1.0, 0.), "bjet" : TColor(20004,1., 0., 0.), "jet": TColor(20003,1,0.5,0.5), "etmiss" : TColor(20006,0.,0,0.9), "mltop" : TColor(20005,0.,0.,1.)}

        ecol = []
        fcol = []
        for x in ecolors :
          ecol.append(ecolors[x])
          fcol.append(fcolors[x])

        import os
        if not os.path.exists(self.getImageDir()): os.makedirs(self.getImageDir())
        image_name = self.getImageDir()+"/"+str(eventinfo.eventNumber())+".jpg"

        size = 224
        c1 = TCanvas("c1", "c1", size, size)
        c1.Range(-4.5,-math.pi,4.5,math.pi);
        c1.SetCanvasSize(size,size);
        jets = []
        csv_jets = ''

        size_factor = 2.
        # Represent Jets
        for jet in goodJets :
            scalePt = size_factor*self.hist_jetspt.GetXaxis().FindBin(jet.pt())
            if VERBOSE :
                f.write(str(eventinfo.eventNumber())+" : jet :: "+str(jet.pt())+" "+str(jet.phi())+" "+str(jet.eta())+" "+str(scalePt)+"\n")

            if jet.mv2c10() > 0.8244273 :
                # elm = AH.DrawObjectROOT(jet, scalePt+AH.mlb(jet,leadlepton), "mltop")
                # jets.append(elm)
                # jets[-1].Draw()

                elj = AH.DrawObjectROOT(jet, scalePt, "bjet")
                jets.append(elj)
                jets[-1].Draw()
            else :
                elj = AH.DrawObjectROOT(jet, scalePt, "jet")
                jets.append(elj)
                jets[-1].Draw()
            csv_jets += (","+sfloat(jet.pt())+","+sfloat(jet.phi())+","+sfloat(jet.eta())+","+sfloat(jet.e())+","+str(int(jet.mv2c10() > 0.8244273)))

        scalePt = size_factor*self.hist_etmiss.GetXaxis().FindBin(etmiss.et())
        elmet = AH.DrawObjectROOT(etmiss, scalePt, "etmiss")
        elmet.Draw()

        if VERBOSE :
            f.write(str(eventinfo.eventNumber())+" : etmiss :: "+str(etmiss.et())+" "+str(etmiss.phi())+" "+str(scalePt)+"\n")
        csv_met = (","+sfloat(etmiss.et())+" "+sfloat(etmiss.phi()))

        # Represent Leptons
        scalePt = size_factor*self.hist_leptpt.GetXaxis().FindBin(leadlepton.pt())
        if abs(leadlepton.pdgId()) == 11 :
            ell = AH.DrawObjectROOT(leadlepton, scalePt, "leptons")
            ell.Draw()
        else :
            ell = AH.DrawObjectROOT(leadlepton, scalePt, "leptons")
            ell.Draw()
        if VERBOSE :
            f.write(str(eventinfo.eventNumber())+" : lepton :: "+str(leadlepton.pt())+" "+str(leadlepton.phi())+" "+str(leadlepton.eta())+" "+str(scalePt)+"\n")

        csv_lep = (","+sfloat(leadlepton.pt())+","+sfloat(leadlepton.phi())+","+sfloat(leadlepton.eta())+","+ sfloat(leadlepton.e()))

        csv_row += csv_met+csv_lep+csv_jets+"\n"

        if saveCSV :
            events_file.write(csv_row)
            events_file.close()

        '''
        #'event_numEvent.jpg'
        img = TASImage()
        img.FromPad(c1);
        img.SetImageQuality(3)
        img.SetImageCompression(50)
        img.WriteImage(image_name);
'''
        del c1, jets, elj, ell, elmet, ecol, fcol

      else :
        # Create Plot

        import math
        import matplotlib.pyplot as plt

        # Create the empty figure
        fig = plt.figure(frameon=False)
        fig.set_size_inches(4,4)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis([-3,3,-math.pi,math.pi])
        fig.add_axes(ax)

        size_factor = 2.
        # Represent Jets
        for jet in goodJets :
          scalePt = size_factor*self.hist_jetspt.GetXaxis().FindBin(jet.pt())
          # scalePt = math.log(jet.pt(),11./10.)
          if VERBOSE :
                f.write(str(eventinfo.eventNumber())+" : jet :: "+str(jet.pt())+" "+str(jet.phi())+" "+str(jet.eta())+" "+str(scalePt)+"\n")
          if jet.mv2c10() > 0.8244273 :
            ax.add_artist(AH.DrawObject(jet, scalePt, "bjet"))
            ax.add_artist(AH.DrawObject(jet, scalePt+AH.mlb(jet,leadlepton), "top"))
            
          else :
            ax.add_artist(AH.DrawObject(jet, scalePt, "jet"))

        # Represent MET
        scalePt = size_factor*self.hist_etmiss.GetXaxis().FindBin(etmiss.et())
        # scalePt = math.log(etmiss.et(),11./10.)
        ax.add_artist(AH.DrawObject(etmiss, scalePt, "etmiss"))

        if VERBOSE :
            f.write(str(eventinfo.eventNumber())+" : etmiss :: "+str(etmiss.et())+" "+str(etmiss.phi())+" "+str(scalePt)+"\n")

        # Represent Leptons
        scalePt = size_factor*self.hist_leptpt.GetXaxis().FindBin(leadlepton.pt())
        # scalePt = math.log(leadlepton.pt(),11./10.)
        if abs(leadlepton.pdgId()) == 11 :
          ax.add_artist(AH.DrawObject(leadlepton, scalePt, "lepton"))
        else :
          ax.add_artist(AH.DrawObject(leadlepton, scalePt, "lepton"))

        if VERBOSE :
          f.write(str(eventinfo.eventNumber())+" : lepton :: "+str(leadlepton.pt())+" "+str(leadlepton.phi())+" "+str(leadlepton.eta())+" "+str(scalePt)+"\n")

        #'event_numEvent.jpg'
        import os
        if not os.path.exists(self.getImageDir()): os.makedirs(self.getImageDir())
        image_name = self.getImageDir()+"/"+str(eventinfo.eventNumber())+".jpg"
        fig.savefig(image_name, dpi=56)
        plt.close(fig)
        del plt, fig

      if VERBOSE :
        f.close()

      return True

  def finalize(self):
      pass

def isGoodLepton(Lepton):
    if not Lepton.isTightID(): return False
    if not Lepton.pt() > 30: return False
    if not Lepton.isoetconerel20() < 0.15: return False
    if not Lepton.isoptconerel30() < 0.15: return False
    if abs(Lepton.pdgId()==11 and (abs(Lepton.eta()>2.47) or (abs(Lepton.eta()>1.37) and abs(Lepton.eta()<1.52)))): return False
    return True
