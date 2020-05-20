#!/usr/bin/env python

import os, re, sys, commands, string, random, math
import multiprocessing as mp
import getpass
from optparse import OptionParser

import ROOT
from ROOT import TTree, TChain, TF1, TH1I, TH1F, TH1D, TH2F, TH2D, TGraphErrors, TGraph, THStack, TLegend, TPie, TEllipse
from ROOT import TFile, TCanvas, TPad, TLine, TLorentzVector, TMathText, TLatex
from ROOT import TCanvas, TASImage, TColor

from array import array

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'+'WARNING :: '
    FAIL = '\033[91m'+'ERROR :: '
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def createCanvas(event, goodJets, goodBJets, Leptons, etmiss, imageDir) :

    pt_bins = [0.,10.,20.,30.,50.,100.,250.,500.,1000.,1500.,2000.]
    histo_bin =  ROOT.TH1D("hito_bins", "p_{T} [GeV]", len(pt_bins)-1, array('d', pt_bins))

    # Create the empty figure
    ecolors = {"leptons" : TColor(10007, 0., 1.0, 0.), "bjet" : TColor(10004,1., 0., 0.), "jet": TColor(10003,1,0.5,0.5), "etmiss" : TColor(10006,0.,0,0.9), "mltop" : TColor(10005,0,0,1)}
    fcolors = {"leptons" : TColor(20007, 0., 1.0, 0.), "bjet" : TColor(20004,1., 0., 0.), "jet": TColor(20003,1,0.5,0.5), "etmiss" : TColor(20006,0.,0,0.9), "mltop" : TColor(20005,0,0,0)}

    ecol = []
    fcol = []
    for x in ecolors :
      ecol.append(ecolors[x])
      fcol.append(fcolors[x])

    import os
    if not os.path.exists(imageDir): os.makedirs(imageDir)
    image_name = imageDir+"/"+str(event)+".jpg"

    size = 224
    c1 = TCanvas("c1", "c1", size, size)
    c1.Range(-4.5,-math.pi,4.5,math.pi);
    c1.SetCanvasSize(size,size);

    jets = []
    # Represent Jets
    for jet in goodBJets :
        scalePt = 1.5*histo_bin.GetXaxis().FindBin(jet.Pt())

        elj = DrawObjectROOT(jet, scalePt, "bjet")
        jets.append(elj)
        jets[-1].Draw()

    # Represent Jets
    for jet in goodJets :
        scalePt = 1.5*histo_bin.GetXaxis().FindBin(jet.Pt())

        elj = DrawObjectROOT(jet, scalePt, "jet")
        jets.append(elj)
        jets[-1].Draw()

    scalePt = 1.5*histo_bin.GetXaxis().FindBin(etmiss.Pt())
    elmet = DrawObjectROOT(etmiss, scalePt, "etmiss")
    elmet.Draw()


    # Represent Leptons
    scalePt = 1.5*histo_bin.GetXaxis().FindBin(Leptons[0].Pt())
    ell = DrawObjectROOT(Leptons[0], scalePt, "leptons")
    ell.Draw()

    #'event_numEvent.jpg'
    img = TASImage()
    img.FromPad(c1);
    img.SetImageQuality(3)
    img.SetImageCompression(50)
    img.WriteImage(image_name);



def MTW(lepton, met) :

    pt = lepton.Pt()
    a1 = lepton.Phi()

    Etmiss = met.Pt()
    a2 = met.Phi()

    return math.sqrt( 2.*pt*Etmiss*(1. - math.cos(a1-a2)) )

# Draw Physics Objects as Circles
def DrawObjectROOT(particle, pt, ptype) :

    phiAxis = pt *2.* math.pi / 224. # Ellypse axis
    etaAxis = pt *9. / 224.

    y = particle.Phi()
    try  :
        x = particle.Eta()
    except:
        x = 0

    ellipse = TEllipse(x,y,etaAxis, phiAxis)
    if 'mltop' in ptype :
        ellipse.SetLineColor(10000+len(ptype))
        ellipse.SetLineWidth(2)
        ellipse.SetFillColorAlpha(20000+len(ptype), 0.0)
    else :
        ellipse.SetLineWidth(0)
        ellipse.SetFillColor(20000+len(ptype))
        ellipse.SetFillStyle(1001)

    return ellipse


# ====================================================================
#  main
# ====================================================================
def main():

    # parse options from command line
    parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
    parser.add_option("-f", "--input",        dest="infile",                       help="Input File (default: %default)")
    parser.add_option("-o", "--outdir",       dest="outdir",                       help="Out Directory (default: %default)")
    parser.add_option("-v", "--verbose",      dest="verbose", action="store_true", help="print more information (default: %default)")
    parser.set_defaults(verbose=True, infile='ttbar_10fb_filter.csv', outdir="imagesDir")
    (options,args) = parser.parse_args()

    Save = True
    try:
        InputFile = open(options.infile, 'r')
        eventId = 0
        while Save:
            line = InputFile.readline()
            if not line: break

            line = line.strip()
            objects = line.split(";")

            # Objects
            nBJets = []
            nLep = []
            nJets = []

            # MET
            met = TLorentzVector()
            met.SetPtEtaPhiM(float(objects[3])/1000.,0,float(objects[4]),0)

            # Check Particles (leptons, jets and bjets)
            for obj in objects :
                if len(obj.split(",")) == 5 :
                    part = obj.split(",")

                    if part[0] == 'b' :
                        bjet = TLorentzVector()
                        bjet.SetPtEtaPhiE(float(part[2])/1000.,float(part[3]),float(part[4]),float(part[1])/1000.)
                        nBJets.append(bjet)

                    if part[0] == 'j' :
                        jet = TLorentzVector()
                        jet.SetPtEtaPhiE(float(part[2])/1000.,float(part[3]),float(part[4]),float(part[1])/1000.)
                        nJets.append(jet)

                    if (part[0][0] == 'e' or part[0][0] == "m") :
                        lep = TLorentzVector()
                        lep.SetPtEtaPhiE(float(part[2])/1000.,float(part[3]),float(part[4]),float(part[1])/1000.)
                        nLep.append(lep)

            # Number of Leptons
            if len(nLep) != 1 :
                continue

            # Transverse Mass Cut
            if not MTW(nLep[0],met) > 30 :
                continue

            eventNumber = int(objects[0])*1000000 + eventId
            createCanvas(eventNumber, nJets, nBJets, nLep, met, options.outdir)
            eventId += 1

            if eventId > 10 : Save = False

    except IOError:
        errmsg = " File %s does not exist!" % options.infile
        print( bcolors.FAIL + errmsg + bcolors.ENDC)
        sys.exit()

# ====================================================================
#  __main__
# ====================================================================
if __name__ == '__main__':
    main()
