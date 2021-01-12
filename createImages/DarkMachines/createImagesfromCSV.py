#!/usr/bin/env python

import os, re, sys, commands, string, random, math
import multiprocessing as mp
import getpass
import importlib
from optparse import OptionParser
from random import seed
from random import randint

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

def createCanvas(event, goodJets, goodBJets, Leptons, etmiss, imageDir, topmass, size_factor) : #I added boolean topmass and float size_factor variables

    pt_bins = [0.,20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,250.,300.,350.,400.,450.,500.]
    histo_bin =  ROOT.TH1D("hito_bins", "p_{T} [GeV]", len(pt_bins)-1, array('d', pt_bins))

    # Create the empty figure
    ecolors = {"leptons" : TColor(10007, 0., 1.0, 0.), "bjet" : TColor(10004,1., 0., 0.), "jet": TColor(10003,1,0.5,0.5), "etmiss" : TColor(10006,0.,0,0.9), "mltop" : TColor(10005,0.5,0.5,0.5)}
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
        scalePt = size_factor*histo_bin.GetXaxis().FindBin(jet.Pt())

        elj = DrawObjectROOT(jet, scalePt, "bjet")
        jets.append(elj)
        jets[-1].Draw()

        if topmass == True :
            elm = DrawObjectROOT(jet, scalePt+mlb(jet,Leptons[0]), "mltop")
            jets.append(elm)
            jets[-1].Draw()


    # Represent Jets
    for jet in goodJets :
        scalePt = size_factor*histo_bin.GetXaxis().FindBin(jet.Pt())

        elj = DrawObjectROOT(jet, scalePt, "jet")
        jets.append(elj)
        jets[-1].Draw()


    # Represent MET
    scalePt = size_factor*histo_bin.GetXaxis().FindBin(etmiss.Pt())
    elmet = DrawObjectROOT(etmiss, scalePt, "etmiss")
    elmet.Draw()


    # Represent Leptons
    scalePt = size_factor*histo_bin.GetXaxis().FindBin(Leptons[0].Pt())
    ell = DrawObjectROOT(Leptons[0], scalePt, "leptons")
    ell.Draw()


    #'event_numEvent.jpg'
    img = TASImage()
    img.FromPad(c1);
    img.SetImageQuality(3)
    img.SetImageCompression(50)
    img.WriteImage(image_name);


def mlb(jet,lepton):
    pstop = jet + lepton
    return pstop.M()/10.

def MTW(lepton, met) :

    pt = lepton.Pt()
    a1 = lepton.Phi()

    Etmiss = met.Pt()
    a2 = met.Phi()

    return math.sqrt( 2.*pt*Etmiss*(1. - math.cos(a1-a2)) )

def HT(jets, bjets):
    H_T = 0
    for j in jets: 
         H_T += j.Pt()
    for b in bjets:
         H_T += b.Pt()
    return H_T


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
#        ellipse.SetFillColorAlpha(20000+len(ptype), 0.0)
        ellipse.SetFillStyle(0)
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
    parser.add_option("-c", "--configfile",   dest="configfile",                   help="files to be analysed (default: %default)")
    parser.add_option("-p", "--process",      dest="process",                      help="process to be converted in images (default: %default)")
    parser.add_option("-n", "--nevents",      dest="nevents",      type="int",     help="Number of events to be converted (default: %default)")
    parser.add_option("-m", "--mltop",        dest="mltop",   action="store_true", help="If mlb circumferences are included or not (default: %default)")
    parser.add_option("-s", "--sfactor",      dest="sfactor",      type="float",   help="Size factor to manipulate the circles size (default: %default)")
    parser.add_option("-r", "--random",       dest="rand",    action="store_true", help="Select the events randomly from the csv file (default: %default)")
    parser.add_option("-v", "--verbose",      dest="verbose", action="store_true", help="print more information (default: %default)")
    parser.set_defaults(verbose=True, infile="", outdir="", configfile="DarkConfiguration.py")
    (options,args) = parser.parse_args()

    # The .csv path is built using ./DarkConfiguration.py

    configModuleName = options.configfile.replace(".py","")
    configuration = importlib.import_module(configModuleName)


    # Reading set-up information from configfile. But priority is given to the parser options.
    # (default: %default) makes non-specified option variables to store as a None by default.
    process = options.process if options.process != None else configuration.SetUp["Process"]
    Nevents = options.nevents if options.nevents != None else configuration.SetUp["Nevents"]
    mltop = options.mltop if options.mltop != None else configuration.SetUp["mltop"]
    sfactor = options.sfactor if options.sfactor != None else configuration.SetUp["sf"]
    random = options.rand if options.rand != None else configuration.SetUp["random"]


    # According to the process specified, the corresponding .csv and the output directory are indicated
    if process != None:
         infile = options.infile if options.infile != "" else configuration.Directories["InputCSVdirectory"] + "/" + configuration.Processes[process]
         outdir = options.outdir if options.outdir != "" else configuration.Directories["OutputDirectory"] + "/" + process
         if not os.path.exists(outdir):
           os.makedirs(outdir)
    else:
         print("No process was specified")
         sys.exit()


    Save = True
    positive = 0 # To count how many + and - charged leptons are in the selected data
    negative = 0
    try:
        InputFile = open(infile, 'r')
        lines = InputFile.readlines()
        if len(lines)-1 < Nevents: Nevents = len(lines)-1

        #Display image conversion set up
        print("//////////////////////////////////////////////////////////////////////////////////")
        print("// Reading from %s and saving into %s " % (configuration.Processes[process], outdir))
        print("// This file contains %d %s events" % (len(lines)-1, process))
        print("// Converting %d %s events into images %s" % (Nevents, process, "randomly" if random else ""))
        print("// These images consist in filled circles %s" % ("with mlb circumfereces included" if mltop else " "))
        print("//////////////////////////////////////////////////////////////////////////////////")
        
        count = 0  # It counts every completed iteration
        attempt = 0    # It counts every tried iteration
        seed(1)
        cache = []
        while Save:
            if not lines: break
            
            if random :            
                  #Random choice of lines, without repeating
                  rnd = randint(0, len(lines)-1)
                  if rnd in cache:
                      continue
                  else:
                      cache.append(rnd)
                      eventline = rnd
            else:
                  #Reading every line in order
                  eventline = attempt
                  attempt += 1
            
            line = lines[eventline]
#            print(eventline)
            line = line.strip()
            objects = line.split(";")

            # Objects
            nBJets = []
            nLep = []
            nJets = []

            lepton = ""

#            print(eventline)           

            # MET
            met = TLorentzVector()
            met.SetPtEtaPhiM(float(objects[3])/1000.,0,float(objects[4]),0)

#            print(met.Pt())

#            print(eventline)

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
                        lepton = part[0]
                        lep = TLorentzVector()
                        lep.SetPtEtaPhiE(float(part[2])/1000.,float(part[3]),float(part[4]),float(part[1])/1000.)
                        nLep.append(lep)

#            print(eventline)

           
            # Number of Leptons
            if len(nLep) != 1 :
                continue

#            print(eventline)

            # Transverse Mass Cut
            if not MTW(nLep[0],met) > 30 :
                continue

#            print(eventline)

            # H_T > 100 GeV cut
            if not HT(nJets, nBJets) > 100 :
                continue

            if met.Pt() < 30:
                print("MET =" + str(met.Pt()))


            # Number of positive and negative charge lepton for t-channel and Wt single top processes
            if lepton[1] == '-' :
                negative += 1
            else:
                positive += 1

#            print(eventline)
            # Labeling every image
            eventNumber = int(objects[0])*1000000 + eventline
#            print("%s %d" % (objects[0],eventline))
            createCanvas(eventNumber, nJets, nBJets, nLep, met, outdir, mltop, sfactor)
            count += 1

            #Display the progress             
            if float(count)/5000.-int(float(count)/5000.) == 0:
                print("Images already generated: "+ str(count))

            if count >= Nevents-1 or attempt >= Nevents-1 :
                    Save = False
                    

    except IOError:
        errmsg = " File %s does not exist!" % infile
        print( bcolors.FAIL + errmsg + bcolors.ENDC)
        sys.exit()

    print("A total of %d events were converted into images" % count)
    print("There are %d positive leptons and %d negative ones" % (positive, negative))

# ====================================================================
#  __main__
# ====================================================================
if __name__ == '__main__':
    main()
