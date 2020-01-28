Job = {
    "Batch"           : True,
    "Analysis"        : "DMAnalysis",
    "Fraction"        : 0.001,
    "MaxEvents"       : 1234567890,
    "OutputDirectory" : "results/",
    "ImagesDirectory" : "images/"
}

#VBSAnalysis

Processes = {
  # Diboson processes
  "Diboson"               : "Input/MC/mc_10598*.*.root",

  # single top processes
  "sgtop"                  : "Input/MC/mc_110*.stop_*.root",

  # top pair processes
  "ttbar"                  : "Input/MC/mc_1170*.ttbar*.root",


  # Z+jets processes
  "Zjets"                 : "Input/MC/mc_1477*.root",

  # W+jets processes
  "Wjets"                 : "Input/MC/mc_1677*.W*.root",

  # Data
  "data_Egamma"           : "Input/Data/DataEgamma*.root",
  "data_Muons"            : "Input/Data/DataMuons*.root",
}
