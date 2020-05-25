Job = {
    "Batch"           : True,
    "CreateImages"    : True,
    "Analysis"        : "DMAnalysis",
#    "Fraction"        : 0.025,
    "Fraction"        : 1.0,
    "MaxEvents"       : 1234567890,
    "OutputDirectory" : "results/",
    "ImagesDirectory" : "images/"
}

Processes = {

    # Diboson processes merged
    "Diboson"                            : "Input/MC/mc15_13TeV.999996.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_Diboson.1lep_raw.root",

    # Z+jets processes merged
    "Zjets"                              : "Input/MC/mc15_13TeV.999997.Sh_221_NNPDF30NNLO_Zjets.1lep_raw.root",

    # W+jets above merged
    "Wjets"                              : "Input/MC/mc15_13TeV.999998.Sh_221_NNPDF30NNLO_Wjets.1lep_raw.root",

    # top pair processes
    "ttbar"                              : "Input/MC/mc15_13TeV.410000.PwPyEG_P2012_ttbar_hdamp172p5_nonallhad.1lep_raw.root",

    # single top processes merged
    "single_top_tchan"                   : "Input/MC/mc15_13TeV.410010.PwPyEG_P2012_singletop_tchan_lept.1lep_raw.root",
    "single_top_wtchan"                  : "Input/MC/mc15_13TeV.410015.PwPyEG_P2012_Wt_inclusive.1lep_raw.root",
    "single_top_schan"                   : "Input/MC/mc15_13TeV.410020.PwPyEG_P2012_singletop_schan_lept.1lep_raw.root",

    # Data
    #"dataA_1lep"                        : "Input/Data/dataA_1lep.root",
    #"dataB_1lep"                        : "Input/Data/dataB_1lep.root",
    #"dataC_1lep"                        : "Input/Data/dataC_1lep.root",
    #"dataD_1lep"                        : "Input/Data/dataD_1lep.root"
}
