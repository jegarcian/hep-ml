Job = {
    "Batch"           : True,
    "Analysis"        : "DMAnalysis",
#    "Fraction"        : 0.025,
    "Fraction"        : 1.0,
    "MaxEvents"       : 1234567890,
    "OutputDirectory" : "results/",
    "ImagesDirectory" : "images/"
}

Processes = {

    # Diboson processes
#    "WWlvlv"               : "Input/MC/mc15_13TeV.361600.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WWlvlv.1lep_raw.root",
#    "WZlvll"               : "Input/MC/mc15_13TeV.361601.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZlvll_mll4.1lep_raw.root",
#    "WZlvvv"               : "Input/MC/mc15_13TeV.361602.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZlvvv_mll4.1lep_raw.root",
#    "ZZllll"               : "Input/MC/mc15_13TeV.361603.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_ZZllll_mll4.1lep_raw.root",
#    "ZZvvll"               : "Input/MC/mc15_13TeV.361604.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_ZZvvll_mll4.1lep_raw.root",
#    "WWlvqq"               : "Input/MC/mc15_13TeV.361606.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WWlvqq.1lep_raw.root",
#    "WZqqll"               : "Input/MC/mc15_13TeV.361607.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZqqll_mll20.1lep_raw.root",
#    "WZlvqq"               : "Input/MC/mc15_13TeV.361609.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZlvqq_mqq20.1lep_raw.root",
#    "ZZqqll"               : "Input/MC/mc15_13TeV.361610.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_ZZqqll_mqq20mll20.1lep_raw.root",

    # Z+jets processes
#    "Zee"                  : "Input/MC/mc15_13TeV.361106.PwPy8EG_AZNLOCTEQ6L1_Zee.1lep_raw.root",
#    "Zmumu"                : "Input/MC/mc15_13TeV.361107.PwPy8EG_AZNLOCTEQ6L1_Zmumu.1lep_raw.root",
#    "Ztautau"              : "Input/MC/mc15_13TeV.361108.PwPy8EG_AZNLOCTEQ6L1_Ztautau.1lep_raw.root",
    
    # W+jets processes
#    "Wplusenu"             : "Input/MC/mc15_13TeV.361100.PwPy8EG_AZNLOCTEQ6L1_Wplusenu.1lep_raw.root",
#    "Wplusmunu"            : "Input/MC/mc15_13TeV.361101.PwPy8EG_AZNLOCTEQ6L1_Wplusmunu.1lep_raw.root",
#    "Wplustaunu"           : "Input/MC/mc15_13TeV.361102.PwPy8EG_AZNLOCTEQ6L1_Wplustaunu.1lep_raw.root",
#    "Wminusenu"            : "Input/MC/mc15_13TeV.361103.PwPy8EG_AZNLOCTEQ6L1_Wminusenu.1lep_raw.root",
#    "Wminusmunu"           : "Input/MC/mc15_13TeV.361104.PwPy8EG_AZNLOCTEQ6L1_Wminusmunu.1lep_raw.root",
#    "Wminustaunu"          : "Input/MC/mc15_13TeV.361105.PwPy8EG_AZNLOCTEQ6L1_Wminustaunu.1lep_raw.root",

    # W+jets above merged
    "Wjets"                : "Input/MC/mc15_13TeV.999999.PwPy8EG_AZNLOCTEQ6L1_Wjets.1lep_raw.root",

    # top pair processes
    "ttbar"                          : "Input/MC/mc15_13TeV.410000.PwPyEG_P2012_ttbar_hdamp172p5_nonallhad.1lep_raw.root",

    # single top processes
#    "single_top_tchan"                   : "Input/MC/mc15_13TeV.410011.PwPyEG_P2012_singletop_tchan_lept_top.1lep_raw.root",
#    "single_antitop_tchan"               : "Input/MC/mc15_13TeV.410012.PwPyEG_P2012_singletop_tchan_lept_antitop.1lep_raw.root",
#    "single_top_wtchan"                  : "Input/MC/mc15_13TeV.410013.PwPyEG_P2012_Wt_inclusive_top.1lep_raw.root",
#    "single_antitop_wtchan"              : "Input/MC/mc15_13TeV.410014.PwPyEG_P2012_Wt_inclusive_antitop.1lep_raw.root",
#    "single_top_schan"                   : "Input/MC/mc15_13TeV.410025.PwPyEG_P2012_SingleTopSchan_noAllHad_top.1lep_raw.root",
#    "single_antitop_schan"               : "Input/MC/mc15_13TeV.410026.PwPyEG_P2012_SingleTopSchan_noAllHad_antitop.1lep_raw.root",

    # Data
#    "dataA_1lep.root"      : "Input/Data/dataA_1lep.root",
#    "dataB_1lep.root"      : "Input/Data/dataB_1lep.root",
#    "dataC_1lep.root"      : "Input/Data/dataC_1lep.root",
#    "dataD_1lep.root"      : "Input/Data/dataD_1lep.root"
}
