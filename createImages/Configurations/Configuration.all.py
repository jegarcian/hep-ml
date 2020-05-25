Job = {
    "Batch"           : True,
    "CreateImages"    : False,
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
    "dataA_1lep"                        : "Input/Data/dataA_1lep.root",
    "dataB_1lep"                        : "Input/Data/dataB_1lep.root",
    "dataC_1lep"                        : "Input/Data/dataC_1lep.root",
    "dataD_1lep"                        : "Input/Data/dataD_1lep.root"
}

Processes = {

    # Data
    "dataA_1lep" : "Input/Data/dataA_1lep.root",
    "dataB_1lep" : "Input/Data/dataB_1lep.root",
    "dataC_1lep" : "Input/Data/dataC_1lep.root",
    "dataD_1lep" : "Input/Data/dataD_1lep.root",


    # Diboson processes
    "WWlvlv" : "Input/MC/mc15_13TeV.361600.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WWlvlv.1lep_raw.root",
    "WZlvll" : "Input/MC/mc15_13TeV.361601.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZlvll_mll4.1lep_raw.root",
    "WZlvvv" : "Input/MC/mc15_13TeV.361602.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZlvvv_mll4.1lep_raw.root",
    "ZZllll" : "Input/MC/mc15_13TeV.361603.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_ZZllll_mll4.1lep_raw.root",
    "ZZvvll" : "Input/MC/mc15_13TeV.361604.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_ZZvvll_mll4.1lep_raw.root",
    "WWlvqq" : "Input/MC/mc15_13TeV.361606.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WWlvqq.1lep_raw.root",
    "WZqqll" : "Input/MC/mc15_13TeV.361607.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZqqll_mll20.1lep_raw.root",
    "WZlvqq" : "Input/MC/mc15_13TeV.361609.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_WZlvqq_mqq20.1lep_raw.root",
    "ZZqqll" : "Input/MC/mc15_13TeV.361610.PwPy8EG_CT10nloME_AZNLOCTEQ6L1_ZZqqll_mqq20mll20.1lep_raw.root",

    # Z+jets processes
    "Zmumu_PTV0_70_CFilterBVeto"  : "Input/MC/mc15_13TeV.364101.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CFilterBVeto.1lep_raw.root",
    "Zmumu_PTV0_70_CVetoBVeto" : "Input/MC/mc15_13TeV.364100.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.1lep_raw.root",
    "Zmumu_PTV0_70_BFilter" : "Input/MC/mc15_13TeV.364102.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.1lep_raw.root",
    "Zmumu_PTV70_140_CVetoBVeto" : "Input/MC/mc15_13TeV.364103.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_CVetoBVeto.1lep_raw.root",
    "Zmumu_PTV70_140_CFilterBVeto" : "Input/MC/mc15_13TeV.364104.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_CFilterBVeto.1lep_raw.root",
    "Zmumu_PTV70_140_BFilter" : "Input/MC/mc15_13TeV.364105.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.1lep_raw.root",
    "Zmumu_PTV140_280_CVetoBVeto" : "Input/MC/mc15_13TeV.364106.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_CVetoBVeto.1lep_raw.root",
    "Zmumu_PTV140_280_CFilterBVeto" : "Input/MC/mc15_13TeV.364107.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_CFilterBVeto.1lep_raw.root",
    "Zmumu_PTV140_280_BFilter" : "Input/MC/mc15_13TeV.364108.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.1lep_raw.root",
    "Zmumu_PTV280_500_CVetoBVeto" : "Input/MC/mc15_13TeV.364109.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_CVetoBVeto.1lep_raw.root",
    "Zmumu_PTV280_500_CFilterBVeto" : "Input/MC/mc15_13TeV.364110.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_CFilterBVeto.1lep_raw.root",
    "Zmumu_PTV280_500_BFilter" : "Input/MC/mc15_13TeV.364111.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.1lep_raw.root",
    "Zmumu_PTV500_1000" : "Input/MC/mc15_13TeV.364112.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.1lep_raw.root",
    "Zmumu_PTV1000_E_CMS" : "Input/MC/mc15_13TeV.364113.Sh_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.1lep_raw.root",

    "Zee_PTV0_70_CVetoBVeto" : "Input/MC/mc15_13TeV.364114.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV0_70_CVetoBVeto.1lep_raw.root",
    "Zee_PTV0_70_CFilterBVeto" : "Input/MC/mc15_13TeV.364115.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV0_70_CFilterBVeto.1lep_raw.root",
    "Zee_PTV0_70_BFilter" : "Input/MC/mc15_13TeV.364116.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV0_70_BFilter.1lep_raw.root",
    "Zee_PTV70_140_CVetoBVeto" : "Input/MC/mc15_13TeV.364117.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV70_140_CVetoBVeto.1lep_raw.root",
    "Zee_PTV70_140_CFilterBVeto" : "Input/MC/mc15_13TeV.364118.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV70_140_CFilterBVeto.1lep_raw.root",
    "Zee_PTV70_140_BFilter" : "Input/MC/mc15_13TeV.364119.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV70_140_BFilter.1lep_raw.root",
    "Zee_PTV140_280_CVetoBVeto" : "Input/MC/mc15_13TeV.364120.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV140_280_CVetoBVeto.1lep_raw.root",
    "Zee_PTV140_280_CFilterBVeto" : "Input/MC/mc15_13TeV.364121.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV140_280_CFilterBVeto.1lep_raw.root",
    "Zee_PTV140_280_BFilter" : "Input/MC/mc15_13TeV.364122.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV140_280_BFilter.1lep_raw.root",
    "Zee_PTV280_500_CVetoBVeto" : "Input/MC/mc15_13TeV.364123.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV280_500_CVetoBVeto.1lep_raw.root",
    "Zee_PTV280_500_CFilterBVeto" : "Input/MC/mc15_13TeV.364124.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV280_500_CFilterBVeto.1lep_raw.root",
    "Zee_PTV280_500_BFilter" : "Input/MC/mc15_13TeV.364125.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV280_500_BFilter.1lep_raw.root",
    "Zee_PTV500_1000" : "Input/MC/mc15_13TeV.364126.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV500_1000.1lep_raw.root",
    "Zee_PTV1000_E_CMS" : "Input/MC/mc15_13TeV.364127.Sh_221_NNPDF30NNLO_Zee_MAXHTPTV1000_E_CMS.1lep_raw.root",

    "Ztautau_PTV0_70_CVetoBVeto" : "Input/MC/mc15_13TeV.364128.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV0_70_CVetoBVeto.1lep_raw.root",
    "Ztautau_PTV0_70_CFilterBVeto" : "Input/MC/mc15_13TeV.364129.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV0_70_CFilterBVeto.1lep_raw.root",
    "Ztautau_PTV0_70_BFilter" : "Input/MC/mc15_13TeV.364130.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV0_70_BFilter.1lep_raw.root",
    "Ztautau_PTV70_140_CVetoBVeto" : "Input/MC/mc15_13TeV.364131.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV70_140_CVetoBVeto.1lep_raw.root",
    "Ztautau_PTV70_140_CFilterBVeto" : "Input/MC/mc15_13TeV.364132.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV70_140_CFilterBVeto.1lep_raw.root",
    "Ztautau_PTV70_140_BFilter" : "Input/MC/mc15_13TeV.364133.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV70_140_BFilter.1lep_raw.root",
    "Ztautau_PTV140_280_CVetoBVeto" : "Input/MC/mc15_13TeV.364134.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV140_280_CVetoBVeto.1lep_raw.root",
    "Ztautau_PTV140_280_CFilterBVeto" : "Input/MC/mc15_13TeV.364135.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV140_280_CFilterBVeto.1lep_raw.root",
    "Ztautau_PTV140_280_BFilter" : "Input/MC/mc15_13TeV.364136.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV140_280_BFilter.1lep_raw.root",
    "Ztautau_PTV280_500_CVetoBVeto" : "Input/MC/mc15_13TeV.364137.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV280_500_CVetoBVeto.1lep_raw.root",
    "Ztautau_PTV280_500_CFilterBVeto" : "Input/MC/mc15_13TeV.364138.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV280_500_CFilterBVeto.1lep_raw.root",
    "Ztautau_PTV280_500_BFilter" : "Input/MC/mc15_13TeV.364139.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV280_500_BFilter.1lep_raw.root",
    "Ztautau_PTV500_1000" : "Input/MC/mc15_13TeV.364140.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV500_1000.1lep_raw.root",
    "Ztautau_PTV1000_E_CMS" : "Input/MC/mc15_13TeV.364141.Sh_221_NNPDF30NNLO_Ztautau_MAXHTPTV1000_E_CMS.1lep_raw.root",

    # W+jets processes
    "Wmunu_PTV0_70_CVetoBVeto" : "Input/MC/mc15_13TeV.364156.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV0_70_CVetoBVeto.1lep_raw.root",
    "Wmunu_PTV0_70_CFilterBVeto" : "Input/MC/mc15_13TeV.364157.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV0_70_CFilterBVeto.1lep_raw.root",
    "Wmunu_PTV0_70_BFilter" : "Input/MC/mc15_13TeV.364158.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV0_70_BFilter.1lep_raw.root",
    "Wmunu_PTV70_140_CVetoBVeto" : "Input/MC/mc15_13TeV.364159.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV70_140_CVetoBVeto.1lep_raw.root",
    "Wmunu_PTV70_140_CFilterBVeto" : "Input/MC/mc15_13TeV.364160.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV70_140_CFilterBVeto.1lep_raw.root",
    "Wmunu_PTV70_140_BFilter" : "Input/MC/mc15_13TeV.364161.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV70_140_BFilter.1lep_raw.root",
    "Wmunu_PTV140_280_CVetoBVeto" : "Input/MC/mc15_13TeV.364162.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV140_280_CVetoBVeto.1lep_raw.root",
    "Wmunu_PTV140_280_CFilterBVeto" : "Input/MC/mc15_13TeV.364163.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV140_280_CFilterBVeto.1lep_raw.root",
    "Wmunu_PTV140_280_BFilter" : "Input/MC/mc15_13TeV.364164.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV140_280_BFilter.1lep_raw.root",
    "Wmunu_PTV280_500_CVetoBVeto" : "Input/MC/mc15_13TeV.364165.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV280_500_CVetoBVeto.1lep_raw.root",
    "Wmunu_PTV280_500_CFilterBVeto" : "Input/MC/mc15_13TeV.364166.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV280_500_CFilterBVeto.1lep_raw.root",
    "Wmunu_PTV280_500_BFilter" : "Input/MC/mc15_13TeV.364167.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV280_500_BFilter.1lep_raw.root",
    "Wmunu_PTV500_1000" : "Input/MC/mc15_13TeV.364168.Sh_221_NNPDF30NNLO_Wmunu_MAXHTPTV500_1000.1lep_raw.root",
    "Wenu_PTV0_70_CVetoBVeto" : "Input/MC/mc15_13TeV.364170.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV0_70_CVetoBVeto.1lep_raw.root",
    "Wenu_PTV0_70_CFilterBVeto" : "Input/MC/mc15_13TeV.364171.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV0_70_CFilterBVeto.1lep_raw.root",
    "Wenu_PTV0_70_BFilter" : "Input/MC/mc15_13TeV.364172.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV0_70_BFilter.1lep_raw.root",
    "Wenu_PTV70_140_CVetoBVeto" : "Input/MC/mc15_13TeV.364173.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV70_140_CVetoBVeto.1lep_raw.root",
    "Wenu_PTV70_140_CFilterBVeto" : "Input/MC/mc15_13TeV.364174.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV70_140_CFilterBVeto.1lep_raw.root",
    "Wenu_PTV70_140_BFilter" : "Input/MC/mc15_13TeV.364175.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV70_140_BFilter.1lep_raw.root",
    "Wenu_PTV140_280_CVetoBVeto" : "Input/MC/mc15_13TeV.364176.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV140_280_CVetoBVeto.1lep_raw.root",
    "Wenu_PTV140_280_CFilterBVeto" : "Input/MC/mc15_13TeV.364177.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV140_280_CFilterBVeto.1lep_raw.root",
    "Wenu_PTV140_280_BFilter" : "Input/MC/mc15_13TeV.364178.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV140_280_BFilter.1lep_raw.root",
    "Wenu_PTV280_500_CVetoBVeto" : "Input/MC/mc15_13TeV.364179.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV280_500_CVetoBVeto.1lep_raw.root",
    "Wenu_PTV280_500_CFilterBVeto" : "Input/MC/mc15_13TeV.364180.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV280_500_CFilterBVeto.1lep_raw.root",
    "Wenu_PTV280_500_BFilter" : "Input/MC/mc15_13TeV.364181.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV280_500_BFilter.1lep_raw.root",
    "Wenu_PTV500_1000" : "Input/MC/mc15_13TeV.364182.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV500_1000.1lep_raw.root",
    "Wenu_PTV1000_E_CMS" : "Input/MC/mc15_13TeV.364183.Sh_221_NNPDF30NNLO_Wenu_MAXHTPTV1000_E_CMS.1lep_raw.root",

    "Wtaunu_PTV0_70_CVetoBVeto" : "Input/MC/mc15_13TeV.364184.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV0_70_CVetoBVeto.1lep_raw.root",
    "Wtaunu_PTV0_70_CFilterBVeto" : "Input/MC/mc15_13TeV.364185.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV0_70_CFilterBVeto.1lep_raw.root",
    "Wtaunu_PTV0_70_BFilter" : "Input/MC/mc15_13TeV.364186.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV0_70_BFilter.1lep_raw.root",
    "Wtaunu_PTV70_140_CVetoBVeto" : "Input/MC/mc15_13TeV.364187.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV70_140_CVetoBVeto.1lep_raw.root",
    "Wtaunu_PTV70_140_CFilterBVeto" : "Input/MC/mc15_13TeV.364188.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV70_140_CFilterBVeto.1lep_raw.root",
    "Wtaunu_PTV70_140_BFilter" : "Input/MC/mc15_13TeV.364189.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV70_140_BFilter.1lep_raw.root",
    "Wtaunu_PTV140_280_CVetoBVeto" : "Input/MC/mc15_13TeV.364190.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV140_280_CVetoBVeto.1lep_raw.root",
    "Wtaunu_PTV140_280_CFilterBVeto" : "Input/MC/mc15_13TeV.364191.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV140_280_CFilterBVeto.1lep_raw.root",
    "Wtaunu_PTV140_280_BFilter" : "Input/MC/mc15_13TeV.364192.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV140_280_BFilter.1lep_raw.root",
    "Wtaunu_PTV280_500_CVetoBVeto" : "Input/MC/mc15_13TeV.364193.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV280_500_CVetoBVeto.1lep_raw.root",
    "Wtaunu_PTV280_500_CFilterBVeto" : "Input/MC/mc15_13TeV.364194.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV280_500_CFilterBVeto.1lep_raw.root",
    "Wtaunu_PTV280_500_BFilter" : "Input/MC/mc15_13TeV.364195.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV280_500_BFilter.1lep_raw.root",
    "Wtaunu_PTV500_1000" : "Input/MC/mc15_13TeV.364196.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV500_1000.1lep_raw.root",
    "Wtaunu_PTV1000_E_CMS" : "Input/MC/mc15_13TeV.364197.Sh_221_NNPDF30NNLO_Wtaunu_MAXHTPTV1000_E_CMS.1lep_raw.root",

    # top pair processes
    "ttbar_lep" : "Input/MC/mc15_13TeV.410000.PwPyEG_P2012_ttbar_hdamp172p5_nonallhad.1lep_raw.root",

    # single top processes
    "single_top_tchan" : "Input/MC/mc15_13TeV.410011.PwPyEG_P2012_singletop_tchan_lept_top.1lep_raw.root",
    "single_antitop_tchan" : "Input/MC/mc15_13TeV.410012.PwPyEG_P2012_singletop_tchan_lept_antitop.1lep_raw.root",
    "single_top_wtchan" : "Input/MC/mc15_13TeV.410013.PwPyEG_P2012_Wt_inclusive_top.1lep_raw.root",
    "single_antitop_wtchan" : "Input/MC/mc15_13TeV.410014.PwPyEG_P2012_Wt_inclusive_antitop.1lep_raw.root",
    "single_top_schan" : "Input/MC/mc15_13TeV.410025.PwPyEG_P2012_SingleTopSchan_noAllHad_top.1lep_raw.root",
    "single_antitop_schan" : "Input/MC/mc15_13TeV.410026.PwPyEG_P2012_SingleTopSchan_noAllHad_antitop.1lep_raw.root"

}
