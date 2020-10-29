W_str = r"W" + r"\rightarrow" + r"l" + r"\nu_{l}"
Z_str = r"Z" + r"\rightarrow" + r"l^{+}" + r"l^{-}"
ttbar_str = r"t" + r"\bar{t}"
config = {
"Luminosity": 10064,
"InputDirectory": "results",

"Histograms" : {
    "WtMass"             : {"log_y" : True },
    "etmiss"             : {"log_y" : True },
    "lep_n"              : {"log_y" : True },
    "lep_pt"             : {"log_y" : True },
    "lep_eta"            : {"log_y" : True },
    "lep_E"              : {"log_y" : True },
    "lep_phi"            : {"y_margin" : 0.6, "log_y" : True},
    "lep_charge"         : {"y_margin" : 0.6, "log_y" : True},
    "lep_type"           : {"y_margin" : 0.5, "log_y" : True},
    "lep_ptconerel30"    : {},
    "lep_etconerel20"    : {},
    "lep_d0"             : {},
    "lep_z0"             : {},
    "n_jets"             : {},
    "jet_pt"             : {"log_y" : True },
    "jet_eta"            : {},
    "jet_MV2c10"         : {"y_margin" : 0.3},
    "mtop"               : {"log_y" : True},

},

"Paintables": {
    "Stack": {
        "Order"     : ["Diboson", "Z+jets", "W+jets", "sgtop", "ttbar"],
        "Processes" : {
            "Diboson" : {
                "Color"         : "#6568C5",
                "Contributions" : ["WZlvll","WZlvvv","ZZllll","ZZvvll","WWlvqq","WZqqll","WZlvqq","ZZqqll"]},

            "W+jets": {
                "Color"         : "#64C84D",
                "Contributions" : ["Wmunu_PTV0_70_CVetoBVeto","Wmunu_PTV0_70_CFilterBVeto","Wmunu_PTV0_70_BFilter","Wmunu_PTV70_140_CVetoBVeto","Wmunu_PTV70_140_CFilterBVeto","Wmunu_PTV70_140_BFilter","Wmunu_PTV140_280_CVetoBVeto","Wmunu_PTV140_280_CFilterBVeto","Wmunu_PTV140_280_BFilter","Wmunu_PTV280_500_CVetoBVeto","Wmunu_PTV280_500_CFilterBVeto","Wmunu_PTV280_500_BFilter","Wmunu_PTV500_1000","Wenu_PTV0_70_CVetoBVeto","Wenu_PTV0_70_CFilterBVeto","Wenu_PTV0_70_BFilter","Wenu_PTV70_140_CVetoBVeto","Wenu_PTV70_140_CFilterBVeto","Wenu_PTV70_140_BFilter","Wenu_PTV140_280_CVetoBVeto","Wenu_PTV140_280_CFilterBVeto","Wenu_PTV140_280_BFilter","Wenu_PTV280_500_CVetoBVeto","Wenu_PTV280_500_CFilterBVeto","Wenu_PTV280_500_BFilter","Wenu_PTV500_1000","Wenu_PTV1000_E_CMS","Wtaunu_PTV0_70_CVetoBVeto","Wtaunu_PTV0_70_CFilterBVeto","Wtaunu_PTV0_70_BFilter","Wtaunu_PTV70_140_CVetoBVeto","Wtaunu_PTV70_140_CFilterBVeto","Wtaunu_PTV70_140_BFilter","Wtaunu_PTV140_280_CVetoBVeto","Wtaunu_PTV140_280_CFilterBVeto","Wtaunu_PTV140_280_BFilter","Wtaunu_PTV280_500_CVetoBVeto","Wtaunu_PTV280_500_CFilterBVeto","Wtaunu_PTV280_500_BFilter","Wtaunu_PTV500_1000","Wtaunu_PTV1000_E_CMS"]},

            "Z+jets": {
                "Color"         : "#ED4CC7",
                "Contributions" : ["Zmumu_PTV0_70_CVetoBVeto","Zmumu_PTV0_70_CFilterBVeto","Zmumu_PTV0_70_BFilter","Zmumu_PTV70_140_CVetoBVeto","Zmumu_PTV70_140_CFilterBVeto","Zmumu_PTV70_140_BFilter","Zmumu_PTV140_280_CVetoBVeto","Zmumu_PTV140_280_CFilterBVeto","Zmumu_PTV140_280_BFilter","Zmumu_PTV280_500_CVetoBVeto","Zmumu_PTV280_500_CFilterBVeto","Zmumu_PTV280_500_BFilter","Zmumu_PTV500_1000","Zmumu_PTV1000_E_CMS","Zee_PTV0_70_CVetoBVeto","Zee_PTV0_70_CFilterBVeto","Zee_PTV0_70_BFilter","Zee_PTV70_140_CVetoBVeto","Zee_PTV70_140_CFilterBVeto","Zee_PTV70_140_BFilter","Zee_PTV140_280_CVetoBVeto","Zee_PTV140_280_CFilterBVeto","Zee_PTV140_280_BFilter","Zee_PTV280_500_CVetoBVeto","Zee_PTV280_500_CFilterBVeto","Zee_PTV280_500_BFilter","Zee_PTV500_1000","Zee_PTV1000_E_CMS","Ztautau_PTV0_70_CVetoBVeto","Ztautau_PTV0_70_CFilterBVeto","Ztautau_PTV0_70_BFilter","Ztautau_PTV70_140_CVetoBVeto","Ztautau_PTV70_140_CFilterBVeto","Ztautau_PTV70_140_BFilter","Ztautau_PTV140_280_CVetoBVeto","Ztautau_PTV140_280_CFilterBVeto","Ztautau_PTV140_280_BFilter","Ztautau_PTV280_500_CVetoBVeto","Ztautau_PTV280_500_CFilterBVeto","Ztautau_PTV280_500_BFilter","Ztautau_PTV500_1000","Ztautau_PTV1000_E_CMS"]},

            "sgtop": {
                "Color"         : "#62CAFA",
                "Contributions" : ["single_top_tchan","single_antitop_tchan","single_top_schan","single_antitop_schan","single_top_wtchan","single_antitop_wtchan"]},

            "ttbar": {
                "Color"         : "#F29D38",
                "Contributions" : ["ttbar_lep"]}
        }
    },

    "data" : {
        "Contributions": ["dataA_1lep", "dataB_1lep", "dataC_1lep", "dataD_1lep" ]}
},

"Depictions": {
    "Order": ["Main", "Data/MC"],
    "Definitions" : {
        "Data/MC": {
            "type"       : "Agreement",
            "Paintables" : ["data", "Stack"]
        },

        "Main": {
            "type"      : "Main",
            "Paintables": ["Stack", "data"]
        },
    }
},
}
