W_str = r"W" + r"\rightarrow" + r"l" + r"\nu_{l}"
Z_str = r"Z" + r"\rightarrow" + r"l^{+}" + r"l^{-}"
ttbar_str = r"t" + r"\bar{t}"
config = {
"Luminosity": 10064,
"InputDirectory": "results",

"Histograms" : {
    "WtMass"             : {},
    "etmiss"             : {},
    "lep_n"              : {},
    "lep_pt"             : {},
    "lep_eta"            : {},
    "lep_E"              : {},
    "lep_phi"            : {"y_margin" : 0.6},
    "lep_charge"         : {"y_margin" : 0.6},
    "lep_type"           : {"y_margin" : 0.5},
    "lep_ptconerel30"    : {},
    "lep_etconerel20"    : {},
    "lep_d0"             : {},
    "lep_z0"             : {},
    "n_jets"             : {},
    "jet_pt"             : {},
    "jet_eta"            : {},
    "jet_MV2c10"         : {"y_margin" : 0.3},
},

"Paintables": {
    "Stack": {
        "Order"     : ["Diboson", "W", "Z", "stop", "ttbar"],
        "Processes" : {                
            "Diboson" : {
                "Color"         : "#fa7921",
                "Contributions" : ["WWlvlv", "WZlvll", "WZlvvv", "ZZllll", "ZZvvll", "WWlvqq", "WZqqll", "WZlvqq", "ZZqqll"]},
                                
            "W": {              
                "Color"         : "#e55934",
                "Contributions" : ["Wplusenu", "Wplusmunu", "Wplustaunu", "Wminusenu", "Wminusmunu", "Wminustaunu"]},
                                
            "Z": {              
                "Color"         : "#086788",
                "Contributions" : ["Zee", "Zmumu", "Ztautau"]},
                  
            "stop": {
                "Color"         : "#fde74c",
                "Contributions" : ["single_top_tchan", "single_antitop_tchan", "single_top_wtchan", "single_antitop_wtchan", "single_top_schan", "single_antitop_schan"]},
            
            "ttbar": {
                "Color"         : "#9bc53d",
                "Contributions" : ["ttbar_lep"]}
        }
    },

    "data" : {
        "Contributions": ["dataA_1lep.root", "dataB_1lep.root", "dataC_1lep.root", "dataD_1lep.root" ]}
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
