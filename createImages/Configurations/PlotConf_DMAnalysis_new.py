config = {
"Luminosity": 1000,
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
    "jet_m"              : {},
    "jet_jvf"            : {"y_margin" : 0.4},
    "jet_eta"            : {},
    "jet_MV1"            : {"y_margin" : 0.3},
    "vxp_z"              : {},
    "pvxp_n"             : {},
},

"Paintables": {
    "Stack": {
        "Order"     : ["Diboson", "Zjets", "Wjets", "sgtop", "ttbar"],
        "Processes" : {
            "Diboson" : {
                "Color"         : "#6568C5",
                "Contributions" : ["WW", "WZ", "ZZ"]},

            "Wjets": {
                "Color"         : "#64C84D",
                "Contributions" : ["WenuJetsBVeto", "WenuWithB", "WenuNoJetsBVeto", "WmunuJetsBVeto", "WmunuWithB", "WmunuNoJetsBVeto", "WtaunuJetsBVeto", "WtaunuWithB", "WtaunuNoJetsBVeto", "DYeeM08to15", "DYeeM15to40", "DYmumuM08to15", "DYmumuM15to40", "DYtautauM08to15", "DYtautauM15to40"]},

            "Zjets": {
                "Color"         : "#ED4CC7",
                "Contributions" : ["Zee", "Zmumu", "Ztautau"]},

            "sgtop": {
                "Color"         : "#62CAFA",
                "Contributions" : ["stop_tchan_top", "stop_tchan_antitop", "stop_schan", "stop_wtchan"]},

            "ttbar": {
                "Color"         : "#F29D38",
                "Contributions" : ["ttbar_lep", "ttbar_had"]}
        }
    },

    "data" : {
        "Contributions": ["data_Egamma", "data_Muons"]}
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
