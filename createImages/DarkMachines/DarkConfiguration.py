Directories = {
    "InputCSVdirectory"  : "/lustre/ific.uv.es/ml/ific037/data/hep-ml/csv",
    "OutputDirectory" : "images"
}

SetUp = {

    "Nevents"         :   1000,             # Number of events to be converted into images
    "Process"         :   "Wjets",           # One of the Processes below 
    "mltop"           :   False,             # For drawing mlb circumferences around each bjet circle
    "sf"              :   2.,                # Size factor for manipulating the size of all the circles
    "random"          :   False,             # Select random events from the csv file

}

Processes = {

    # Backgrounds
    # ttbar
    "ttbar"           :   "ttbar_10fb_filter.csv",

    # W+jets
    "Wjets"           :   "w_jets_10fb_filter.csv",

    # t-channel single top
    "tchannel"        :   "single_top_10fb_merge_filter.csv",

    # Wt single top
    "Wt"              :   "wtopbar_10fb_merge_filter.csv",

    # Signal
    "Med2000"         :   "delphes_LHE_mMed2000_filter.csv",
    "Med2250"         :   "delphes_LHE_mMed2250_filter.csv",
    "Med2500"         :   "delphes_LHE_mMed2500_filter.csv"
 
}
