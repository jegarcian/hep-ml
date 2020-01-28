## About
This is the analysis code that may be used to analyse the data of the ATLAS published dataset.

## Setup
After checking out the repository, go to the root-folder of your installation and do the following:

You can populate the _Input_ folder with the files from the dataset by downloading it from opendata.cern.ch.
The provided zip-file should be placed into the Input folder and unzipped via

> unzip ATLASDataSet.zip

## General Usage
### Analysis
The files in the root directory of the installation are the various run scripts. Configuration files can be found in the *Configurations* folder. 

As a first test to check whether everything works fine you can simply run a preconfigured analyis via

>     python RunScript.py -a DMAnalysis -s "Wjets, ttbar"

What you have done here is to run the code in single core mode using the TTbarAnalysis and specifying that you only want to analyse the WW and WZ sample 
as defined in the Configurations/Configuration.py. 
The runscript has several options which may be displayed by typing

>     python RunScript.py --help

The options include:

>     -a,            --analysis              overrides the analysis that is stated in the configuration file
>     -s,            --samples               comma separated string that contains the keys for a subset of processes to run over
>     -p,            --parallel              enables running in parallel (default is single core use)
>     -n NWORKERS,   --nWorkers NWORKERS     specifies the number of workers if multi core usage is desired (default is 4)
>     -c CONFIGFILE, --configfile CONFIGFILE specifies the config file to be read (default is Configurations/Configuration.py)
>     -o OUTPUTDIR,  --output OUTPUDIR       specifies the output directory you would like to use instead of the one in the configuration file

The Configuration.py file specifies how an analysis should behave. The Job portion of the configuration looks like this:

>      Job = {
>          "Batch"           : True,              (switches progress bar on and off, forced to be off when running in parallel mode)
>          "Analysis"        : "TTbarAnalysis",   (names the analysis to be executed)
>          "Fraction"        : 1,                 (determines the fraction of events per file to be analysed)
>          "MaxEvents"       : 1234567890,        (determines the maximum number of events per file to be analysed)
>          "OutputDirectory" : "results/"         (specifies the directory where the output root files should be saved)
>      }

The second portion of the configuration file specifies which 
The locations of the individual files that are to be used for the different 
processes can be set es such:

>     Processes = {
>         # Diboson processes
>         "WW"                    : "Input/MC/mc_105985.WW.root",  (single file)
>         ...
>         "data_Egamma"           : "Input/Data/DataEgamma*.root", (potentially many files)
>     }

The files associated with the processes are found via python's glob module, enabling the use of unix style wildcards.

The names chosen for the processes are important as they are the keys that are used later in the _infofile.py_ to determine the necessary 
scaling factors for correct plotting. 

Now we want to run over the full set of available samples. For this simply type:

>     python RunScript.py -a DMAnalysis

Use the options -p and -n if you have a multi core system and want to use multiple cores.
Execution times are between 1 to 1.5 hours in single core mode or ~ 15 minutes in multi core mode.

### Preparing Training and Validation datasets

Train and validation datasets, including csv files, to be used for training can be created with:

> python CreateTrainValDatasets.py

Using the script input and output paths for datasets can be changed, as well as ratio between train/validation.
