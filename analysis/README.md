# Analysis directories

Each directory contains a calibration sub-experiment. The main experiments are in the directories

 - `borehole/`: Calibration with the borehole GL12-2A data
 - `synthetic/`: Calibration with synthetic borehole data

The supplementary directories consist of:

 - `leave_one_out/`: Repeated calibation experiment, individually considering each of the 100 test simulations as data
 - `numerics/`: Comparison of GlaDS simulations for different numerical tolerances
 - `sensitivity`: Comparison of GlaDS simulations using different choices for creep-opening, moulins, and the sheet-flow parameterization
