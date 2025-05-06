This is the contence of a weekend long hack I did to train a decision transformer to play the game of doorkey and then use the transformer lens library to use MechInterp tecniques on it. 

The main file for viewing is 01_probe_has_key.ipynb. This is a notebook containing the MechInterp experiments and results. All other files in the project are either utility files for MechInterp functions (DecisionTransfromerInterp/), my linear probe (probe.py) that is trained on the 'has_key' concept, or files to train the decision transformer. 

TODO: Add a video visualisation of the agent moving towards the door without the key when the 'has_key' concept is amplified and the agent hasn't actually picked up the key. 
