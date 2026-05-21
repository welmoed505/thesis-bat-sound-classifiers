# Annotation_tool.py 
Used as annotation tool to manuall select bat call onsets to normalize IPI between calls
Requires Audacity to be installed and mod-script-pipe enables (Edit > Preferences > Modules > mod-script-pipe > Enabled)
Additional libraries to be installed: pip install paramiko numpy soundfile
Temporarily download files from server to local machine, ssh access required

# Training_pipeline_passt_effnet.py
Trains and evaluates Passt and EfficientNet with a kfold and gridsearch
# -- model can be "effnet" or "passt" depending on which model you want to train 
# -- dataset can be "with_IPI" or "IPI_removed" depending on which dataset version you want to use
