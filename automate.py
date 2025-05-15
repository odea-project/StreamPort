"""
Workflow automation script for AnomalyDetector package
"""

#import device engine
from src.StreamPort.device.DeviceEngine import DeviceEngine

#User input for path to data
base_dir = input('Enter string path to Chemstation data: ')
if base_dir is None:
    #specify path to get analyses from
    base_dir = r'C:\Users\PC0118\Desktop\ExtractedSignals'
else:
    r'{}'.format(base_dir)

#Creates an empty DeviceEngine object. 'source' is passed to specify path to data handled by the device 
dev = DeviceEngine(source = base_dir)

#Add project headers as dict
dev.add_headers(headers = {'name': 'Pressure Curve Analysis', 'author': 'Sandeep H.'})

#find analyses present in 'source' attributed to current device
analyses = dev.find_analyses()

#method ids handled by this device are automatically stored within the device engine object as attributes after reading from source
print('Method Ids for this device: \n')
print(dev._method_ids)

#add analyses to device object
dev.add_analyses(analyses)

#import processing settings
from src.StreamPort.device.DeviceProcSettings import ExtractPressureFeatures as features, DecomposeCurves as decompose, FourierTransform as fourier, Scaler 
settings = [features(), decompose(), fourier(), Scaler()]
#add settings to device
for set in settings:
    dev.add_settings(set)

#run through settings workflow for current device
dev.run_workflow()

#plot analyses and results by method
dev.plot_results()

#Machine learning imports and object declaration
from src.StreamPort.ml.MachineLearningEngine import MachineLearningEngine
ml_engine = MachineLearningEngine()

#isolation forest runs through all analyses grouped by method and scaled and creates new objects for post-iForest use.
from src.StreamPort.ml.MachineLearningProcessingSettings import MakeModelIsoForest
iso_forest = MakeModelIsoForest(dev, random_state=22)#22 seemed to pick better train sets

#set ml engine for iForest implementation on dev analyses. This also plots results
ml_engine.add_settings(iso_forest)
method_objects = iso_forest.run(ml_engine)

