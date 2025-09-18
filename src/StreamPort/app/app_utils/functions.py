from datetime import datetime, date, time as dtime
import os

## Functions ##
# Data Collection
def collect_data(path):
    batches = os.listdir(path)
    batches = [os.path.join(path, file) for file in batches]

    files = []
    for batch in batches:
        batch_files = os.listdir(batch)
        batch_files = [os.path.join(batch, file) for file in batch_files if ".D" in file]
        files.extend(batch_files)
    return files

# Analyses Creation
def create_analyses(files, ana_type):
    if ana_type == "Pressure Curves":
        from src.StreamPort.device.analyses import PressureCurvesAnalyses 
        analyses = PressureCurvesAnalyses(files=files)
    elif ana_type == "Mass Spec":
        from src.StreamPort.device.analyses import MassSpecAnalyses 
        analyses = MassSpecAnalyses(files=files)
    else:
        analyses = None
    return analyses

# Feature Extraction
def extract_features(engine, processor):
    engine.workflow.clear()
    engine.workflow.append(processor)
    engine.run() 

#Train Set Selection
def select_train_set_pc(engine, method='SAA_411_Pac.M', date_threshold_min=None):
    if date_threshold_min is None:
        date_threshold_min = datetime(2021, 8, 19)
    elif isinstance(date_threshold_min, date) and not isinstance(date_threshold_min, datetime):
        # Convert date to datetime at midnight
        date_threshold_min = datetime.combine(date_threshold_min, dtime.min)
    indices = engine.analyses.get_method_indices(method)
    """
    Train Set
    """
    train_indices = []
    for i in indices:
        meta = engine.analyses.get_metadata(i)
        batch_position = meta["batch_position"].item()
        start_time = meta["start_time"].item()
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H-%M-%S")
        if batch_position > 5 and start_time < date_threshold_min:
            train_indices.append(i)
    train_indices.sort()
    
    """
    Test Set
    """
    test_indices = []
    remaining = list(set(indices) - set(train_indices))
    remaining.sort()
    for i in remaining:
        mt = engine.analyses.get_metadata(i)
        start_time = mt["start_time"].item()
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H-%M-%S")
        if start_time >= date_threshold_min:
            test_indices.append(i)

    return train_indices, test_indices

def select_train_set_ms(engine, date_threshold_min=None):
    all_batches = engine.analyses.get_batches()
    if date_threshold_min is None:
        date_threshold_min = datetime(2025, 6, 20)
    elif isinstance(date_threshold_min, date) and not isinstance(date_threshold_min, datetime):
        date_threshold_min = datetime.combine(date_threshold_min, dtime.min)
    train_batch_names = [batch for batch in all_batches if datetime.strptime(" ".join(batch.split(" ")[-2:]), "%Y-%m-%d %H-%M-%S") < date_threshold_min]

    """
    Using the same train batch from PressureCurvesAnalyses
    """
    train_indices = []
    for batch in train_batch_names:
        indices = sorted(engine.analyses.get_batch_indices(batch))

        metadata = engine.analyses.get_metadata(indices=indices)
        metadata.sort_values("index", inplace=True)
        
        # Filter out additional data
        metadata = metadata[
            (metadata["batch_position"] > 4) & # if batch position above 4 
            (~metadata["sample"].isin(["Flush", "Blank"])) & # if "Flush" or "Blank" 
            (metadata["batch"].str.contains("x100ng-mL", na=False)) # if Mix 100ng-mL
            ]

        indices = sorted(metadata["index"])
        train_indices.extend(indices)

    for i in [4, 5, 6, 124, 125]:
        if i in train_indices:
            train_indices.remove(i)
    train_indices.sort()

    """
    Get Test Samples
    """
    remaining_batches = list(set(all_batches) - set(train_batch_names))
    remaining_batches.sort(key=lambda f: (
        datetime.strptime(f.split(' ')[-2] + '_' + f.split(' ')[-1], "%Y-%m-%d_%H-%M-%S")  # Sort by date
        )  
    )

    test_indices = []
    date_threshold_old = date_threshold_min
    for batch in remaining_batches:
        batch_date = datetime.strptime(" ".join(batch.split(" ")[-2:]), "%Y-%m-%d %H-%M-%S") 

        if batch_date >= date_threshold_old: # search and collect test samples
            date_threshold_old = batch_date 
            
            test_indices = engine.analyses.get_batch_indices(batch)

            for j in test_indices:
                ft = engine.analyses.get_features(j)
                if ft.empty:
                    continue
                test_indices.append(j)
                
    return train_indices, test_indices

# Scaling
def scale_data(engine, train_indices, scaler_type: str=None):
    from src.StreamPort.machine_learning.analyses import MachineLearningAnalyses

    metadata = engine.analyses.get_metadata(train_indices)
    variables = engine.analyses.get_features(train_indices)

    ml = MachineLearningAnalyses(variables, metadata)
    
    from src.StreamPort.machine_learning.methods import MachineLearningScaleFeaturesScalerSklearn

    scaler = MachineLearningScaleFeaturesScalerSklearn(scaler_type=scaler_type)
    ml = scaler.run(ml)
    return ml

# Model Creation/Training
def create_iforest(ml):
    from src.StreamPort.machine_learning.methods import MachineLearningMethodIsolationForestSklearn

    iso = MachineLearningMethodIsolationForestSklearn()
    ml = iso.run(ml)
    ml.train()
    return ml

# Testing
def test_sample(ml, engine, test_index, threshold="auto", n_tests=None):
    test_data = engine.analyses.get_features(test_index)
    test_metadata = engine.analyses.get_metadata(test_index)
    ml.predict(test_data, test_metadata)
    outliers = ml.test_prediction_outliers(threshold=threshold, n_tests=n_tests)
    print(outliers)
    return ml

###-----------------------------------------------------------------------------------------------------
