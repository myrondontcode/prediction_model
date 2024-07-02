import sys
import os
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline
from prediction_model.pipeline import pipe

def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    
    # Debugging: Print the loaded data columns
    print("Loaded data columns:", train_data.columns)
    
    y = train_data[config.TARGET].map({'N': 0, 'Y': 1})
    pipe.fit(train_data[config.FEATURES], y)
    save_pipeline(pipe)

if __name__ == '__main__':
    perform_training()
