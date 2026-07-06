import os
from enum import Enum

# ==============================================================================
# ENUMS (Definition of project constants)
# ==============================================================================

class Algorithm(Enum):
    """Supported metaheuristic and instance selection algorithms."""
    RANDOM_SEARCH = "RS"
    LOCAL_SEARCH = "LS"
    GENETIC = "GEN"
    MEMETIC = "MEM"   


class Metric(Enum):
    """Evaluation metrics for the models."""
    ACCURACY = "accuracy"
    F1_SCORE = "f1"
    PRECISION = "precision"
    RECALL = "recall"


class ModelName(Enum):
    """Supported Deep Learning architectures (Agreed with tutors)."""
    ALEXNET = "alexnet"
    RESNEXT = "resnext"
    EFFICIENTNETV2 = "efficientnetv2"
    SWIN_TRANSFORMER = "swintransformer"


class DatasetName(Enum):
    """Supported datasets for the experiments."""
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"          
    TINYIMAGENET = "TINYIMAGENET" 


class PrintMode(Enum):
    """Modes for printing or plotting results."""
    CONSTRAINED = "constrained"  
    FREE = "free"                
    BOTH = "both"                
    COMBINED = "combined"        


# ==============================================================================
# SINGLETON: Global Configuration
# ==============================================================================

class GlobalConfig:
    """
    Singleton class to hold global configuration state (Task ID, Dataset, etc.)
    across the entire execution without passing variables constantly.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create the instance only the first time
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            
            # Initialize with default values or kwargs provided during first instantiation
            cls._instance.date = kwargs.get("date", "default_date")
            cls._instance.task_id = kwargs.get("task_id", "default_task")
            cls._instance.dataset_name = kwargs.get("dataset_name", "MNIST").upper()
            cls._instance.model_name = kwargs.get("model_name", "alexnet").lower() # <-- NEW
            
            # Construct paths dynamically
            cls._instance.dataset_path = os.path.join("data", cls._instance.dataset_name)
            # Path to save/load pre-extracted features: data/features/MNIST/alexnet/
            cls._instance.features_path = os.path.join("data", "features", cls._instance.dataset_name, cls._instance.model_name) # <-- NEW
            
        return cls._instance

    def update_config(self, **kwargs):
        """
        Allows updating the configuration dynamically after initialization.
        
        Args:
            **kwargs: Arbitrary keyword arguments (date, task_id, dataset_name, model_name).
        """
        if "date" in kwargs:
            self.date = kwargs["date"]
            
        if "task_id" in kwargs:
            self.task_id = kwargs["task_id"]
            
        if "dataset_name" in kwargs:
            self.dataset_name = kwargs["dataset_name"].upper()
            self.dataset_path = os.path.join("data", self.dataset_name)
            
        if "model_name" in kwargs: # <-- NEW
            self.model_name = kwargs["model_name"].lower()
            
        # Recalculate features path if either dataset or model changes
        self.features_path = os.path.join("data", "features", self.dataset_name, self.model_name) # <-- NEW