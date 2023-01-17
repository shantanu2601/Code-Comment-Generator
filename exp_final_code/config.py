import os 
import torch 


# Paths 
PATH_LOGS = os.environ.get("PATH_LOGS", "./runs")

PATH_BASE_MODELS = os.environ.get("PATH_BASE_MODELS", "./base_models")
PATH_CACHE_DATASETS = os.environ.get("PATH_CACHE_DATASETS", "./data/cache")

PATH_CHECKPOINT_MODELS = os.environ.get("PATH_CHECKPOINT_MODELS", "./models")

PATH_SAVE_NL_ENCODER = os.environ.get("PATH_SAVE_NL_ENCODER", "./models/codecg-nl-encoder")
PATH_SAVE_NL_DECODER = os.environ.get("PATH_SAVE_NL_DECODER", "./models/codecg-nl-decoder")
PATH_SAVE_NL_LM = os.environ.get("PATH_SAVE_NL_LM", "./models/code-cg-nl-lm/lm.pt")
PATH_SAVE_PL_ENCODER = os.environ.get("PATH_SAVE_PL_ENCODER", "./models/codecg-pl-encoder")

# Hyperparams 
GLOBAL_SEED = 42 

TOKENIZER_MODEL = "Salesforce/codet5-base"
NL_ENCODER_BASE_MODEL = "roberta-base"
NL_DECODER_BASE_MODEL = "gpt2"

PL_ENCODER_BASE_MODEL = "microsoft/codebert-base"
VOCAB_SIZE = 32100

MAX_SEQUENCE_LENGTH = 128
PADDING = "max_length"

MAX_EPOCHS = 20
BATCH_SIZE = 32

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 1e-3

TEMPERATURE = 5

DATASET_NAME = "code_x_glue_ct_code_to_text"
PL = "python"

# Hardware
NUM_WORKERS = min(4, int(os.cpu_count() / 2))
AVAIL_GPUS = min(1, torch.cuda.device_count())

# Project
PROJECT_NAME = "CodeCG"