DEVICE = "cuda"

# phobert: PhoBertEncoder, trans: MaskedSoftBert, char_trans: CharWordTransformerEncoding
MODEL_NAME = 'phobert'

TRAINING_DATA_PERCENT=0.9

# Learning rate value
LEARNING_RATE=1e-5

# Continue train with the previous model
IS_CONTINUOUS_TRAIN=False

PATH_PRETRAINED_MODEL='output_model/'

# Using the default split data for training and validation
IS_SPLIT_INDEXES=False

# Number epoches
N_EPOCH=10

# lambda  for control the important level of detection loss value
LAMDA=0.6

# parameter controlled the contribution of the label 0 in the detection loss
PENALTY_VALUE=0.3

USE_DETECTION_CONTEXT=False

# TransformerEncoderDataset if True else PhoBertDataset
IS_TRANSFORMER = False

# Combine embedding char level to word embedding; 
ADD_CHAR_LEVEL = False

# For PhoBERT
IS_BERT = True

# Fine-tuned BERT pretrained model
FINE_TUNED=True

# Batch size samples
BATCH_SIZE=2

DOMAIN = 'general'
N_WORDS = {
    'luanvan': 13013,
    'general': 57981
}

# For preprocessing
BERT_PRETRAINED = 'vinai/phobert-base-v2'

MODEL_SAVE_PATH = 'output_models/'
SOFT_MASKED_BERT_MODEL = f'model_{DOMAIN}.pth'

PKL_PATH = 'input/general/'