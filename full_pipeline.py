import torch
from torch import nn
from transformers import AlbertConfig, AlbertModel, Trainer, TrainingArguments
from transformers.optimization import Lamb
import torch.nn.functional as F


# Model definition
class HierarchicalTransformerModel(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size):
        super(HierarchicalTransformerModel, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        
        self.char_embedding = nn.Embedding(self.char_vocab_size, 64)
        self.char_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=6)
        
        self.word_embedding = nn.Embedding(self.word_vocab_size, 128)
        self.word_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=192, nhead=8), num_layers=6)  # d_model=192 (64+128)
        
        self.fc_combined = nn.Linear(192, 256)
        
        self.binary_classifier = nn.Linear(256, 1)
        self.token_classifier = nn.Linear(256, self.word_vocab_size)
        
    def forward(self, char_input_ids, char_attention_mask, word_input_ids, word_attention_mask):
        # Character-level embeddings and transformer
        char_embeddings = self.char_embedding(char_input_ids)
        char_outputs = self.char_transformer(char_embeddings.transpose(0, 1), 
                                             src_key_padding_mask=~char_attention_mask.bool()).transpose(0, 1)
        char_avg_output = char_outputs.mean(dim=1)
        
        # Word-level embeddings
        word_embeddings = self.word_embedding(word_input_ids)
        
        # Concatenate average character output with word embeddings
        combined_embeddings = torch.cat((char_avg_output.unsqueeze(1).repeat(1, word_embeddings.size(1), 1), 
                                         word_embeddings), dim=-1)
        
        # Pass combined embeddings through the word-level encoder
        word_outputs = self.word_encoder(combined_embeddings.transpose(0, 1), 
                                         src_key_padding_mask=~word_attention_mask.bool()).transpose(0, 1)
        
        # Apply fully connected layer
        combined_outputs = self.fc_combined(word_outputs[:, 0, :])
        
        # Two tasks
        binary_logits = self.binary_classifier(combined_outputs)
        token_logits = self.token_classifier(combined_outputs)
        
        return binary_logits, token_logits

# Tokenizer
class CharTokenizer:
    def __init__(self, vocab_size=400):
        self.vocab_size = vocab_size
        self.vocab = {}
    
    def build_vocab(self, texts):
        chars = set(''.join(texts))
        self.vocab = {ch: idx for idx, ch in enumerate(chars)}
        self.vocab = {ch: idx for idx, ch in list(self.vocab.items())[:self.vocab_size]}
    
    def encode_plus(self, text, max_length):
        input_ids = [self.vocab.get(ch, 0) for ch in text[:max_length]]
        attention_mask = [1] * len(input_ids)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

class WordTokenizer:
    def __init__(self, vocab_size=60000):
        self.vocab_size = vocab_size
        self.vocab = {}
    
    def build_vocab(self, texts):
        words = ' '.join(texts).split()
        self.vocab = {word: idx for idx, word in enumerate(set(words))}
        self.vocab = {word: idx for idx, word in list(self.vocab.items())[:self.vocab_size]}
    
    def __call__(self, text, max_length=128, return_tensors="pt", padding='max_length', truncation=True):
        input_ids = [self.vocab.get(word, 0) for word in text.split()[:max_length]]
        attention_mask = [1] * len(input_ids)
        if padding == 'max_length':
            input_ids += [0] * (max_length - len(input_ids))
            attention_mask += [0] * (max_length - len(attention_mask))
        return {'input_ids': torch.tensor([input_ids]), 'attention_mask': torch.tensor([attention_mask])}

# Dataset
from torch.utils.data import Dataset

class SpellingDataset(Dataset):
    def __init__(self, input_texts, target_texts, char_tokenizer, word_tokenizer):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.char_tokenizer = char_tokenizer
        self.word_tokenizer = word_tokenizer
    
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        
        char_inputs = self.char_tokenizer.encode_plus(input_text, max_length=128)
        char_target = self.char_tokenizer.encode_plus(target_text, max_length=128)
        
        word_inputs = self.word_tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True)
        word_target = self.word_tokenizer(target_text, return_tensors="pt", padding='max_length', truncation=True)
        
        binary_labels = (word_inputs['input_ids'].squeeze() != word_target['input_ids'].squeeze()).long()
        
        return {
            'char_input_ids': torch.tensor(char_inputs['input_ids']),
            'char_attention_mask': torch.tensor(char_inputs['attention_mask']),
            'word_input_ids': word_inputs['input_ids'].squeeze(),
            'word_attention_mask': word_inputs['attention_mask'].squeeze(),
            'word_target_ids': word_target['input_ids'].squeeze(),
            'binary_labels': binary_labels
        }

# Custom Loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.token_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, binary_logits, token_logits, binary_labels, token_labels):
        # Binary classification loss
        binary_labels = binary_labels.float()
        binary_loss = self.binary_loss(binary_logits.view(-1), binary_labels.view(-1))
        
        # Token classification loss
        token_loss = self.token_loss(token_logits.view(-1, token_logits.size(-1)), token_labels.view(-1))
        # Only calculate token loss for tokens predicted as incorrect
        token_loss = token_loss * binary_labels.view(-1)
        token_loss = token_loss.mean()
        
        total_loss = binary_loss + token_loss
        return total_loss

# Training 
from transformers import DataCollatorForSeq2Seq
from torch.optim import AdamW
from transformers.optimization import get_scheduler

# Sample data
input_texts = ["Xin chào", "tôi là AI"]
target_texts = ["Xin chào", "tôi là AI"]

# Initialize tokenizers and dataset
char_tokenizer = CharTokenizer()
char_tokenizer.build_vocab(input_texts + target_texts)
word_tokenizer = WordTokenizer()
word_tokenizer.build_vocab(input_texts + target_texts)

dataset = SpellingDataset(input_texts, target_texts, char_tokenizer, word_tokenizer)

# Initialize model
model = HierarchicalTransformerModel(char_vocab_size=400, word_vocab_size=60000)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=word_tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1.76e-3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    lr_scheduler_type='polynomial',
    warmup_steps=0,
    power=1.0,
)

# Define the compute_metrics function
def compute_metrics(eval_pred):
    binary_logits, token_logits = eval_pred
    binary_labels, token_labels = binary_logits[:, 1], token_logits[:, 1]
    predictions = (binary_logits.sigmoid() > 0.5).float()
    accuracy = (predictions == binary_labels).float().mean()
    return {"accuracy": accuracy.item()}

# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        char_input_ids = inputs['char
