# Suppress urllib3 OpenSSL warnings at the very top
import warnings
import os
import sys

# Suppress urllib3 warnings globally
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# Set environment variable to enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Create the refactored code with BERT-based model, enhanced logging, colorful output, and improved checkpointing
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
import glob
import argparse
import logging
import matplotlib.pyplot as plt
import re
import json
import time
from datetime import datetime, timedelta
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# ========== M3 ULTRA OPTIMIZATIONS ==========
# Enable MPS (Metal Performance Shaders) for M3 Ultra
if torch.backends.mps.is_available():
    try:
        torch.backends.mps.empty_cache()
        print("🚀 M3 Ultra GPU (MPS) detected and enabled!")
    except AttributeError:
        # Handle older PyTorch versions that don't have mps.empty_cache()
        print("🚀 M3 Ultra GPU (MPS) detected and enabled!")
else:
    print("⚠️  M3 Ultra GPU not detected, using CPU")

# Optimize for M3 Ultra
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ========== WANDB IMPORT ==========
try:
    import wandb
    WANDB_AVAILABLE = True
    WANDB_API_KEY = "ae09b54e3de87193441e1c5d78777a22cc817458"
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with 'pip install wandb' for enhanced logging.")

# ========== COLORFUL LOGGING SETUP ==========
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
        'BOLD': '\033[1m',       # Bold
        'BLUE': '\033[34m',      # Blue
        'PURPLE': '\033[95m',    # Purple
    }

    def format(self, record):
        # Add color to levelname
        levelname_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{levelname_color}{record.levelname}{self.COLORS['RESET']}"

        # Add color to timestamp
        log_time = self.formatTime(record, self.datefmt)
        colored_time = f"{self.COLORS['BLUE']}{log_time}{self.COLORS['RESET']}"

        # Format the message
        if hasattr(record, 'color'):
            message = f"{record.color}{record.getMessage()}{self.COLORS['RESET']}"
        else:
            message = record.getMessage()

        return f"{colored_time} [{record.levelname}] {message}"

# ========== OUTPUT DIRECTORY ==========
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== ENHANCED LOGGING SETUP ==========
def setup_logging():
    """Setup enhanced logging with colors and detailed formatting"""

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = ColoredFormatter(
        datefmt="%H:%M:%S"
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler for detailed logs
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "email_classification_training.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler for colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# ========== PROGRESS TRACKING ==========
class TrainingProgress:
    """Class to track and save training progress"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "email_classification_progress.json")
        self.start_time = time.time()
        self.epoch_times = []

    def save_progress(self, epoch, train_loss, val_loss, val_accuracy, best_val_loss,
                     train_losses, val_losses, val_accuracies, model_params):
        """Save detailed progress information"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = model_params['num_epochs'] - (epoch + 1)
            eta = remaining_epochs * avg_epoch_time
        else:
            avg_epoch_time = 0
            eta = 0

        progress_data = {
            'current_epoch': epoch,
            'total_epochs': model_params['num_epochs'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'best_val_loss': best_val_loss,
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'average_epoch_time': avg_epoch_time,
            'eta_seconds': eta,
            'eta_formatted': str(timedelta(seconds=int(eta))),
            'timestamp': datetime.now().isoformat(),
            'train_losses_history': train_losses,
            'val_losses_history': val_losses,
            'val_accuracy_history': val_accuracies,
            'model_parameters': model_params,
            'epoch_times': self.epoch_times
        }

        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    def load_progress(self):
        """Load progress from file if exists"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return None

    def add_epoch_time(self, epoch_time):
        """Add epoch time for ETA calculation"""
        self.epoch_times.append(epoch_time)

# ========== DATA CLEANING ==========
def clean_text(text):
    """Enhanced text cleaning with detailed logging"""
    original_length = len(text)

    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'#+', '', text)  # Remove markdown headers
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    text = re.sub(r'[*_~]', '', text)  # Remove markdown formatting
    text = re.sub(r'[\[\]<>]', '', text)  # Remove brackets
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()

    cleaned_length = len(text)
    reduction_percent = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0

    if reduction_percent > 50:
        logger.debug(f"High text reduction: {reduction_percent:.1f}% (from {original_length} to {cleaned_length} chars)")

    return text

def clean_sentences(sentences):
    """Clean sentences with progress tracking"""
    logger.info(f"🧹 Cleaning {len(sentences):,} sentences...")

    cleaned = []
    empty_sentences = 0

    for i, sent in enumerate(tqdm(sentences, desc="Cleaning sentences", disable=len(sentences) < 10000)):
        if isinstance(sent, str):
            sent_str = sent
        else:
            sent_str = ' '.join(sent)
        
        sent_str = clean_text(sent_str)
        words = [w for w in sent_str.split() if w]

        if words:
            cleaned.append(words)
        else:
            empty_sentences += 1

        if i % 50000 == 0 and i > 0:
            logger.debug(f"Processed {i:,} sentences, {empty_sentences} empty after cleaning")

    logger.info(f"✅ Cleaning complete: {len(cleaned):,} sentences retained, {empty_sentences:,} empty sentences removed")
    return cleaned

# ========== EMAIL CLASSIFICATION DATASET ==========
class EmailClassificationDataset(Dataset):
    def __init__(self, texts, labels, word2idx, seq_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.cls_token = word2idx['[CLS]']
        self.sep_token = word2idx['[SEP]']
        self.pad_token = word2idx['[PAD]']
        self.unk_token = word2idx['[UNK]']

        logger.debug(f"Created EmailClassificationDataset with {len(texts):,} samples, seq_len={seq_len}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert to token IDs
        tokens = [self.cls_token]
        for word in text:
            tokens.append(self.word2idx.get(word, self.unk_token))
        tokens.append(self.sep_token)

        # Truncate or pad to seq_len
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens.extend([self.pad_token] * (self.seq_len - len(tokens)))

        # Create attention mask
        attention_mask = [1 if token != self.pad_token else 0 for token in tokens]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def build_bert_vocab(sentences, min_freq=2):
    """Build vocabulary with BERT special tokens"""
    logger.info(f"🔤 Building BERT vocabulary with min_freq={min_freq}...")

    counter = Counter()
    total_words = 0

    for sent in tqdm(sentences, desc="Counting words", disable=len(sentences) < 10000):
        if isinstance(sent, list):
            counter.update(sent)
            total_words += len(sent)
        else:
            words = sent.split()
            counter.update(words)
            total_words += len(words)

    # Filter by frequency
    vocab_filtered = [w for w, c in counter.items() if c >= min_freq]

    # BERT special tokens
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    vocab = special_tokens + vocab_filtered

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    # Statistics
    total_unique_words = len(counter)
    vocab_size = len(vocab)
    coverage = sum(c for w, c in counter.items() if c >= min_freq) / total_words * 100

    logger.info(f"📊 BERT Vocabulary Statistics:")
    logger.info(f"   Total words: {total_words:,}")
    logger.info(f"   Unique words: {total_unique_words:,}")
    logger.info(f"   Vocabulary size (after filtering): {vocab_size:,}")
    logger.info(f"   Coverage: {coverage:.2f}%")
    logger.info(f"   Special tokens: {special_tokens}")
    logger.info(f"   Most common words: {counter.most_common(10)}")

    return vocab, word2idx, idx2word

# ========== MPS-COMPATIBLE BERT MODEL FOR CLASSIFICATION ==========
class MPSCompatibleBERTClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=8, num_heads=8,
                 intermediate_size=512, max_seq_len=64, dropout=0.1, pad_token_id=0, num_classes=2):
        super().__init__()

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Custom transformer layers (MPS compatible)
        self.transformer_layers = nn.ModuleList([
            MPSCompatibleTransformerLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        # Classification head (instead of MLM head)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.num_classes = num_classes

        # Initialize weights for better convergence
        self.apply(self._init_weights)

        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"🏗️  MPS-Compatible BERT Email Classifier Architecture:")
        logger.info(f"   Vocabulary size: {vocab_size:,}")
        logger.info(f"   Hidden size: {hidden_size}")
        logger.info(f"   Number of layers: {num_layers}")
        logger.info(f"   Number of heads: {num_heads}")
        logger.info(f"   Intermediate size: {intermediate_size}")
        logger.info(f"   Max sequence length: {max_seq_len}")
        logger.info(f"   Number of classes: {num_classes}")
        logger.info(f"   Dropout: {dropout}")
        logger.info(f"   Pad token ID: {pad_token_id}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    def _init_weights(self, module):
        """Initialize weights for better convergence"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        batch_size, seq_len = input_ids.size()

        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Token type IDs (default to 0 for single sentence)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Use [CLS] token for classification
        cls_output = hidden_states[:, 0]  # First token is [CLS]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }

class MPSCompatibleTransformerLayer(nn.Module):
    """MPS-compatible transformer layer that avoids problematic operations"""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout):
        super().__init__()
        self.attention = MPSCompatibleMultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attention_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ff_output)

        return hidden_states

class MPSCompatibleMultiHeadAttention(nn.Module):
    """MPS-compatible multi-head attention that avoids nested tensor operations"""

    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Linear projections
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to the right shape and type
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * -10000.0  # Convert to large negative values
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, V)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # Output projection
        output = self.output_projection(context)

        return output

# ========== EMAIL DATA LOADING ==========
def load_email_data(business_folder, spam_folder):
    """Load emails from folders and create labels"""
    texts = []
    labels = []

    # Load business emails (label = 0)
    logger.info(f"📧 Loading business emails from: {business_folder}")
    if os.path.exists(business_folder):
        business_files = [f for f in os.listdir(business_folder) if f.endswith('.txt')]
        for filename in tqdm(business_files, desc="Loading business emails"):
            filepath = os.path.join(business_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty emails
                        texts.append(content)
                        labels.append(0)  # Business email
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")

    # Load spam emails (label = 1)
    logger.info(f"📧 Loading spam emails from: {spam_folder}")
    if os.path.exists(spam_folder):
        spam_files = [f for f in os.listdir(spam_folder) if f.endswith('.txt')]
        for filename in tqdm(spam_files, desc="Loading spam emails"):
            filepath = os.path.join(spam_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty emails
                        texts.append(content)
                        labels.append(1)  # Spam email
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")

    business_count = len([l for l in labels if l == 0])
    spam_count = len([l for l in labels if l == 1])
    
    logger.info(f"✅ Email loading complete:")
    logger.info(f"   Business emails: {business_count:,}")
    logger.info(f"   Spam emails: {spam_count:,}")
    logger.info(f"   Total emails: {len(texts):,}")
    logger.info(f"   Class balance: {business_count/(business_count+spam_count)*100:.1f}% business, {spam_count/(business_count+spam_count)*100:.1f}% spam")

    return texts, labels

def split_email_data(texts, labels, train_ratio=0.8, val_ratio=0.1):
    """Split email data into train, validation, and test sets with stratification"""
    # Combine and shuffle while maintaining class balance
    combined = list(zip(texts, labels))
    
    # Separate by class
    business_emails = [(t, l) for t, l in combined if l == 0]
    spam_emails = [(t, l) for t, l in combined if l == 1]
    
    # Shuffle each class
    random.shuffle(business_emails)
    random.shuffle(spam_emails)
    
    # Split each class
    def split_class_data(class_data, train_r, val_r):
        n = len(class_data)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        
        train = class_data[:train_end]
        val = class_data[train_end:val_end]
        test = class_data[val_end:]
        
        return train, val, test
    
    business_train, business_val, business_test = split_class_data(business_emails, train_ratio, val_ratio)
    spam_train, spam_val, spam_test = split_class_data(spam_emails, train_ratio, val_ratio)
    
    # Combine and shuffle
    train_combined = business_train + spam_train
    val_combined = business_val + spam_val
    test_combined = business_test + spam_test
    
    random.shuffle(train_combined)
    random.shuffle(val_combined)
    random.shuffle(test_combined)
    
    # Separate texts and labels
    train_texts, train_labels = zip(*train_combined) if train_combined else ([], [])
    val_texts, val_labels = zip(*val_combined) if val_combined else ([], [])
    test_texts, test_labels = zip(*test_combined) if test_combined else ([], [])

    logger.info(f"📊 Email data split:")
    logger.info(f"   Train set: {len(train_texts):,} samples ({len([l for l in train_labels if l == 0])} business, {len([l for l in train_labels if l == 1])} spam)")
    logger.info(f"   Validation set: {len(val_texts):,} samples ({len([l for l in val_labels if l == 0])} business, {len([l for l in val_labels if l == 1])} spam)")
    logger.info(f"   Test set: {len(test_texts):,} samples ({len([l for l in test_labels if l == 0])} business, {len([l for l in test_labels if l == 1])} spam)")

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

# ========== ENHANCED CHECKPOINTING ==========
def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path,
                   train_losses, val_losses, val_accuracies, progress_tracker):
    """Enhanced checkpoint saving with more metadata"""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'timestamp': datetime.now().isoformat(),
        'epoch_times': progress_tracker.epoch_times,
        'total_training_time': time.time() - progress_tracker.start_time
    }

    torch.save(checkpoint_data, checkpoint_path)

    # Also save a backup checkpoint
    backup_path = checkpoint_path.replace('.pt', f'_epoch_{epoch+1}.pt')
    torch.save(checkpoint_data, backup_path)

    logger.info(f"💾 Checkpoint saved at epoch {epoch+1} (backup: {os.path.basename(backup_path)})")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Enhanced checkpoint loading with validation"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
        return 0, float('inf'), [], [], []

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])

        logger.info(f"✅ Checkpoint loaded from epoch {epoch+1}")
        logger.info(f"   Best validation loss: {best_val_loss:.4f}")
        logger.info(f"   Training history: {len(train_losses)} epochs")

        return epoch, best_val_loss, train_losses, val_losses, val_accuracies

    except Exception as e:
        logger.error(f"❌ Error loading checkpoint: {e}")
        return 0, float('inf'), [], [], []

def should_pause():
    return os.path.exists("PAUSE.TXT")

# ========== CLASSIFICATION METRICS ==========
def calculate_metrics(predictions, labels):
    """Calculate classification metrics"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ========== OPTIMIZED TRAINING LOOP FOR CLASSIFICATION ==========
def train_epoch(model, loader, optimizer, criterion, device, epoch, best_val_loss,
               checkpoint_path, progress_tracker, train_losses, val_losses, val_accuracies,
               wandb_enabled=False):
    """Optimized training epoch for email classification"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    num_batches = len(loader)
    epoch_start_time = time.time()

    # Create progress bar
    pbar = tqdm(loader, desc=f"🚂 Epoch {epoch+1} Training",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for batch_idx, batch in enumerate(pbar):
        batch_start_time = time.time()

        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * input_ids.size(0)

        # Get predictions for metrics
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_labels = labels.cpu().numpy()
        
        predictions.extend(batch_predictions)
        true_labels.extend(batch_labels)

        # Calculate current accuracy
        current_acc = accuracy_score(true_labels, predictions) * 100

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{batch_loss:.4f}',
            'Avg': f'{total_loss / ((batch_idx + 1) * loader.batch_size):.4f}',
            'Acc': f'{current_acc:.1f}%'
        })

        # Log to wandb
        if wandb_enabled and WANDB_AVAILABLE:
            wandb.log({
                "train/batch_loss": batch_loss,
                "train/batch_accuracy": current_acc,
                "epoch": epoch+1
            })

        # Log detailed batch info every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            batch_time = time.time() - batch_start_time
            logger.debug(f"Batch {batch_idx}/{num_batches}: loss={batch_loss:.4f}, acc={current_acc:.1f}%, time={batch_time:.3f}s")

        # Check for pause signal
        if should_pause():
            logger.warning("⏸️  Pause signal detected. Saving checkpoint and exiting...")
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path,
                          train_losses, val_losses, val_accuracies, progress_tracker)
            logger.info("You can now safely restart your machine. To resume, run with --resume.")
            exit(0)

    pbar.close()
    epoch_time = time.time() - epoch_start_time
    progress_tracker.add_epoch_time(epoch_time)

    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(predictions, true_labels)

    logger.info(f"📈 Training epoch {epoch+1} completed in {epoch_time:.1f}s")
    logger.info(f"   Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']:.4f}")

    return avg_loss, metrics

def eval_epoch(model, loader, criterion, device, epoch, wandb_enabled=False):
    """Optimized evaluation epoch for email classification"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    num_samples = 0

    pbar = tqdm(loader, desc=f"🔍 Epoch {epoch+1} Validation",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']

            batch_loss = loss.item()
            total_loss += batch_loss * input_ids.size(0)
            num_samples += input_ids.size(0)

            # Get predictions
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

            current_acc = accuracy_score(true_labels, predictions) * 100
            pbar.set_postfix({'Val Loss': f'{batch_loss:.4f}', 'Acc': f'{current_acc:.1f}%'})

    pbar.close()
    avg_loss = total_loss / num_samples
    metrics = calculate_metrics(predictions, true_labels)

    # Log to wandb
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.log({
            "val/loss": avg_loss,
            "val/accuracy": metrics['accuracy'],
            "val/f1": metrics['f1'],
            "val/precision": metrics['precision'],
            "val/recall": metrics['recall'],
            "epoch": epoch+1
        })

    logger.info(f"📊 Validation epoch {epoch+1} completed")
    logger.info(f"   Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']:.4f}")

    return avg_loss, metrics, predictions, true_labels

# ========== ENHANCED VISUALIZATION ==========
def plot_metrics(train_losses, val_losses, val_accuracies, output_dir):
    """Enhanced plotting for email classification metrics"""
    epochs = range(1, len(train_losses) + 1)

    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Email Classification)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    val_acc_values = [m['accuracy']*100 if isinstance(m, dict) else m for m in val_accuracies]
    ax2.plot(epochs, val_acc_values, label='Val Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy (Email Classification)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # F1 Score plot
    val_f1_values = [m['f1'] if isinstance(m, dict) else 0 for m in val_accuracies]
    if any(val_f1_values):
        ax3.plot(epochs, val_f1_values, label='Val F1 Score', color='purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Validation F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Learning curve (log scale)
    ax4.semilogy(epochs, train_losses, label='Train Loss (log)', color='blue', linewidth=2)
    ax4.semilogy(epochs, val_losses, label='Val Loss (log)', color='red', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Learning Curves (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "email_classification_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"📊 Email classification training curves saved to {plot_path}")

def plot_confusion_matrix(true_labels, predictions, class_names=['Business', 'Spam'], output_dir=OUTPUT_DIR):
    """Plot confusion matrix for email classification"""
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Email Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'email_classification_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"📊 Confusion matrix saved to {cm_path}")

# ========== EMAIL CLASSIFICATION INFERENCE ==========
def classify_email(model, word2idx, text, device, seq_len=64):
    """Classify a single email"""
    model.eval()

    # Preprocess text
    words = clean_text(text).lower().split()
    
    # Convert to token IDs
    cls_token = word2idx['[CLS]']
    sep_token = word2idx['[SEP]']
    pad_token = word2idx['[PAD]']
    unk_token = word2idx['[UNK]']
    
    tokens = [cls_token]
    for word in words:
        tokens.append(word2idx.get(word, unk_token))
    tokens.append(sep_token)
    
    # Truncate or pad
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    else:
        tokens.extend([pad_token] * (seq_len - len(tokens)))
    
    # Create attention mask
    attention_mask = [1 if token != pad_token else 0 for token in tokens]
    
    # Convert to tensors
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()

    class_names = ['Business', 'Spam']
    confidence = probabilities[0][prediction].item()

    return {
        'prediction': class_names[prediction],
        'confidence': confidence,
        'probabilities': {
            'Business': probabilities[0][0].item(),
            'Spam': probabilities[0][1].item()
        }
    }

# ========== MAIN SCRIPT ==========
def main():
    parser = argparse.ArgumentParser(description="MPS-Compatible BERT Email Classification for M3 Ultra")
    parser.add_argument('--pause', action='store_true', help='Pause training at the end of the next batch')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--restart', action='store_true', help='Restart training from scratch')
    parser.add_argument('--business-path', type=str, default="/Users/harsha/Desktop/Dev/bertLLM/generated_business_emails_txt",
                       help='Path to business emails directory')
    parser.add_argument('--spam-path', type=str, default="/Users/harsha/Desktop/Dev/bertLLM/generated_emails_txt",
                       help='Path to spam emails directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='email-classification-bert', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--wandb-name', type=str, default=None, help='W&B run name')
    args = parser.parse_args()

    # Print startup banner
    logger.info("=" * 80)
    logger.info("🚀 MPS-COMPATIBLE BERT EMAIL CLASSIFICATION FOR M3 ULTRA")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")

    # --- OPTIMIZED Hyperparameters for M3 Ultra ---
    seq_len = args.seq_len          # 64 - optimized for emails
    batch_size = args.batch_size    # 32 - optimized for classification
    hidden_size = 128               # Reduced from 512
    num_heads = 8                   # Optimized
    num_layers = 8                  # Reduced from 12
    intermediate_size = 512         # Reduced from 2048
    dropout = 0.1
    num_epochs = args.epochs        # 10 epochs for classification
    lr = args.lr                    # 1e-4 for classification
    min_freq = 2
    num_workers = 0                 # Set to 0 for MPS compatibility
    num_classes = 2                 # Business vs Spam
    checkpoint_path = os.path.join(OUTPUT_DIR, "email_classification_checkpoint.pt")
    best_model_path = os.path.join(OUTPUT_DIR, "best_email_classification_model.pt")

    # Initialize progress tracker
    progress_tracker = TrainingProgress(OUTPUT_DIR)

    # Log hyperparameters
    model_params = {
        'seq_len': seq_len, 'batch_size': batch_size, 'hidden_size': hidden_size,
        'num_heads': num_heads, 'num_layers': num_layers, 'intermediate_size': intermediate_size,
        'dropout': dropout, 'num_epochs': num_epochs, 'lr': lr, 'min_freq': min_freq,
        'num_classes': num_classes
    }

    logger.info("⚡ EMAIL CLASSIFICATION Hyperparameters:")
    for key, value in model_params.items():
        logger.info(f"   {key}: {value}")

    # --- WANDB INITIALIZATION ---
    wandb_enabled = args.wandb and WANDB_AVAILABLE
    if wandb_enabled:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=model_params
        )
        logger.info("✅ Weights & Biases logging enabled")
    elif args.wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️  wandb requested but not available. Please install with 'pip install wandb'")
        wandb_enabled = False
    else:
        logger.info("⏭️  Weights & Biases logging disabled")

    # --- Email Data Loading ---
    logger.info("📧 Loading email data...")
    texts, labels = load_email_data(args.business_path, args.spam_path)

    if len(texts) == 0:
        logger.error("❌ No emails found! Please check your folder paths.")
        return

    # --- Data Cleaning ---
    logger.info("🧹 Cleaning email texts...")
    cleaned_texts = []
    for text in tqdm(texts, desc="Cleaning emails"):
        cleaned_text = clean_text(text)
        words = cleaned_text.lower().split()
        cleaned_texts.append(words)

    # --- Build BERT Vocab ---
    vocab, word2idx, idx2word = build_bert_vocab(cleaned_texts, min_freq=min_freq)
    vocab_size = len(vocab)
    pad_token_id = word2idx['[PAD]']

    # --- Data split ---
    train_data, val_data, test_data = split_email_data(cleaned_texts, labels)

    # --- Device selection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Using M3 Ultra GPU (MPS) with fallback enabled")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🖥️  Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device("cpu")
        logger.info("🖥️  Using CPU")

    # --- Datasets and Loaders ---
    train_dataset = EmailClassificationDataset(train_data[0], train_data[1], word2idx, seq_len)
    val_dataset = EmailClassificationDataset(val_data[0], val_data[1], word2idx, seq_len)
    test_dataset = EmailClassificationDataset(test_data[0], test_data[1], word2idx, seq_len)

    logger.info(f"📦 Dataset sizes: {len(train_dataset):,} train, {len(val_dataset):,} val, {len(test_dataset):,} test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # --- Model, Optimizer, Loss ---
    model = MPSCompatibleBERTClassifier(
        vocab_size, hidden_size, num_layers, num_heads,
        intermediate_size, seq_len, dropout, pad_token_id, num_classes
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    # --- Checkpoint Handling ---
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []

    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss, train_losses, val_losses, val_accuracies = load_checkpoint(
            model, optimizer, checkpoint_path)
        start_epoch += 1
        logger.info(f"🔄 Resuming from epoch {start_epoch}")
    elif args.restart:
        logger.info("🔄 Restarting training from scratch")

    # --- Training Loop ---
    logger.info("🎯 Starting email classification training...")
    logger.info("=" * 80)

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            logger.info(f"\n🔄 EPOCH {epoch+1}/{num_epochs}")
            logger.info("-" * 50)

            # Training
            train_loss, train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device,
                epoch, best_val_loss, checkpoint_path, progress_tracker,
                train_losses, val_losses, val_accuracies, wandb_enabled
            )

            # Validation
            val_loss, val_metrics, val_predictions, val_true_labels = eval_epoch(
                model, val_loader, criterion, device, epoch, wandb_enabled
            )

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Update metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_metrics)

            # Log epoch metrics to wandb
            if wandb_enabled and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch+1,
                    "train/epoch_loss": train_loss,
                    "train/epoch_accuracy": train_metrics['accuracy'],
                    "train/epoch_f1": train_metrics['f1'],
                    "val/epoch_loss": val_loss,
                    "val/epoch_accuracy": val_metrics['accuracy'],
                    "val/epoch_f1": val_metrics['f1'],
                    "best_val_loss": best_val_loss,
                    "learning_rate": current_lr
                })

            # Check for best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                if wandb_enabled and WANDB_AVAILABLE:
                    wandb.save(best_model_path)
                logger.info("🏆 New best model saved!")

            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\n📊 EPOCH {epoch+1} SUMMARY:")
            logger.info(f"   Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']*100:.2f}% | F1: {train_metrics['f1']:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f} {'🏆 (BEST!)' if is_best else ''} | Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1']:.4f}")
            logger.info(f"   Epoch Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")

            # Save progress
            progress_tracker.save_progress(
                epoch, train_loss, val_loss, val_metrics['accuracy'], best_val_loss,
                train_losses, val_losses, val_accuracies, model_params
            )

            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path,
                          train_losses, val_losses, val_accuracies, progress_tracker)

            # Clear MPS cache
            if device.type == 'mps':
                try:
                    torch.backends.mps.empty_cache()
                except AttributeError:
                    pass

            if args.pause:
                logger.info("⏸️  Pausing training as requested.")
                break

    except KeyboardInterrupt:
        logger.info("\n⏹️  Training interrupted by user")
        save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path,
                      train_losses, val_losses, val_accuracies, progress_tracker)
        if wandb_enabled and WANDB_AVAILABLE:
            wandb.finish()
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        if wandb_enabled and WANDB_AVAILABLE:
            wandb.finish()
        raise

    logger.info("\n🎉 Email classification training completed!")
    logger.info("=" * 80)

    # --- Final Test Evaluation ---
    logger.info("🧪 Final test evaluation...")
    test_loss, test_metrics, test_predictions, test_true_labels = eval_epoch(
        model, test_loader, criterion, device, num_epochs, wandb_enabled
    )

    logger.info(f"📊 FINAL TEST RESULTS:")
    logger.info(f"   Test Loss: {test_loss:.4f}")
    logger.info(f"   Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"   Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"   Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"   Test Recall: {test_metrics['recall']:.4f}")

    # --- Visualization ---
    if train_losses:
        plot_metrics(train_losses, val_losses, val_accuracies, OUTPUT_DIR)
        plot_confusion_matrix(test_true_labels, test_predictions, output_dir=OUTPUT_DIR)

        if wandb_enabled and WANDB_AVAILABLE:
            wandb.log({
                "test/final_accuracy": test_metrics['accuracy'],
                "test/final_f1": test_metrics['f1'],
                "test/final_precision": test_metrics['precision'],
                "test/final_recall": test_metrics['recall'],
                "training_curves": wandb.Image(os.path.join(OUTPUT_DIR, "email_classification_training_curves.png")),
                "confusion_matrix": wandb.Image(os.path.join(OUTPUT_DIR, "email_classification_confusion_matrix.png"))
            })

    # --- Save Model Artifacts ---
    logger.info("💾 Saving model artifacts...")
    artifacts_dir = os.path.join(OUTPUT_DIR, "email_classifier_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save vocabulary
    with open(os.path.join(artifacts_dir, "word2idx.json"), "w") as f:
        json.dump(word2idx, f, indent=2)
    
    with open(os.path.join(artifacts_dir, "idx2word.json"), "w") as f:
        json.dump(idx2word, f, indent=2)

    # Save model config
    model_config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'intermediate_size': intermediate_size,
        'max_seq_len': seq_len,
        'dropout': dropout,
        'pad_token_id': pad_token_id,
        'num_classes': num_classes
    }
    
    with open(os.path.join(artifacts_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Save final results
    final_results = {
        'test_metrics': test_metrics,
        'model_config': model_config,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    }
    
    with open(os.path.join(artifacts_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"✅ Model artifacts saved to {artifacts_dir}")

    # --- Interactive Email Classification ---
    logger.info("\n🎮 Interactive email classification mode!")
    logger.info("💡 Enter email text to classify as Business or Spam")
    logger.info("💡 Press Ctrl+C or Ctrl+D to exit")
    logger.info("-" * 50)

    try:
        while True:
            try:
                user_input = input("\n📧 Email text: ").strip()
                if not user_input:
                    logger.warning("⚠️  Please enter some email text.")
                    continue

                logger.info("🤖 Classifying email...")
                result = classify_email(model, word2idx, user_input, device, seq_len)

                print("\n" + "="*60)
                print(f"📧 Email: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
                print(f"🤖 Classification: {result['prediction']}")
                print(f"🎯 Confidence: {result['confidence']:.3f}")
                print(f"📊 Probabilities:")
                print(f"   Business: {result['probabilities']['Business']:.3f}")
                print(f"   Spam: {result['probabilities']['Spam']:.3f}")
                print("="*60)

                if wandb_enabled and WANDB_AVAILABLE:
                    wandb.log({
                        "inference/prediction": result['prediction'],
                        "inference/confidence": result['confidence']
                    })

            except EOFError:
                break
            except KeyboardInterrupt:
                break

    except Exception as e:
        logger.error(f"❌ Inference error: {e}")

    # Finish wandb run
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.finish()

    logger.info("\n👋 Email classification complete! Thanks for using the system!")
    logger.info(f"📊 All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
