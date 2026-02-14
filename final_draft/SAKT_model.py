#installations:
#!pip  install coloredlogs
#!pip install prefetch_generator

#JEREMY removed above cell since only bottom code is needed to operate in Google Colab (we are not using Kaggle)

#utils.py

import torch
import numpy as np
import pandas as pd #HDKIM
from torch.autograd import Variable

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(x, pad):
    "Create a mask to hide padding and future words."
    mask = torch.unsqueeze((x!=pad), -1)

    tgt_mask = mask & Variable(
        subsequent_mask(x.size(-1)).type_as(mask.data))
    #         print('tgt_mask size after: ', tgt_mask.size())
    return tgt_mask

#JEREMY adding non-cuda functionality
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# multihead_attn.py

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import copy
from torch.nn import LayerNorm


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, key_masks=None, query_masks=None, future_masks=None, dropout=None, infer=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    layernorm = LayerNorm(d_k).to(device)
    # query shape = [nbatches, h, T_q, d_k]       key shape = [nbatches, h, T_k, d_k] == value shape
    # scores shape = [nbatches, h, T_q, T_k]  == p_attn shape
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # if key_masks is not None:
    #     scores = scores.masked_fill(key_masks.unsqueeze(1).cuda() == 0, -1e9)
    if future_masks is not None:
        scores = scores.masked_fill(future_masks.unsqueeze(0).to(device) == 0, -1e9)


    p_attn = F.softmax(scores, dim=-1)
    outputs = p_attn
    # if query_masks is not None:
    #     outputs = outputs * query_masks.unsqueeze(1)
    if dropout is not None:
        outputs = dropout(outputs)
    outputs = torch.matmul(outputs, value)

    outputs += query
    return layernorm(outputs), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.2, infer=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = LayerNorm(d_model).to(device)
        self.infer = infer

    def forward(self, query, key, value, key_masks=None, query_masks=None, future_masks=None):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [F.relu(l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2), inplace=True)
             for l, x in zip(self.linears, (query, key, value))]
        # k v shape = [nbatches, h, T_k, d_k],  d_k * h = d_model
        # q shape = [nbatches, h, T_q, d_k]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, query_masks=query_masks,
                                 key_masks=key_masks, future_masks=future_masks, dropout=self.dropout, infer=self.infer)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.layernorm(x)
    
#wordtest.py

import logging
import coloredlogs
import pickle

logger = logging.getLogger('__file__')
coloredlogs.install(level='INFO', logger=logger)

def pickle_io(path, mode='r', obj=None):
    """
    Convinient pickle load and dump.
    """
    if mode in ['rb', 'r']:
        logger.info("Loading obj from {}...".format(path))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Load obj successfully!")
        return obj
    elif mode in ['wb', 'w']:
        logger.info("Dumping obj to {}...".format(path))
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Dump obj successfully!")

class WordTestResource(object):

    def __init__(self, resource_path, verbose=False):

        resource = pickle_io(resource_path, mode='r')

        self.id2index = resource['id2index']
        self.index2id = resource['index2id']
        self.num_skills = len(self.id2index)

        if verbose:
            self.word2id = resource['word2id']
            self.id2all = resource['id2all']
            # rank0 already be set to a large number
            self.words_by_rank = resource['words_by_rank']
            self.pos2id = resource['pos2id']
            self.words_by_rank.sort(key=lambda x: x[u'rank'])
            self.id_by_rank = [x[u'word_id'] for x in self.words_by_rank]

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

#config.py

class DefaultConfig(object):
    model = 'SAKT'
    #train_data = "../input/assist2015files/assist2015_train.csv"  # train_data_path
    #test_data = "../input/assist2015files/assist2015_test.csv"
    batch_size = 4 #HDKIM 256
    state_size = 200
    num_heads = 5
    max_len = 50
    dropout = 0.1
    max_epoch = 5 #10
    lr = 3e-3
    lr_decay = 0.9
    max_grad_norm = 1.0
    weight_decay = 0  # l2正则化因子

    #JEREMY EDIT - adding parameters for typing tool
    timestamp_buckets = 10 #for using timestamps as a difficulty measurement
    lambda_time = 0.3 #for using time loss weight
    word_padding = 0
    log_time = True
    #************************

opt = DefaultConfig()

# dataset.py

import csv
import torch
import time
import itertools
import numpy as np
#from config import DefaultConfig
#from wordtest import WordTestResource
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

import joblib #HDKIM

class Data(Dataset):
    #HDKIM def __init__(self, train=True):
    def __init__(self, df, train=True):
        start_time = time.time()
        #HDKIM if train:
        #HDKIM    fileName = opt.train_data
        #HDKIM else:
        #HDKIM     fileName = opt.test_data
        self.students = []
        self.max_skill_num = 0
        begin_index = 1e9
        
        #HDKIM with open(fileName, "r") as csvfile:
            #HDKIM for num_ques, ques, ans in itertools.zip_longest(*[csvfile] * 3):
                #HDKIM num_ques = int(num_ques.strip().strip(','))
                #HDKIM ques = [int(q) for q in ques.strip().strip(',').split(',')]
                #HDKIM ans = [int(a) for a in ans.strip().strip(',').split(',')]
        for index, row in df.iterrows():
                num_ques = int(row['num_ques'])
                #print(row['num_ques'])
                #print(row['ques'])
                #print(row['ans'])
                ques = [int(q) for q in row['ques']]
                ans = [int(a) for a in row['ans']]
                
                #JEREMY EDIT - adding times to also be returned as a tensor for SAKT
                times = [float(t) for t in row['times']]
                #************************

                tmp_max_skill = max(ques)
                tmp_min_skill = min(ques)
                begin_index = min(tmp_min_skill, begin_index)
                self.max_skill_num = max(tmp_max_skill, self.max_skill_num)
                
                #HDKIM if (num_ques <= 2):
                #HDKIM     continue
                #HDKIM elif num_ques <= opt.max_len:
                #HDKIM if num_ques <= opt.max_len:
                '''
                if num_ques <= opt.max_len:
                    problems = np.zeros(opt.max_len, dtype=np.int64)
                    correct = np.ones(opt.max_len, dtype=np.int64)
                    problems[-num_ques:] = ques[-num_ques:]
                    correct[-num_ques:] = ans[-num_ques:]
                    self.students.append((num_ques, problems, correct))
                else:
                    start_idx = 0
                    while opt.max_len + start_idx <= num_ques:
                        problems = np.array(ques[start_idx:opt.max_len + start_idx])
                        correct = np.array(ans[start_idx:opt.max_len + start_idx])
                        tup = (opt.max_len, problems, correct)
                        start_idx += opt.max_len
                        self.students.append(tup)
                    left_num_ques = num_ques - start_idx
                ''' 
                #HDKIM
                # first part of the student
                copy_len = opt.max_len - 1
                if copy_len > num_ques:
                    copy_len = num_ques
                problems = np.zeros(opt.max_len, dtype=np.int64)
                correct = np.ones(opt.max_len, dtype=np.int64)

                #JEREMY - adding time
                time_array = np.zeros(opt.max_len, dtype=np.float32)

                problems[-copy_len:] = ques[-copy_len:]
                correct[-copy_len:] = ans[-copy_len:]
                time_array[-copy_len:] = np.array(times[-copy_len:], dtype=np.float32)

                tup = (copy_len, problems, correct, time_array)
                #****************

                self.students.append(tup)
                
                if num_ques > opt.max_len - 1:
                    start_idx = opt.max_len - 1
                    while opt.max_len - 1 + start_idx <= num_ques:
                        problems = np.array(ques[(start_idx-1):(start_idx + opt.max_len -1 )])
                        correct = np.array(ans[(start_idx-1):(start_idx + opt.max_len -1)])
                        
                        #JEREMY TIME ADDED
                        time_array = np.array(times[(start_idx-1):(start_idx + opt.max_len -1)])
                        
                        tup = (opt.max_len, problems, correct, time_array)
                        #***

                        self.students.append(tup)
                        start_idx += (opt.max_len-1)
                    left_num_ques = num_ques - start_idx
                    
                    #HDKIM if left_num_ques>2: 
                    if left_num_ques>0:
                        problems = np.zeros(opt.max_len, dtype=np.int64)
                        correct = np.ones(opt.max_len, dtype=np.int64)

                        #JEREMY - adding time
                        time_array = np.zeros(opt.max_len, dtype=np.float32)
                        time_array[-left_num_ques:] = times[-left_num_ques:]
                        #****

                        problems[-left_num_ques:] = ques[-left_num_ques:]
                        correct[-left_num_ques:] = ans[-left_num_ques:]

                        #added time to tup
                        tup = (left_num_ques, problems, correct, time_array)

                        self.students.append(tup)
                        
        if train==False:
            if len(self.students) % opt.batch_size > 0:
                for i in range(opt.batch_size - (len(self.students) % opt.batch_size)):
                    self.students.append(tup)
                    
        print(len(self.students))


    def __getitem__(self, index):
        student = self.students[index]
        problems = student[1]
        #print("before",problems)
        correct = student[2]

        #Jeremy added to SAKT and replaced ''' code
        times = student[3]

        '''
        #HDKIM x = np.zeros(opt.max_len - 1)
        x = problems[:-1].copy()
        # we assume max_skill_num + 1 = num_skills because skill index starts from 0 to max_skill_num
        x += (correct[:-1] == 1) * (self.max_skill_num + 1)
        problems = problems[1:]
        correct = correct[1:]
        
        #print("after",problems)
        
        return x, problems, correct
        '''

        question_in = problems[:-1].copy()
        answer_in = correct[:-1].copy()
        time_in = times[:-1].copy()

        example_bins = np.array([150,250,350,500,700,900,1200,1600,2200], dtype=np.float32)
        timestamp_bucket_in = np.digitize(time_in, example_bins).astype(np.int64)

        question_next = problems[1:].copy()
        correct_target = correct[1:].copy()
        time_target = times[1:].copy()

        return question_in, answer_in, timestamp_bucket_in, question_next, correct_target, time_target
        #*********

    def __len__(self):
        return len(self.students)


    
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        
        self.device = torch.device(device) if isinstance(device, str) else device

        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
            
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

        #using CUDA
        if self.stream is not None:
          with torch.cuda.stream(self.stream):
              for k in range(len(self.batch)):
                  self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)
        #using CPU
        else:
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.device)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
      if self.stream is not None:
          torch.cuda.current_stream().wait_stream(self.stream)

      batch = self.batch
      self.preload()
      return batch
    
# student_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from config import DefaultConfig
#from utils import subsequent_mask
from torch.autograd import Variable
#from multihead_attn import MultiHeadedAttention
from torch.nn import LayerNorm

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, state_size, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, state_size)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, state_size, 2) *
                             -(math.log(10000.0) / state_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class student_model(nn.Module):

    def __init__(self, num_skills, state_size, num_heads=2, dropout=0.2, infer=False):
        super(student_model, self).__init__()
        self.infer = infer
        self.num_skills = num_skills
        self.state_size = state_size
        
        '''JEREMY COMMENTED OUT
        # we use the (num_skills * 2 + 1) as key padding_index
        self.embedding = nn.Embedding(num_embeddings=num_skills*2+1,
                                      embedding_dim=state_size)
                                      # padding_idx=num_skills*2
        '''
        # self.position_embedding = PositionalEncoding(state_size)
        self.position_embedding = nn.Embedding(num_embeddings=opt.max_len-1,
                                               embedding_dim=state_size)
        '''JEREMY COMMENTED OUT
        # we use the (num_skills + 1) as query padding_index
        self.problem_embedding = nn.Embedding(num_embeddings=num_skills+1,
                                      embedding_dim=state_size)
                                      # padding_idx=num_skills)
        '''

        self.multi_attn = MultiHeadedAttention(h=num_heads, d_model=state_size, dropout=dropout, infer=self.infer)
        self.feedforward1 = nn.Linear(in_features=state_size, out_features=state_size)
        self.feedforward2 = nn.Linear(in_features=state_size, out_features=state_size)
        
        '''JEREMY COMMENTED OUT
        self.pred_layer = nn.Linear(in_features=state_size, out_features=num_skills)
        '''

        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(state_size)

        #JEREMY IMPLEMENTING
        self.word_emb = nn.Embedding(num_skills+1, state_size) #padding is +1
        self.resp_emb = nn.Embedding(2, state_size) #how correct, 0 or 1
        self.time_emb = nn.Embedding(opt.timestamp_buckets, state_size)

        #2 error heads
        self.error_head = nn.Linear(state_size, 1) #correctness logits
        self.time_head = nn.Linear(state_size, 1) #log time prediction
        #****************

    #JEREMY - making forward predict 2 sequences (error and time)
    '''
    def forward(self, x, problems, target_index):
        # self.key_masks = torch.unsqueeze( (x!=self.num_skills*2).int(), -1)
        # self.problem_masks = torch.unsqueeze( (problems!=self.num_skills).int(), -1)
        x = self.embedding(x)
        pe = self.position_embedding(torch.arange(x.size(1)).unsqueeze(0).cuda())
        x += pe
        # x = self.position_embedding(x)
        problems = self.problem_embedding(problems)
        # self.key_masks = self.key_masks.type_as(x)
        # self.problem_masks = self.problem_masks.type_as(problems)
        # x *= self.key_masks
        # problems *= self.problem_masks
        x = self.dropout(x)
        res = self.multi_attn(query=self.layernorm(problems), key=x, value=x,
                              key_masks=None, query_masks=None, future_masks=None)
        outputs = F.relu(self.feedforward1(res))
        outputs = self.dropout(outputs)
        outputs = self.dropout(self.feedforward2(outputs))
        # Residual connection
        outputs += self.layernorm(res)
        outputs = self.layernorm(outputs)
        logits = self.pred_layer(outputs)
        
        #HDKIM logits = logits.contiguous().view(logits.size(0) * opt.max_len - 1, -1)
        logits = logits.contiguous().view(logits.size(0) * (opt.max_len - 1), -1)
        logits = logits.contiguous().view(-1)
        selected_logits = torch.gather(logits, 0, torch.LongTensor(target_index).cuda())
        return selected_logits
        '''
    def forward(self, ques_in, ans_in, timebucket, next_ques):
        input_x = self.word_emb(ques_in) + self.resp_emb(ans_in) + self.time_emb(timebucket)

        query = self.word_emb(next_ques)#getting next words

        #positional embedding implementation
        pe = self.position_embedding(torch.arange(input_x.size(1)).unsqueeze(0).to(input_x.device))
        input_x += pe
        query += pe
        
        input_x = self.dropout(input_x)
        res = self.multi_attn(query=self.layernorm(query), key=input_x, value=input_x,
                              key_masks=None, query_masks=None, future_masks=None)
        
        out = F.relu(self.feedforward1(res))
        out = self.dropout(out)
        out = self.dropout(self.feedforward2(out))
        out += self.layernorm(res)
        out = self.layernorm(out)

        error_logits = self.error_head(out).squeeze(-1)
        predicted_times = self.time_head(out).squeeze(-1)

        return error_logits, predicted_times
    #**************************************************

import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score

#JEREMY Changed the whole function (params and how it works)

def run_epoch(m, dataloader, optimizer, scheduler,
              criterion1, criterion2,
              pad_id, lambda_time,
              epoch_id=None, writer=None, is_training=True):

    epoch_start_time = time.time()
    m.to(device)
    if is_training:
        m.train()
    else:
        m.eval()

    actual_labels = []
    pred_labels = []

    #using regression in log space for normalizing times
    actual_log_times = []
    pred_log_times = []

    num_batch = len(dataloader)
    prefetcher = DataPrefetcher(dataloader, device=device)
    batch = prefetcher.next()
    counter = 0

    if is_training:
        while batch is not None:
            #new batches with added params
            q_in, resp_in, time_bucket_in, q_next, y_correct, y_time = batch

            q_in = q_in.long().to(device)
            resp_in = resp_in.long().to(device)
            time_bucket_in = time_bucket_in.long().to(device)
            q_next = q_next.long().to(device)

            y_correct = y_correct.float().to(device) #ensures either 0 or 1
            y_time = y_time.float().to(device)

            #ensures padding is ignored
            mask = (q_next != pad_id)
            if mask.sum().item() == 0:
                batch = prefetcher.next()
                continue

            #applies log to time targets
            y_log_time = torch.log1p(y_time)

            #forward pass
            err_logits, pred_log_time = m(q_in, resp_in, time_bucket_in, q_next)

            #get losses for criterions
            loss_err = criterion1(err_logits[mask], y_correct[mask])
            loss_time = criterion2(pred_log_time[mask], y_log_time[mask])

            loss = loss_err + lambda_time * loss_time

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), opt.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            #measures correctness of SAKT
            with torch.no_grad():
                pred_prob = torch.sigmoid(err_logits)
                actual_labels += list(y_correct[mask].detach().cpu().numpy())
                pred_labels += list(pred_prob[mask].detach().cpu().numpy())

                actual_log_times += list(y_log_time[mask].detach().cpu().numpy())
                pred_log_times += list(pred_log_time[mask].detach().cpu().numpy())

            counter += 1

            if counter % 500 == 0:
                print(f"\r batch{counter}/{num_batch}", end="")

            if counter >= num_batch:
                break

            batch = prefetcher.next()

    else:
        with torch.no_grad():
            while batch is not None:
                q_in, resp_in, time_bucket_in, q_next, y_correct, y_time = batch

                q_in = q_in.long().to(device)
                resp_in = resp_in.long().to(device)
                time_bucket_in = time_bucket_in.long().to(device)
                q_next = q_next.long().to(device)

                y_correct = y_correct.float().to(device)
                y_time = y_time.float().to(device)

                mask = (q_next != pad_id)
                if mask.sum().item() == 0:
                    batch = prefetcher.next()
                    continue

                y_log_time = torch.log1p(y_time)

                err_logits, pred_log_time = m(q_in, resp_in, time_bucket_in, q_next)
                pred_prob = torch.sigmoid(err_logits)

                actual_labels += list(y_correct[mask].cpu().numpy())
                pred_labels += list(pred_prob[mask].cpu().numpy())

                actual_log_times += list(y_log_time[mask].cpu().numpy())
                pred_log_times += list(pred_log_time[mask].cpu().numpy())

                counter += 1

                if counter % 500 == 0:
                    print(f"\r batch{counter}/{num_batch}", end="")

                if counter >= num_batch:
                    break

                batch = prefetcher.next()

    #getting the SAKT scores
    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))#root mean square error can be used for evaluation
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)#represents degree of seperation between true positive rate and false positive rate
    r2 = r2_score(actual_labels, pred_labels)#gets "goodness of fit" / coefficient of determination
    accuracy = metrics.accuracy_score(actual_labels, np.array(pred_labels) >= 0.5)

    #gets the root mean square error for log times
    time_rmse = sqrt(mean_squared_error(actual_log_times, pred_log_times)) if len(actual_log_times) else None

    return rmse, auc, r2, accuracy, pred_labels, time_rmse

'''
IF RUNNING ON GOOGLE COLAB:

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/"Colab Notebooks"
'''

import torch
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

#JEREMY changed whole function and how it works, and so it would work in our coding environment

# assumes these already exist from your notebook:
# opt, Data, DataLoaderX, student_model, run_epoch

def build_word_vocab(typing_log_df, min_freq=1):
    #words returned from typing logs

    counts = typing_log_df["word"].value_counts()
    words = counts[counts >= min_freq].index.tolist()
    word2id = {w: (i + 1) for i, w in enumerate(words)}  # start at 1
    return word2id

def make_sequences_df(typing_log_df, word2id, max_seq_len):
    #turns typing log into num_ques,ques,ans,times with one user per row

    rows = []
    for user_id, g in typing_log_df.groupby("user_id"):
        #figures out timestamp order (order times are inputted)
        if "timestamp" in g.columns:
            g = g.sort_values("timestamp")

        ques = [word2id.get(w, 0) for w in g["word"].astype(str).tolist()]
        ans  = [1 if m == 0 else 0 for m in g["mistypes"].astype(int).tolist()]
        times = g["time_ms"].astype(float).tolist()

        #using chunks
        for start in range(0, len(ques), max_seq_len):
            q_chunk = ques[start:start+max_seq_len]
            a_chunk = ans[start:start+max_seq_len]
            t_chunk = times[start:start+max_seq_len]
            if len(q_chunk) < 2:
                continue  #needs 2 steps to properly predict

            rows.append({
                "num_ques": len(q_chunk),
                "ques": q_chunk,
                "ans": a_chunk,
                "times": t_chunk
            })

    return pd.DataFrame(rows)

def split_train_valid(seqs_df, valid_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(seqs_df))
    rng.shuffle(idx)

    if len(idx) < 2:
        return seqs_df.copy(), seqs_df.iloc[:0].copy() #if there's not enough data to split

    n_valid = max(1, int(len(idx) * valid_frac))
    n_valid = min(n_valid, len(idx) - 1)
    #above ensures minimum of 1 train sample

    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    train_df = seqs_df.iloc[train_idx].reset_index(drop=True)
    valid_df = seqs_df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df

def save_sakt_bundle(save_path, model, opt, word2id, time_bins_ms):
    bundle = {
        "state_dict": model.state_dict(),
        "opt": {k: getattr(opt, k) for k in dir(opt)
                if not k.startswith("__") and not callable(getattr(opt, k))},
        "word2id": word2id,
        "id2word": {str(v): k for k, v in word2id.items()},
        "time_bins_ms": list(map(float, time_bins_ms)),
    }
    torch.save(bundle, save_path)

if __name__ == "__main__":

    #load typing log
    typing_log = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/typing_log.csv")

    #build vocab
    word2id = build_word_vocab(typing_log, min_freq=1)

    #for SAKT how may skills (in this case words)
    opt.num_skills = max(word2id.values()) + 1

    seqs_df = make_sequences_df(typing_log, word2id, 30) #max seq will always be 100 for now (not too long not too short) #changed to 30 due to less typing logs
    train_df, valid_df = split_train_valid(seqs_df, valid_frac=0.2, seed=0)

    time_bins_ms = np.array(np.linspace(800, 18000, 10), dtype=np.float32)

    #train, validate, and export model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = Data(train_df, train=True)
    valid_dataset = Data(valid_df, train=False)

    train_loader = DataLoaderX(train_dataset, batch_size=opt.batch_size, num_workers=2,
                               pin_memory=True, shuffle=True)
    valid_loader = DataLoaderX(valid_dataset, batch_size=opt.batch_size, num_workers=2,
                               pin_memory=True, shuffle=False)

    model = student_model(
        num_skills=opt.num_skills,
        state_size=opt.state_size,
        num_heads=opt.num_heads,
        dropout=opt.dropout,
        infer=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=opt.lr_decay)

    criterion_err = nn.BCEWithLogitsLoss()
    criterion_time = nn.SmoothL1Loss()

    #trained SAKT model we can use:
    os.makedirs("./artifacts", exist_ok=True)
    save_path = "./artifacts/sakt_typing_bundle.pt"

    best_auc = -1.0

    for epoch in range(opt.max_epoch):
        print(f"\ncurrent epoch: {epoch+1} out of {opt.max_epoch}")

        #train model
        train_rmse, train_auc, train_r2, train_acc, _, train_time_rmse = run_epoch(
            model, train_loader,
            optimizer, scheduler,
            criterion_err, criterion_time,
            pad_id=opt.word_padding,
            lambda_time=opt.lambda_time,
            is_training=True
        )

        #checking if any valid samples
        if len(valid_df) == 0:
          print("skipping validation - no valid samples")
          continue

        #validate model
        with torch.no_grad():
            val_rmse, val_auc, val_r2, val_acc, _, val_time_rmse = run_epoch(
                model, valid_loader,
                optimizer=None, scheduler=None,
                criterion1=criterion_err, criterion2=criterion_time,
                pad_id=opt.word_padding,
                lambda_time=opt.lambda_time,
                is_training=False
            )

        print(f"valid AUC = {val_auc:.4f} ACC = {val_acc:.4f} time_rmse(log) = {val_time_rmse}")

        #save best model
        if val_auc > best_auc:
            best_auc = val_auc
            save_sakt_bundle(save_path, model, opt, word2id, time_bins_ms)
            print(f"best bundle saved in: {save_path}")

    print("SAKT training finished")