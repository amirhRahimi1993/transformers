import os
from pathlib import Path
import torch
import re
import random
import transformers,datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset,DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import torch.nn as nn

MAX_LEN = 64

corpuse_movie_conv = 'datasets/movie_conversations.txt'
corpuse_movie_lines = 'datasets/movie_lines.txt'

with open(corpuse_movie_conv,'r',encoding='iso-8859-1') as c:
    conv = c.readlines()

with open(corpuse_movie_lines,'r',encoding='iso-8859-1') as l:
    lines = l.readlines()

lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

pairs = []
for con in conv:
    ids= eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []
        if i == len(ids) - 1:
            break
        first = lines_dic[ids[i]].strip()
        second= lines_dic[ids[i+1]].strip()

        qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
        qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)
print(pairs[20])

text_data = []
file_count = 0
for sample in tqdm.tqdm([x[0] for x in pairs]):
    text_data.append(sample)
    if len(text_data) == 10000:
        with open(f'data/text_{file_count}.txt','w',encoding='utf-8') as fp:
            fp.write("\n".join(text_data))
        text_data=[]
        file_count +=1

paths = [str(x) for x in Path('data').glob('**/*.txt')]

tokenizer = BertWordPieceTokenizer(
    clean_text= True,
    handle_chinese_chars=False,
    strip_accents = False,
    lowercase=True
)
tokenizer.train(
    files=paths,
    vocab_size= 30_000,
    min_frequency= 5,
    limit_alphabet= 1000,
    wordpieces_prefix='##',
    special_tokens= ['[PAD]','[CLS]','[SEP]','[MASK]','[UNK]']
)
tokenizer.save_model('bert-it-1','bert-it')
tokenizer= BertTokenizer.from_pretrained('bert-it-1/bert-it-vocab.txt', local_files_only=True)

class BERTDataset(Dataset):
    def __init__(self,data_pair,tokenizer,seq_length= 64):
        self.tokenizer= tokenizer
        self.seq_len = seq_length
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1,t2,is_next_label= self.get_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random,t2_label = self.random_word(t2)

        t1= [self.tokenizer.vocab["[CLS]"]] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        #segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t1))])[:self.seq_len]
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1+t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}
        return {key: torch.tensor(value) for key,value in output.items()}

    def random_word(self,sentence):
        tokens= sentence.split()
        output_label = []
        output = []

        for i, tokens in enumerate(tokens):
            prob = random.random()

            tokens_id= self.tokenizer(tokens)['input_ids'][1:-1]
            if prob < 0.15:
                prob/=0.15
                if prob < 0.8:
                    for i in range(len(tokens_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])
                elif prob < 0.9:
                    for i in range(len(tokens_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))
                else:
                    output.append(tokens_id)
                output_label.append(tokens_id)
            else:
                output.append(tokens_id)
                for i in range(len(tokens_id)):
                    output_label.append(0)
        output= list(itertools.chain(*[[x] if not isinstance(x,list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_sent(self,index):
        t1,t2= self.get_corpus_line(index)
        if random.random() > 0.5:
            return t1,t2,1
        else:
            return t1,self.get_random_line(),0
    def get_corpus_line(self,item):
        return self.lines[item][0],self.lines[item][1]
    def get_random_line(self):
        return self.lines[random.randrange(len(self.lines))][1]

class PositionaEmbedding(torch.nn.Module):
    def __init__(self,d_model,max_len=120):
        super().__init__()
        pe= torch.zeros(max_len,d_model).float()
        pe.required_grad= False

        for pos in range(max_len):
            for i in range(0,d_model,2):
                pe[pos,i] = math.sin(pos/(10000 ** ((2*i)/d_model)))
                pe[pos, i] = math.cos(pos / (10000 ** ((i+1) / d_model)))

        self.pe= pe.unsqueeze(0)

    def forward(self,x):
        return self.pe

class BERTEmbedding(torch.nn.Module):
    """
       BERT Embedding which is consisted with under features
           1. TokenEmbedding : normal embedding matrix
           2. PositionalEmbedding : adding positional information using sin, cos
           2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
           sum of all these features are output of BERTEmbedding
       """
    def __init__(self,vocab_size,embed_size,seq_len=64,dropout=0.1):
        super().__init__()
        self.embed_size =embed_size
        self.token = torch.nn.Embedding(vocab_size,embed_size,padding_idx=0)
        self.segment= torch.nn.Embedding(3,embed_size,padding_idx=0)
        self.position = PositionaEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout= torch.nn.Dropout(p=dropout)

    def forward(self,sequence,segment_label):
        x= self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)

class MultiHeadedAttention(torch.nn.Module):

    def __init__(self,heads,d_model,dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model% heads ==0
        self.d_k= d_model// heads
        self.heads= heads
        self.dropout= torch.nn.Dropout(dropout)

        self.query= torch.nn.Linear(d_model,d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)

        self.output_linear= torch.nn.Linear(d_model,d_model)

    def forward(self,query,key,value,mask):

        query = self.query(query)
        key = self.key(key)
        value = self.query(value)

        query = query.view(query.shape[0],-1,self.heads,self.d_k).permute(0,2,1,3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(query,key.permute(0,1,3,2))/ math.sqrt(query.size(-1))

        scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores,dim=-1)
        weights = self.dropout(weights)

        context= torch.matmul(weights,value)
        context= context.permute(0,2,1,3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    def __init__(self,d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model,middle_dim)
        self.fc2= nn.Linear(middle_dim,d_model)
        self.activation= nn.GELU()
        self.dropout= nn.Dropout(dropout)

    def forward(self,x):
        x= self.activation(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model=768,
        heads=12,
        feed_forward_hidden=768 * 4,
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded

class BERT(torch.nn.Module):

    def __init__(self,vocab_size,d_model=768,n_layers=12, heads=12,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        self.feed_forward_hidden = 4 * self.d_model
        self.embedding= BERTEmbedding(vocab_size= vocab_size,embed_size = d_model)

        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model,heads,d_model*4,dropout) for _ in range(self.n_layers)]
        )

    def forward(self,x,segment_info):
        mask = (x>0).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)

        x=self.embedding(x,segment_info)
        for encoder in self.encoder_blocks:
            x= encoder.forward(x,mask)
        return x

class NextSentencePrediction(torch.nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.linear= torch.nn.Linear(hidden,2)
        self.softmax= torch.nn.LogSoftmax(dim=-1)
    def forward(self,x):
        return self.softmax(self.linear(x[:,0]))

class MaskedLanguageModel(torch.nn.Module):
    def __init__(self,hidden,vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden,vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self,x):
        return self.softmax(self.linear(x))
class BERTLM(torch.nn.Module):

    def __init__(self,bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence= NextSentencePrediction(self.bert.d_model)
        self.mask_lm= MaskedLanguageModel(self.bert.d_model,vocab_size)

    def forward(self,x,segment_label):
        x= self.bert(x,segment_label)
        return self.next_sentence(x), self.mask_lm(x)

class ScheduledOtim():

    def __init__(self,optimizer,d_model,n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps= n_warmup_steps
        self.n_current_steps=0
        self.init_lr= np.power(d_model,-0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps,-0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ]
        )

    def _update_learning_rate(self):
        self.n_current_steps +=1
        lr= self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr']= lr

class BERTTrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 test_data_loader = None,
                 lr= 1e-4,
                 weight_decay= 0.01,
                 betas=(0.9,0.999),
                 warmup_steps= 10000,
                 log_freq=10,
                 device= 'cpu'
                 ):
        self.device = device
        self.model= model
        self.train_data = train_dataloader
        self.test_data = test_data_loader

        self.optim= Adam(self.model.parameters(),lr=lr,betas=betas,weight_decay=weight_decay)
        self.optim_schdule= ScheduledOtim(
            self.optim,self.model.bert.d_model,warmup_steps
        )

        self.criterion= torch.nn.NLLLoss(ignore_index=0)
        self.log_freq= log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self,epoch,data_loader,train=False):
        avg_loss= 0.0
        total_correct= 0
        total_element= 0
        mode= 'train' if train else 'test'
        data_iter= tqdm.tqdm(
            enumerate(data_loader),
            desc='EP_%s:%d:' %(mode,epoch),
            total= len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )
        for i, data in data_iter:
            data = {key: value.to(self.device) for key,value in data.items() }
            next_sent_output, mask_lm_output= self.model.forward(data["bert_input"], data["segment_label"])
            next_loss= self.criterion(next_sent_output,data["is_next"])
            mask_loss = self.criterion(mask_lm_output.transpose(1,2),data["bert_label"])
            loss= next_loss + mask_loss
            if train:
                self.optim_schdule.zero_grad()
                loss.backward()
                self.optim_schdule.step_and_update_lr()
            correct= next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix= {
                "epoch":epoch,
                "iter":i,
                "avg_loss":avg_loss/(i+1),
                "avg_acc": total_correct/total_element * 100,
                "loss": loss.item()
            }
            if i%self.log_freq ==0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch},{mode}: "
            f"avg_loss={avg_loss/len(data_iter)},"
            f"total_acc={total_correct * 100.0/ total_element}"
        )


    def train(self,epoch):
        self.iteration(epoch,self.train_data)

    def test(self,epoch):
        self.iteration(epoch,self.test_data,train=False)

'''test run'''

train_data = BERTDataset(
   pairs, seq_length=MAX_LEN, tokenizer=tokenizer)

train_loader = DataLoader(
   train_data, batch_size=32, shuffle=True, pin_memory=True)

bert_model = BERT(
  vocab_size=len(tokenizer.vocab),
  d_model=768,
  n_layers=2,
  heads=12,
  dropout=0.1
)

bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
bert_trainer = BERTTrainer(bert_lm, train_loader, device='cpu')
epochs = 20

for epoch in range(epochs):
  bert_trainer.train(epoch)