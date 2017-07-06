import torch
import sys
from torch import nn, optim
import torch.nn.functional as F
from packages.functions import num_to_var
import numpy as np
from collections import Counter
from torch.autograd import Variable
import time

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        hidden_size = args.hidden_size
        vocab_size = args.vocab_size
        embed_size = args.embed_size


        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.max_enc = args.max_enc
        self.max_oovs = args.max_oovs

        self.embed = nn.Embedding(vocab_size, embed_size)
        # words 0,1,2,3 stand for zeropad, SOS, EOS, UNK
        self.encoder = nn.LSTM(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=True)
        self.decoder = nn.LSTM(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True)

        # linear weights
        self.W_init = nn.Linear(hidden_size*2, hidden_size) # for using encoder's hidden state as decoder input

        self.wh = nn.Linear(hidden_size*2, 1) # for changing context vector into a scalar
        self.ws = nn.Linear(hidden_size, 1) # for changing hidden state into a scalar
        self.wx = nn.Linear(embed_size, 1) # for changing input embedding into a scalar

        self.Wh = nn.Linear(hidden_size*2, hidden_size) # for obtaining e from encoder hidden states
        self.Ws = nn.Linear(hidden_size, hidden_size) # for obtaining e from current state
        self.Wc = nn.Linear(self.max_enc, hidden_size) # for obtaining e from context vector
        self.v = nn.Linear(hidden_size, 1) # for changing to scalar

        self.V1 = nn.Linear(hidden_size*3, hidden_size*3)
        self.V2 = nn.Linear(hidden_size*3, vocab_size)

    def forward(self, input, target, batch, vocab, train=True):
        # input: input sequence (numpy array)
        # target: target sequence (numpy array)
        # batch: a Batch object (used for <UNK> and other stuff)
        # vocab: a Vocab object (used for <UNK> and other stuff)
        # train: whether train or test mode (Bool)

        # 0. get hyperparameters
        b = input.shape[0] # batch size
        in_seq = input.shape[1] # max input sequence length
        tar_seq = target.shape[1] # max target sequence length
        h = self.hidden_size # hidden size
        e = self.embed_size # embedding size

        # 1. obtain encoder output
        unked_input = batch.unk_minibatch(input, vocab)
        encoder_input = num_to_var(unked_input)
        if train: # optional: we need a decoder input for training
            unked_output = batch.unk_minibatch(target, vocab)
            decoder_input = num_to_var(unked_output)

        encoder_out, _ = self.encoder(self.embed(encoder_input)) # encoder_out: [b x in_seq x hid*2]

        # 2. obtain initial hidden state value
        # c0 = Variable(torch.Tensor(b,h).zero_()) # [b x hid]
        # h0 = self.W_init(encoder_out[:,0,:].squeeze()) # [b x hid]
        # C = (h0.unsqueeze(0), c0.unsqueeze(0)) # ([1 x b x hid], [1 x b x hid])

        coverage = Variable(torch.Tensor(b,in_seq).zero_()) # coverage vector [b x in_seq]
        coverage = self.to_cuda(coverage)
        cov_loss = 0
        start = time.time()
        if train:
            next_input = decoder_input[:,0] # which is already <SOS>
        else:
            ones = np.ones([b]) * vocab.w2i['<SOS>']
            next_input = num_to_var(ones)
        out_list = [] # list to concatenate all outputs later

        # 3. for each item in target
        for i in range(tar_seq-1):
            print(i)
            # 3.1. get embedding vectors for the decoder inputs
            embedded = self.embed(next_input) # [b x emb], next_input: Variable
            elapsed = time.time()
            diff = elapsed - start
            print("3-1: ",diff)
            time.sleep(5)
            start = time.time()
            # 3.2. get hidden state from previous hidden state and current decoder input
            if i==0:
                state, C = self.decoder(embedded.unsqueeze(1))
            else:
                state, C = self.decoder(embedded.unsqueeze(1), C)
            # state: [b x 1 x hid]
            
            # 3.3. get current attention distribution from encoder output(1), hidden state(3.2), coverage(3.7)
            # attn: [b x in_seq]
            # if self.max_enc>in_seq:
            # 	att1 = self.Wh(encoder_out.contiguous().view(-1,self.hidden_size*2)) + self.Ws(state.squeeze()).repeat(in_seq,1) + self.Wc(torch.cat([coverage,Variable(torch.Tensor(b,self.max_enc-in_seq).zero_())],1)).repeat(in_seq,1)
            # else:
            attn1 = self.Wh(encoder_out.contiguous().view(-1,encoder_out.size(2))) + self.Ws(state.squeeze()).repeat(in_seq,1) + self.Wc(coverage).repeat(in_seq,1)
            # attn1: [b*in_seq x hidden]
            attn2 = self.v(attn1) # [b*in_seq x 1]
            attn = F.softmax(attn2.view(b,in_seq)) # [b x in_seq]

            elapsed = time.time()
            diff = elapsed - start
            print("3-2,3: ",diff)
            time.sleep(5)
            start = time.time()

            # 3.4. get context vector from encoder_out and attention
            context = torch.bmm(attn.unsqueeze(1),encoder_out) # [b x 1 x in_seq] * [b x in_seq x hidden*2]
            context = context.squeeze() # [b x hidden*2]

            # 3.5. get p_gen using the context vector(3.4), the hidden state(3.2), encoder input(3.1) 
            p_gen = F.sigmoid(self.wh(context) + self.ws(state.squeeze()) + self.wx(embedded)) # [b]

            # 3.6. get coverage loss by comparing the attention with current coverage
            cov_loss += torch.sum(torch.min(attn,coverage))

            # 3.7. update coverage vector by adding attention(3.3)
            coverage += attn

            elapsed = time.time()
            diff = elapsed - start
            print("3-4,5,6,7: ",diff)
            time.sleep(5)
            start = time.time()

            # 3.8. get output vector by adding the two vectors
            # 3.8.1. get p_vocab from state (3.2) and context (3.4)
            p_vocab = F.softmax(self.V2(self.V1(torch.cat([state.squeeze(),context],1)))) # [b x vocab]
            oovs = Variable(torch.Tensor(b,self.max_oovs).zero_())+1.0/self.vocab_size
            oovs = self.to_cuda(oovs)
            p_vocab = torch.cat([p_vocab,oovs],1)



            # 3.8.2. get p_copy from attn (3.3) and input (np array)
            numbers = input.reshape(-1).tolist()
            set_numbers = list(set(numbers)) # all unique numbers
            set_numbers.remove(0)
            c = Counter(numbers)
            dup_list = [k for k in set_numbers if (c[k]>1)]
            masked_idx_sum = np.zeros([b,in_seq])
            
            dup_attn_sum = Variable(torch.FloatTensor(np.zeros([b,in_seq],dtype=float)))
            dup_attn_sum = self.to_cuda(dup_attn_sum)
            
            elapsed = time.time()
            diff = elapsed - start
            print("3-8.2: ",diff)
            time.sleep(5)
            start = time.time()
            # 3.8.3. add all duplicate attns to a distinct matrix
            for dup in dup_list:
                mask = np.array(input==dup, dtype=float)
                masked_idx_sum += mask
                attn_mask = torch.mul(Variable(torch.Tensor(mask)).cuda(),attn)
                attn_sum = attn_mask.sum(1).unsqueeze(1) # [b x 1]
                # print(attn_mask)
                # print(attn_sum)
                # print(dup_attn_sum)
                dup_attn_sum += torch.mul(attn_mask,attn_sum)
            masked_idx_sum = Variable(torch.Tensor(masked_idx_sum).cuda())
            
            elapsed = time.time()
            diff = elapsed - start
            print("3-8.3: ",diff)
            time.sleep(5)
            start = time.time()
            # 3.8.4. 
            attn = torch.mul(attn,(1-masked_idx_sum))+dup_attn_sum
            batch_indices = torch.arange(start=0, end=b).long()
            batch_indices = batch_indices.expand(in_seq, b).transpose(1, 0).contiguous().view(-1)
            idx_repeat = torch.arange(start=0, end=in_seq).repeat(b).long()
            p_copy = torch.zeros(b,self.vocab_size+self.max_oovs)
            p_copy = self.to_cuda(Variable(p_copy))
            word_indices = input.reshape(-1)
            p_copy[batch_indices,word_indices] += attn[batch_indices,idx_repeat]
            
            elapsed = time.time()
            diff = elapsed - start
            print("3-8.4: ",diff)
            time.sleep(5)
            start = time.time()
            
            # en = torch.LongTensor(input) # [b x in_seq]
            # en.unsqueeze_(2) # [b x in_seq x 1]
            # one_hot = torch.FloatTensor(en.size(0),en.size(1),p_vocab.size(1)).zero_() # [b x in_seq x vocab+oov]
            # one_hot.scatter_(2,en,1) # one hot tensor: [b x in_seq x vocab+oov]
            # one_hot = self.to_cuda(one_hot)
            # p_copy = torch.bmm(attn.unsqueeze(1),Variable(one_hot, requires_grad=False)) # [b x 1 x vocab+oov]
            # p_copy = p_copy.squeeze() # [b x vocab+oov]
            # p_gen = p_gen.repeat(1,p_vocab.size(1))
            # p_gen = p_gen.unsqueeze(1) # [b x 1]
            # print(p_gen.size())
            # print(p_vocab.size())
            # print(p_copy.size())
            # print((p_vocab*p_gen).size())
            p_out = torch.mul(p_vocab,p_gen) + torch.mul(p_copy,(1-p_gen)) # [b x extended_vocab]
            # extended_vocab : vocab + max_oov



            # 3.9. append to out_list
            out_list.append(p_out)
            elapsed = time.time()
            diff = elapsed - start
            print("3-9: ",diff)
            time.sleep(5)
            start = time.time()
            
            # 3.10. get next input
            if train:
                next_input = decoder_input[:,i]
            else:
                next_input = p_out.max(1)[1].squeeze() # if test, we take in the previous 
        # 4. concatenate all into a 3-dim tensor
        out_list = torch.stack(out_list,1) # [b x seq x ext_vocab]
        print(out_list.size())
        # 5. delete unnecessary tensor Variables (??)

        # 6. return answer
        return out_list, cov_loss

    def mask(self, matrix):
        # for obtaining a FloatTensor Variable mask out of a LongTensor Variable
        out = matrix==0
        return Variable(out.float().data)

    def to_cuda(self, tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=True)
    def forward(self, x, embed):
        embedded = embed(x) # embedded: [b x seq x emb]
        out, h = self.lstm(embedded) # out: [b x seq x hid*2] (biRNN)
        return out, h

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CopyDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size+hidden_size,
                            hidden_size=hidden_size, batch_first=True,
                            bidirectional=False)

    def forward(self, x, context, prev_state, embed):
        # context: [b x 1 x hidden*2]
        embedded = embed(x) # embedded: [b x 1 x emb] if x: [b x 1]
        input = torch.cat([context,embedded],dim=2)
        output,h = nn.lstm(input)
        return output, h


