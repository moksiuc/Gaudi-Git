import numpy as np
import torch
import torch.nn as nn
import copy

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.embedding_dim = args.hidden_units
        self.nn_parameter = args.nn_parameter

        if self.nn_parameter:
            self.item_emb = nn.Parameter(torch.normal(0,1, size = (self.item_num+1, args.hidden_units)))
            self.pos_emb = nn.Parameter(torch.normal(0,1, size=(args.maxlen, args.hidden_units)))
        else:
            print(f"item_emb = torch.nn.Embedding({self.item_num+1}, {args.hidden_units})")
            self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
            self.item_emb.weight.data.normal_(0.0,1)
            print(f"pos_emb = torch.nn.Embedding({args.maxlen}, {args.hidden_units})")
            self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.args = args

        self.back_state = []

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def debug_embedding(self, fname, vname, v, emb_obj, out):
        print_values = False
        attr = fname + "_" + vname
        try:
            lens = getattr(self, attr)
        except AttributeError:
            setattr(self, attr, [])
            lens = getattr(self, attr)
            print_values = True

        if len(v) not in lens:
            lens.append(len(v))
            print(f"{fname} len({vname}) = {len(v)}")

        if (print_values):
            print(f"{v.shape = }")
            print(f"{emb_obj.weight.shape = }")
            print(f"{out.shape = }")

        pos = len(self.back_state)
        self.back_state.append({})
        state = self.back_state[pos]

        if emb_obj == self.item_emb:
            state["emb_obj_cpu"] = torch.nn.Embedding(self.item_num+1, self.args.hidden_units, padding_idx=0)
        elif emb_obj == self.pos_emb:
            state["emb_obj_cpu"] = torch.nn.Embedding(self.args.maxlen, self.args.hidden_units)

        state["emb_obj_cpu"].weight.data = emb_obj.weight.cpu().detach().requires_grad_(True)
        state["input"] = torch.LongTensor(v)
        state["ref"] = state["emb_obj_cpu"](state["input"])

        cmp = np.allclose(
                out.detach().cpu().numpy(),
                state["ref"].detach().numpy(),
                atol=0,
                rtol=0,
                equal_nan=True,
            )

        if not cmp:
            print(f"{cmp = }")

        def hook_out(pos, grad):
            #print(f"{pos}: out.{grad.shape = }")
            self.back_state[pos]["grad_out"] = grad.cpu().detach()

        def hook_weight(pos, grad):
            #print(f"{pos}: weight.{grad.shape = }")
            self.back_state[pos]["grad_weight"] = grad.cpu().detach()

        state["hook_weight"] = lambda grad, pos=pos: hook_weight(pos, grad)
        state["hook_out"] = lambda grad, pos=pos: hook_out(pos, grad)

        emb_obj.weight.register_hook(state["hook_weight"])
        out.register_hook(state["hook_out"])


    def log2feats(self, log_seqs):
        if self.nn_parameter:
            seqs = self.item_emb[torch.LongTensor(log_seqs).to(self.dev)]
            seqs *= self.embedding_dim **0.5
        else:
            seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
            self.debug_embedding("log2feats", "log_seqs", log_seqs, self.item_emb, seqs)
            seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        #nn.Embedding
        if self.nn_parameter:
            seqs += self.pos_emb[torch.LongTensor(positions).to(self.dev)]
        else:
            add = self.pos_emb(torch.LongTensor(positions).to(self.dev))
            self.debug_embedding("log2feats", "positions", positions, self.pos_emb, add)
            seqs += add

        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        log_feats = self.log2feats(log_seqs)
        if mode == 'log_only':
            log_feats = log_feats[:, -1, :]
            return log_feats

        #nn.Embedding
        if self.nn_parameter:
            pos_embs = self.item_emb[torch.LongTensor(pos_seqs).to(self.dev)]
            neg_embs = self.item_emb[torch.LongTensor(neg_seqs).to(self.dev)]
        else:
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
            self.debug_embedding("forward", "pos_seqs", pos_seqs, self.item_emb, pos_embs)
            self.debug_embedding("forward", "neg_seqs", neg_seqs, self.item_emb, neg_embs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        if mode == 'item':
            return log_feats.reshape(-1, log_feats.shape[2]), pos_embs.reshape(-1, log_feats.shape[2]), neg_embs.reshape(-1, log_feats.shape[2])
        else:
            return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        #nn.Embedding
        if self.nn_parameter:
            item_embs = self.item_emb[torch.LongTensor(item_indices).to(self.dev)]
        else:
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
            self.debug_embedding("predict", "item_indices", item_indices, self.item_emb, item_embs)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
