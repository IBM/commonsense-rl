import torch
import torch.nn as nn
from models import PretrainedEmbeddings, GAT, CQAttention, SelfAttention, Attention
from utils.generic import masked_softmax


class CommandScorerWithKG(nn.Module):
    def __init__(self, word_emb, graph_emb, graph_type, hidden_size,device):
        super(CommandScorerWithKG, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.dropout_ratio = 0.0 # *
        self.n_heads = 1 # *
        self.use_hints = True # *
        self.bidirectional = True
        self.graph_type = graph_type
        n_factor = 2 # command
        bi_factor = (2 if self.bidirectional else 1) # hidden size multiplier when bidirectional is used

        self.word_embedding = PretrainedEmbeddings(word_emb)
        self.word_embedding_size = self.word_embedding.dim  # *
        self.word_embedding_prj = torch.nn.Linear(self.word_embedding_size, self.hidden_size, bias=False)
        if not self.bidirectional:
            self.word_hint_prj = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.graph_embedding = None
        if graph_emb is not None and ('local' in self.graph_type or 'world' in self.graph_type):
            self.graph_embedding = PretrainedEmbeddings(graph_emb, True)
            self.graph_embedding_size = self.graph_embedding.dim
            self.graph_embedding_prj = torch.nn.Linear(self.graph_embedding_size, self.hidden_size, bias=False)
            if not self.bidirectional:
                self.graph_hint_prj = torch.nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

        # Encoder for th observation
        self.encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional= self.bidirectional)
        # Encoder for the commands
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional= self.bidirectional)

        # RNN that keeps track of the encoded state over time
        self.state_gru = nn.GRU(hidden_size*bi_factor, hidden_size*bi_factor, batch_first=True)

        self.kg_word_encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.kg_graph_encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        if 'local' in self.graph_type or 'world' in graph_type:
            self.attention = CQAttention(block_hidden_dim=hidden_size*bi_factor, dropout=self.dropout_ratio)
            self.attention_prj = torch.nn.Linear(hidden_size * bi_factor * 4, hidden_size * bi_factor, bias=False)

        if 'world' in self.graph_type:
            n_factor += 1
            self.worldkg_gat = GAT(hidden_size, hidden_size, self.dropout_ratio, alpha=0.2, nheads=self.n_heads)
            self.worldkg_attention_prj = torch.nn.Linear(hidden_size*bi_factor * 4, hidden_size*bi_factor, bias=False)
            self.world_self_attention = SelfAttention(hidden_size*bi_factor, hidden_size*bi_factor, self.n_heads, self.dropout_ratio)
        if 'local' in graph_type:
            n_factor += 1
            self.localkg_gat = GAT(hidden_size, hidden_size, self.dropout_ratio, alpha=0.2, nheads=self.n_heads)
            self.localkg_attention_prj = torch.nn.Linear(hidden_size * bi_factor * 4, hidden_size * bi_factor, bias=False)
            self.local_self_attention = SelfAttention(hidden_size*bi_factor, hidden_size*bi_factor, self.n_heads, self.dropout_ratio)

        self.state_hidden = []
        self.general_attention = Attention(hidden_size * bi_factor*2,hidden_size * bi_factor) # General attention from [cmd + obs ==> graph_nodes]
        self.world_attention = None
        self.local_attention = None
        self.obs2kg_attention = torch.nn.Linear(hidden_size * bi_factor, hidden_size * bi_factor, bias=False)
        self.critic = nn.Linear(hidden_size*bi_factor, 1)

        self.att_cmd = nn.Sequential(nn.Linear(hidden_size * bi_factor * n_factor, hidden_size * bi_factor),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size * bi_factor,1))
        self.count = 1

    def forward(self, obs, commands, local_graph, local_hints, local_adj, world_graph, world_hints, world_adj, **kwargs):
        input_length = obs.size(1)
        batch_size = obs.size(0)
        nb_cmds = commands.size(1)
        cmd_selector_input = []

        # Observed State
        embedded = self.word_embedding(obs) # batch x word x emb_size
        embedded = self.word_embedding_prj(embedded) # batch x word x hidden
        encoder_output, encoder_hidden = self.encoder_gru(embedded) # encoder_hidden 1/2 x batch x hidden
        encoder_hidden = encoder_hidden.permute(1, 0, 2).reshape(encoder_hidden.shape[1], 1, -1) if \
                        encoder_hidden.shape[0] == 2 else encoder_hidden
        if self.state_hidden is None:
            self.state_hidden = torch.zeros_like(encoder_hidden)
        state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden)
        self.state_hidden = state_hidden.detach()

        value = self.critic(state_output)
        state_hidden = state_hidden.transpose(0, 1).contiguous().squeeze(1) # batch x hidden

        # Commands/Actions
        cmds_embedding = self.word_embedding(commands)
        cmds_embedding = self.word_embedding_prj(cmds_embedding)
        cmds_embedding = cmds_embedding.view(batch_size * nb_cmds, commands.size(2),
                                             self.hidden_size)  # [batch-ncmds] x nentities x hidden_size
        _, cmds_encoding = self.cmd_encoder_gru.forward(cmds_embedding)  # 1/2 x [batch-ncmds] x hidden
        cmds_encoding = cmds_encoding.permute(1, 0, 2).reshape(1, cmds_encoding.shape[1], -1) if \
            cmds_encoding.shape[0] == 2 else cmds_encoding
        cmds_encoding = cmds_encoding.squeeze(0)
        cmds_encoding = cmds_encoding.view(batch_size, nb_cmds, self.hidden_size * (2 if self.bidirectional else 1))
        cmd_selector_input.append(cmds_encoding)  # batch x cmds x hidden

        query_encoding = torch.cat(
            [cmds_encoding, torch.stack([state_hidden] * nb_cmds, dim=1)], dim=-1)  # batch x cmds x hidden*2

        if torch.any(torch.isnan(encoder_hidden)):
            print("error")

        # Local Graph
        localkg_encoding = torch.FloatTensor()
        worldkg_encoding = torch.FloatTensor()
        if 'local' in self.graph_type and local_graph.nelement() > 0:
            # graph # num_nodes x entities
            localkg_embedded = self.word_embedding(local_graph)  # nodes x entities x hidden+
            localkg_embedded = self.word_embedding_prj(localkg_embedded) #  nodes x  entities x hidden
            localkg_embedded = localkg_embedded.mean(1) # nodes x hidden
            localkg_embedded = torch.stack([localkg_embedded]*batch_size,0) # batch x nodes x hidden
            localkg_encoding = self.localkg_gat(localkg_embedded, local_adj.float())

            if self.use_hints:
                # Get hint with word_embedding ids tensor
                hints_embedded = self.word_embedding(local_hints)
                hints_embedded = self.word_embedding_prj(hints_embedded)
                _, hint_encoding = self.kg_word_encoder_gru(hints_embedded)
                hint_encoding = hint_encoding.squeeze(0)

                localkg_encoding = torch.cat(
                    [localkg_encoding, torch.stack([hint_encoding.squeeze(1)] * local_graph.shape[0], dim=1)], dim=-1)
                if not self.bidirectional:
                    localkg_encoding = self.word_hint_prj(localkg_encoding)

        # World Graph
        if 'world' in self.graph_type and self.graph_embedding and world_graph.nelement() > 0:
            # graph # num_nodes x entities
            worldkg_embedded = self.graph_embedding(world_graph)  # nodes x entities x hidden+
            worldkg_embedded = self.graph_embedding_prj(worldkg_embedded) #  nodes x  entities x hidden
            worldkg_embedded = worldkg_embedded.mean(1) # nodes x hidden
            worldkg_embedded = torch.stack([worldkg_embedded]*batch_size,0) # batch x nodes x hidden
            worldkg_encoding = self.worldkg_gat(worldkg_embedded, world_adj.float())

            if self.use_hints:
                # Get hint with graph_embedding ids tensor
                hints_embedded = self.graph_embedding(world_hints)
                hints_embedded = self.graph_embedding_prj(hints_embedded)
                _, hint_encoding = self.kg_graph_encoder_gru(hints_embedded)
                hint_encoding = hint_encoding.squeeze(0)

                worldkg_encoding = torch.cat(
                    [worldkg_encoding, torch.stack([hint_encoding.squeeze(1)] * world_graph.shape[0], dim=1)], dim=-1)
                if not self.bidirectional:
                    worldkg_encoding = self.graph_hint_prj(worldkg_encoding)

        if 'local' in self.graph_type and localkg_encoding.nelement() > 0:  # graphtype = local
                mask = torch.ones((batch_size,1), device=self.device, requires_grad=False).byte()
                state_hidden = state_hidden.unsqueeze(1) # batch x 1 x hidden
                obs_encoding = self.attention(state_hidden, localkg_encoding, mask, local_adj.sum(dim=2) > 0)
                obs_encoding = self.attention_prj(obs_encoding)
                localkg_encoding = self.attention(localkg_encoding, state_hidden, local_adj.sum(dim=2) > 0, mask)
                localkg_encoding = self.localkg_attention_prj(localkg_encoding)
                state_hidden = obs_encoding.squeeze(1) # batch x hidden

                local_nodes = local_adj.sum(dim=2)
                m1 = local_nodes.unsqueeze(-1)
                m2 = local_nodes.unsqueeze(1)
                mask_squared = torch.bmm(m1, m2).byte()
                local2obs_encoding, _ = self.local_self_attention(
                    localkg_encoding, mask_squared, localkg_encoding, localkg_encoding)

                localkg_representation, local_attention = self.general_attention(query_encoding, local2obs_encoding)
                self.local_attention = local_attention.clone().detach()
                localkg_representation = localkg_representation.squeeze(1)

                cmd_selector_input.append(localkg_representation)

        elif 'world' in self.graph_type and worldkg_encoding.nelement() > 0: # graphtype = world
            mask = torch.ones((batch_size, 1), device=self.device, requires_grad=False).byte()
            state_hidden = state_hidden.unsqueeze(1)
            obs_encoding = self.attention(state_hidden, worldkg_encoding,mask,world_adj.sum(dim=2)>0)
            obs_encoding = self.attention_prj(obs_encoding)
            worldkg_encoding = self.attention(worldkg_encoding, state_hidden,world_adj.sum(dim=2)>0, mask)
            worldkg_encoding = self.worldkg_attention_prj(worldkg_encoding)
            state_hidden = obs_encoding.squeeze(1) # batch x hidden

            world_nodes = world_adj.sum(dim=2)  # batch x nworld
            m1 = world_nodes.unsqueeze(-1)
            m2 = world_nodes.unsqueeze(1)
            mask_squared = torch.bmm(m1, m2).byte()
            world2obs_encoding, _ = self.world_self_attention(
                worldkg_encoding, mask_squared, worldkg_encoding, worldkg_encoding)

            worldkg_representation, world_attention = self.general_attention(query_encoding, world2obs_encoding)
            self.world_attention = world_attention.clone().detach()

            cmd_selector_input.append(worldkg_representation)

        self.count += 1

        # Concatenate the observed state (required) and command (required) and scored command history (optional) encodings
        # with kg-based encodings for commnads (optional) and scored command history (optional).
        # State rpresentaton for all types of agents
        cmd_selector_input.append(torch.stack([state_hidden] * nb_cmds, 1))  # batch x cmds x hidden
        cmd_selector_new_input = torch.cat(cmd_selector_input, dim=-1) # batch x ncmds x [hidden*nfactor]

        # Compute one score per command.
        scores = self.att_cmd(cmd_selector_new_input).squeeze(-1)  # batch x ncmds
        probs = masked_softmax(scores, commands.sum(dim=2) > 0, dim=1)  # batch x cmds

        index = probs.multinomial(num_samples=1).unsqueeze(0)  # batch x indx
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size * (2 if self.bidirectional else 1), device=self.device)

    def reset_hidden_per_batch(self, batch_id):
        self.state_hidden[:,batch_id,:] = torch.zeros(1, 1, self.hidden_size * (2 if self.bidirectional else 1), device=self.device)
