"""Offline *actual checkpoint* inference for Q-Guard common-owner link detection.

Reads only raw/ and manifest.json; NEVER reads private/.
Uses the same feature definitions / architecture as the supplied main.ipynb.
"""
from __future__ import annotations
import argparse
import json
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected

MAX_SEQ_LEN = 30
HIDDEN_DIM = 32
N_QUBITS = 4
FEATURES = ['log_amount','is_deposit','is_wager','is_withdrawal','is_payout',
            'is_approved','log_gap_hours','hour_sin','hour_cos','is_weekend']

class TemporalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=10, hidden_size=32, batch_first=True)
        self.norm = nn.LayerNorm(32)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.clamp_min(1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.gru(packed)
        h = self.norm(hidden[-1])
        return h.masked_fill((lengths == 0).to(h.device).unsqueeze(1), 0.0)

class HeteroGraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_resources = HeteroConv({
            ('account','uses_device','device'): SAGEConv((-1,-1), 32),
            ('account','connects_from','network'): SAGEConv((-1,-1), 32),
            ('account','uses_payment','payment'): SAGEConv((-1,-1), 32),
        }, aggr='sum')
        self.to_accounts = HeteroConv({
            ('device','rev_uses_device','account'): SAGEConv((-1,-1), 32),
            ('network','rev_connects_from','account'): SAGEConv((-1,-1), 32),
            ('payment','rev_uses_payment','account'): SAGEConv((-1,-1), 32),
        }, aggr='sum')
        self.skip = nn.Linear(5,32)
        self.norm = nn.LayerNorm(32)

    def forward(self, graph):
        resources = self.to_resources(graph.x_dict,graph.edge_index_dict)
        resource_x = {'account': graph['account'].x,
                      'device': F.relu(resources['device']),
                      'network': F.relu(resources['network']),
                      'payment': F.relu(resources['payment'])}
        account_h = self.to_accounts(resource_x,graph.edge_index_dict)['account']
        return self.norm(F.relu(account_h+self.skip(graph['account'].x)))

class AccountPairDecoder(nn.Module):
    def __init__(self, dropout_after_activation: bool):
        super().__init__()
        layers = [nn.Linear(64,32),nn.GELU()]
        if dropout_after_activation:
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(32,1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings, pairs):
        a=embeddings[pairs[:,0]]
        b=embeddings[pairs[:,1]]
        feat=torch.cat([torch.abs(a-b),a*b],dim=-1)
        return self.classifier(feat).squeeze(-1)

class QuantumFusion(nn.Module):
    def __init__(self, layers, n_qubits=4):
        super().__init__()
        self.temporal_projection = nn.Linear(32,2)
        self.graph_projection = nn.Linear(32,2)
        qdev = qml.device('default.qubit', wires=n_qubits)
        @qml.qnode(qdev,interface='torch',diff_method='backprop')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs,wires=range(n_qubits),rotation='Y')
            qml.StronglyEntanglingLayers(weights,wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.quantum=qml.qnn.TorchLayer(circuit,{'weights':(layers,n_qubits,3)})
        self.fusion=nn.Sequential(nn.Linear(64+n_qubits,32),nn.GELU(),nn.LayerNorm(32))

    def forward(self, temporal_h, graph_h):
        t=self.temporal_projection(temporal_h)
        g=self.graph_projection(graph_h)
        angles=torch.pi*torch.tanh(torch.cat([t,g],dim=-1))
        q_h=self.quantum(angles)
        return self.fusion(torch.cat([temporal_h,graph_h,q_h.to(temporal_h.dtype)],dim=-1))

class QGuardQuantumLink(nn.Module):
    def __init__(self, n_layers, decoder_dropout):
        super().__init__()
        self.fusion=QuantumFusion(n_layers)
        self.decoder=AccountPairDecoder(decoder_dropout)

    def forward(self, ht,hg,pairs):
        unique,inverse=torch.unique(pairs.reshape(-1),sorted=True,return_inverse=True)
        fused=self.fusion(ht[unique],hg[unique])
        return self.decoder(fused,inverse.reshape(-1,2))

def read_public(root: Path):
    names=['accounts','transactions','sessions','payment_events','networks']
    public={k:pd.read_parquet(root/'raw'/f'{k}.parquet') for k in names}
    for k in ['transactions','sessions','payment_events']:
        public[k]['timestamp']=pd.to_datetime(public[k]['timestamp'],utc=True)
    manifest=json.loads((root/'manifest.json').read_text())
    start=pd.Timestamp(manifest['sim_start_utc'])
    snapshot=start+pd.Timedelta(days=90)
    for k in ['transactions','sessions','payment_events']:
        public[k]=public[k].loc[public[k]['timestamp']<snapshot].copy()
    return public,snapshot

def sequences(public,account_map):
    tx=public['transactions'].copy()
    tx=tx.sort_values(['account_id','timestamp','transaction_id'])
    tx['log_amount']=np.log1p(tx['amount_cad'].clip(lower=0))
    for kind in ['deposit','wager','withdrawal','payout']:
        tx[f'is_{kind}']=(tx['transaction_type']==kind).astype(np.float32)
    tx['is_approved']=(tx['status']=='approved').astype(np.float32)
    gaps=(tx.groupby('account_id')['timestamp'].diff().dt.total_seconds()
          .div(3600).fillna(0).clip(lower=0,upper=720))
    tx['log_gap_hours']=np.log1p(gaps)
    hr=tx.timestamp.dt.hour+tx.timestamp.dt.minute/60
    tx['hour_sin']=np.sin(2*np.pi*hr/24)
    tx['hour_cos']=np.cos(2*np.pi*hr/24)
    tx['is_weekend']=(tx.timestamp.dt.dayofweek>=5).astype(np.float32)
    x=np.zeros((len(account_map),MAX_SEQ_LEN,len(FEATURES)),dtype=np.float32)
    lengths=np.zeros(len(account_map),dtype=np.int64)
    for account_id,group in tx.groupby('account_id',sort=False):
        if account_id not in account_map: continue
        vals=group[FEATURES].tail(MAX_SEQ_LEN).to_numpy(dtype=np.float32)
        i=account_map[account_id]
        x[i,:len(vals)]=vals
        lengths[i]=len(vals)
    return torch.from_numpy(x),torch.from_numpy(lengths)

def graph(public,account_map):
    s=public['sessions']
    p=public['payment_events']
    network_types=public['networks'].set_index('network_id')['network_type'].to_dict()
    sn=s[s['network_id'].map(network_types)!='public']
    node_ids={'account':list(account_map),'device':sorted(s['device_id'].dropna().unique()),
              'network':sorted(sn['network_id'].dropna().unique()),
              'payment':sorted(p['payment_token'].dropna().unique())}
    node_maps={k:{v:i for i,v in enumerate(values)} for k,values in node_ids.items()}
    def acct_counts(df,column=None):
        grouped=df.groupby('account_id')
        ser=grouped.size() if column is None else grouped[column].nunique()
        return ser.reindex(node_ids['account'],fill_value=0).to_numpy(dtype=np.float32)
    ac=np.column_stack([acct_counts(s),acct_counts(s,'device_id'),acct_counts(s,'network_id'),
                        acct_counts(p),acct_counts(p,'payment_token')])
    g=HeteroData()
    g['account'].x=torch.tensor(np.log1p(ac),dtype=torch.float32)
    def resource_features(df,column,ids):
        grp=df.groupby(column)
        event_count=grp.size().reindex(ids,fill_value=0).to_numpy(dtype=np.float32)
        account_count=grp['account_id'].nunique().reindex(ids,fill_value=0).to_numpy(dtype=np.float32)
        return torch.tensor(np.column_stack([np.ones(len(ids),dtype=np.float32),
                            np.log1p(event_count),np.log1p(account_count)]),dtype=torch.float32)
    for name,frame,col in [('device',s,'device_id'),('network',sn,'network_id'),
                           ('payment',p,'payment_token')]:
        g[name].x=resource_features(frame,col,node_ids[name])
    for frame,col,node,rel in [(s,'device_id','device','uses_device'),
                                (sn,'network_id','network','connects_from'),
                                (p,'payment_token','payment','uses_payment')]:
        pairs=frame[['account_id',col]].dropna().drop_duplicates()
        src=pairs['account_id'].map(account_map).to_numpy(dtype=np.int64)
        dst=pairs[col].map(node_maps[node]).to_numpy(dtype=np.int64)
        g['account',rel,node].edge_index=torch.tensor(np.stack([src,dst]),dtype=torch.long)
    return ToUndirected()(g),sn

def candidate_evidence(public,account_map,sn,limit_entity_degree=20):
    lookup=defaultdict(lambda: {'shared_devices':set(),'shared_networks':set(),'shared_payments':set()})
    for frame,col,label in [(public['sessions'],'device_id','shared_devices'),
                            (sn,'network_id','shared_networks'),
                            (public['payment_events'],'payment_token','shared_payments')]:
        rel=frame[['account_id',col]].dropna().drop_duplicates()
        for identifier,group in rel.groupby(col):
            indexes=sorted({account_map[a] for a in group['account_id'] if a in account_map})
            if len(indexes)>limit_entity_degree: continue
            for pair in combinations(indexes,2):
                lookup[pair][label].add(str(identifier))
    return lookup

def find_encoder(checkpoint_dir):
    """Find actual pretrained encoders, rejecting classical fusion-only weights."""
    candidates = ['encoder_best.pt', 'classical_encoder_best.pt', 'classical_best.pt']
    for name in candidates:
        path = checkpoint_dir / name
        if not path.is_file():
            continue
        state = torch.load(path, map_location='cpu', weights_only=True)
        if any(k.startswith('temporal.') for k in state) and any(k.startswith('graph_encoder.') for k in state):
            return path, state
    raise FileNotFoundError(
        'Pretrained GRU+GraphSAGE checkpoint not found. The supplied classical_best.pt '
        'contains only the classical fusion/decoder, not the encoders. Export the '
        'trained classical_model.state_dict() as checkpoints/encoder_best.pt '
        '(see README.md) before running model inference.'
    )


def load_models(checkpoint_dir):
    _, encoder_state = find_encoder(checkpoint_dir)
    quantum_state = torch.load(checkpoint_dir / 'Quantum_best.pt', map_location='cpu', weights_only=True)
    layers=int(quantum_state['fusion.quantum.weights'].shape[0])
    if int(quantum_state['fusion.quantum.weights'].shape[1])!=4:
        raise ValueError('Expected four-qubit checkpoint; review architecture before inference.')
    decoder_dropout='decoder.classifier.3.weight' in quantum_state
    temporal=TemporalGRU()
    temporal.load_state_dict({k.removeprefix('temporal.'):v for k,v in encoder_state.items()
                              if k.startswith('temporal.')})
    graph_model=HeteroGraphSAGE()
    # Lazy SAGEConv params materialize on initial forward, so load after first graph pass.
    quantum=QGuardQuantumLink(layers,decoder_dropout)
    quantum.load_state_dict(quantum_state,strict=True)
    temporal.eval()
    quantum.eval()
    return temporal,graph_model,quantum,encoder_state

def run(root,checkpoint_dir,out,max_candidates=None,batch_size=64):
    root=Path(root)
    checkpoint_dir=Path(checkpoint_dir)
    out=Path(out)
    if not (checkpoint_dir/'Quantum_best.pt').is_file():
        raise FileNotFoundError(f'Missing quantum checkpoint: {checkpoint_dir/"Quantum_best.pt"}')
    temporal,graph_model,quantum,encoder_state=load_models(checkpoint_dir)
    public,snapshot=read_public(root)
    ids=sorted(public['accounts']['account_id'].dropna().unique())
    account_map={a:i for i,a in enumerate(ids)}
    X,L=sequences(public,account_map)
    g,sn=graph(public,account_map)
    candidate_map=candidate_evidence(public,account_map,sn)
    ranked=sorted(candidate_map, key=lambda p:(-sum(bool(v) for v in candidate_map[p].values()),p))
    if max_candidates is not None:
        ranked=ranked[:max_candidates]
    graph_model.eval()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='There exist node types')
        with torch.no_grad():
            # Materialize lazy hetero convolutions before loading checkpoint tensors.
            graph_model(g)
    graph_model.load_state_dict({k.removeprefix('graph_encoder.'):v for k,v in encoder_state.items()
                                 if k.startswith('graph_encoder.')},strict=True)
    with torch.inference_mode():
        ht=torch.cat([temporal(X[i:i+1024],L[i:i+1024]) for i in range(0,len(X),1024)])
        hg=graph_model(g)
        outputs=[]
        for start in range(0,len(ranked),batch_size):
            pairs=torch.tensor(ranked[start:start+batch_size],dtype=torch.long)
            scores=torch.sigmoid(quantum(ht,hg,pairs)).cpu().numpy()
            outputs.extend(scores.tolist())
    summaries={}
    tx=public['transactions']
    totals=tx.groupby('account_id').agg(transaction_count=('amount_cad','size'),
                           transaction_total_cad=('amount_cad','sum'))
    last_events=tx.groupby('account_id')['timestamp'].max()
    for a in ids:
        summaries[a]={'transaction_count':int(totals.loc[a,'transaction_count']) if a in totals.index else 0,
                      'transaction_total_cad':round(float(totals.loc[a,'transaction_total_cad']),2) if a in totals.index else 0.0,
                      'last_transaction_utc':str(last_events.loc[a]) if a in last_events.index else ''}
    records=[]
    for k,((i,j),score) in enumerate(zip(ranked,outputs),1):
        a,b=ids[i],ids[j]
        evidence=candidate_map[(i,j)]
        row={'case_id':f'QG-{k:07d}','account_a':str(a),'account_b':str(b),
             'link_score':float(score),'source':'trained_quantum_checkpoint',
             'snapshot_utc':snapshot.isoformat()}
        for label in ['shared_devices','shared_networks','shared_payments']:
            row[label]='|'.join(sorted(evidence[label]))
        for prefix,account in [('a',a),('b',b)]:
            for attr,value in summaries[account].items():
                row[f'{prefix}_{attr}']=value
        records.append(row)
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records,columns=['case_id','account_a','account_b','link_score','source','snapshot_utc',
                 'shared_devices','shared_networks','shared_payments',
                 'a_transaction_count','a_transaction_total_cad','a_last_transaction_utc',
                 'b_transaction_count','b_transaction_total_cad','b_last_transaction_utc']).to_csv(out,index=False)
    manifest={'input_root':str(root.resolve()),'model_directory':str(checkpoint_dir.resolve()),
              'accounts_scanned':len(ids),'candidate_pairs_total':len(candidate_map),
              'candidate_pairs_scored':len(ranked),'snapshot_utc':snapshot.isoformat(),
              'partial_scan':len(ranked)<len(candidate_map),'note':'Ownership linkage, NOT fraud classification.'}
    (out.parent/'inference_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Real Q-Guard quantum model batch inference')
    ap.add_argument('--data',type=Path,required=True,help='Dataset root containing manifest.json and raw/')
    ap.add_argument('--checkpoints',type=Path,default=Path('./checkpoints'))
    ap.add_argument('--out',type=Path,default=Path('./exports/cases.csv'))
    ap.add_argument('--max-candidates',type=int,default=None,
                    help='Limit number of high-evidence candidate pairs; omit for full candidate scan.')
    ap.add_argument('--batch-size',type=int,default=64)
    args=ap.parse_args()
    print(json.dumps(run(args.data,args.checkpoints,args.out,args.max_candidates,args.batch_size),indent=2))
