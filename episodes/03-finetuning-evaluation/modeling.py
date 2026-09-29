import os
os.environ['USE_TF']='0'; os.environ['USE_FLAX']='0'; os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['TORCHDYNAMO_DISABLE']='1'
import gc
import math
import random
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from laya.common import DecisionModel, build_sequence, render_options, serialize_state, collate_items
from common import read, save, sha, optimizer_groups
from logic import metrics


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.set_num_threads(2)
    if hasattr(torch.backends,'mha'): torch.backends.mha.set_fastpath_enabled(False)


def get_base(protocol):
    base=Path(snapshot_download(protocol['model_id'],revision=protocol['model_revision'],token=False,
        allow_patterns=['model.safetensors','rl_agent_config.json','encoder/*.json','tokenizer/*']))
    assert sha(base/'model.safetensors')==protocol['base_sha256']
    return base


def load_model(base):
    cfg=read(base/'rl_agent_config.json'); ec=AutoConfig.from_pretrained(base/'encoder',local_files_only=True)
    ec.reference_compile=False
    encoder=AutoModel.from_config(ec,attn_implementation='sdpa')
    model=DecisionModel(encoder,cfg.get('head_layers',2),len(cfg.get('act_costs',{}))+1)
    state=load_file(str(base/'model.safetensors'),device='cpu'); model.load_state_dict(state,strict=True)
    del state; gc.collect()
    original=model.encoder._update_attention_mask
    def repair(attention_mask, output_attentions):
        repaired=[]
        for mask in original(attention_mask,output_attentions):
            minimum=torch.finfo(mask.dtype).min
            empty=(mask==minimum).all(dim=-1,keepdim=True)
            assert not (empty & attention_mask[:,None,:,None].bool()).any()
            repaired.append(AttentionMaskConverter._unmask_unattended(mask,minimum))
        return tuple(repaired)
    model.encoder._update_attention_mask=repair
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.cuda(); groups=optimizer_groups(model)
    assert sum(p.numel() for g in groups for p in g['params'])==421029889
    tok=AutoTokenizer.from_pretrained(base/'tokenizer',local_files_only=True)
    return model,tok,cfg,groups


def encode(rows,tok,protocol):
    names=protocol['labels']; q=protocol['question']
    options=[[tok.mask_token_id]+tok(' '+s,add_special_tokens=False)['input_ids'] for s in render_options(q)]
    assert len(options)==77 and max(map(len,options))<=49
    head=sum(map(len,options))+64; maximum=int(math.ceil((head+260)/64)*64)
    instruction=tok('choice question: '+q['ins'],add_special_tokens=False)['input_ids']
    assert len(instruction)<=64
    encoded=[]
    for row in rows:
        state={'message':row['text']}
        ids,markers=build_sequence(tok,state,q,max_len=maximum,head_max_len=head)
        assert len(markers)==77
        for i,p in enumerate(markers): assert ids[p:p+len(options[i])]==options[i]
        text=tok(serialize_state(state).replace(tok.mask_token,' '),add_special_tokens=False)['input_ids']
        assert ids[-len(text)-1:-1]==text, 'Customer input was truncated'
        encoded.append({'ids':ids,'markers':markers,'qtype':0,'label':names.index(row['label']),'id':row['id']})
    return encoded,{'head_max_len':head,'max_len':maximum,'max_tokens':max(len(r['ids']) for r in encoded)}


def loader(items,tok,shuffle):
    import functools
    return torch.utils.data.DataLoader([[r] for r in items],batch_size=4,shuffle=shuffle,num_workers=0,
        collate_fn=functools.partial(collate_items,pad_id=tok.pad_token_id),pin_memory=True)


def forward(model,batch):
    values={k:v.to('cuda',non_blocking=True) for k,v in batch.items() if k not in ('label','meta')}
    with torch.autocast('cuda',dtype=torch.float16): logits,_=model(**values)
    assert torch.isfinite(logits).all(), 'Nonfinite logits'
    return logits.float()


@torch.inference_mode()
def score(model,items,tok,names):
    model.eval(); records=[]; offset=0
    for batch in loader(items,tok,False):
        p=forward(model,batch).softmax(-1).cpu().tolist()
        for row,prob in zip(items[offset:offset+len(p)],p):
            records.append({'id':row['id'],'gold':row['label'],'pred':max(range(77),key=prob.__getitem__),'probs':prob})
        offset+=len(p)
    assert offset==len(items)
    return metrics(records,names),records


def save_weights(path,model):
    # Inference snapshot only: never clone the ~3.4GB Adam states onto CPU.
    state={n:p.detach().cpu().contiguous() for n,p in model.state_dict().items()}
    save_file(state,str(path)); del state; gc.collect()


def verify_weights(path,model):
    # Read one tensor at a time; do not create a second full CPU model.
    state=model.state_dict()
    with safe_open(str(path),framework='pt',device='cpu') as stored:
        assert set(stored.keys())==set(state)
        for name,value in state.items():
            assert torch.equal(stored.get_tensor(name),value.detach().cpu()),name
    return {'tensor_count':len(state),'exact_weight_roundtrip':True,'bytes':path.stat().st_size}


def export_model(directory,model,tok,cfg,budget):
    directory.mkdir(parents=True,exist_ok=False)
    save_weights(directory/'model.safetensors',model)
    model.encoder.config.save_pretrained(directory/'encoder'); tok.save_pretrained(directory/'tokenizer')
    new=dict(cfg); new.update(head_max_len=budget['head_max_len'],max_len=budget['max_len'],
        temperature=[1.,1.,1.],temperature_by_options={})
    new['adaptation']={'method':'supervised CE on original native decision logits','action_head_unchanged':True,'confidence_recalibrated':False}
    save(directory/'rl_agent_config.json',new)
