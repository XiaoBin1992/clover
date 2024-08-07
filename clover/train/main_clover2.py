import argparse
import math
import shutil
import torch.distributed as dist
import json
from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os
ENABLE_DEEPSPEED = os.getenv('ENABLE_DEEPSPEED', 0)
if ENABLE_DEEPSPEED:
    print(f"========================== use deepspeed ===================")
    import deepspeed
else:
    import accelerate
    from accelerate import Accelerator
    from accelerate.utils import set_seed

TEST_MODE=0
import torch

from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from contextlib import nullcontext
from transformers import get_linear_schedule_with_warmup, AutoConfig, get_cosine_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True
# from ..model.cnets import Model
# from ..model.clover import CloverModel
from clover.model.clover2 import Clover2Model, ConfigClover
from clover.model.configs import EConfig


parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--project', type=str, default='ess')
parser.add_argument('--basepath', type=str, default='/cpfs01/user/xiaobin/glj/models/vicuna-7b-v1.5/')
parser.add_argument('--evaldata', type=str, default='')
parser.add_argument('--configpath', type=str, default="config.json")
# parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient_checkpointing', type=bool, default=False)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument('--label_mask', type=bool, default=False)
parser.add_argument('--start_epoch', type=int, default=0)
if ENABLE_DEEPSPEED:
    parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
print(args)

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 8000, #2000 16800 epoch10 grad_step1
    "total_steps": 1600000, #800000
    "p_w": 1.0,
    "v_w": 10.0,
    # "head_w": 0.1,
    "num_workers": 2,
    # "embeding": True,
    # "act": "No",
    "data_noise": False,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    # "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    # "grad_clip": 1.0,
    "save_freq": 1,
    # "save_freq_batch": 100,
    "clover": {
        "num_heads": 5,
        "num_layers": 2,
        "heads_coefficient": 1.0,
        "decay_coefficient": 0.7,
    }
}
print(train_config)

config_clover = ConfigClover(train_config["clover"])

if ENABLE_DEEPSPEED:
    deepspeed.init_distributed()
    torch.cuda.set_device(args.local_rank)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor = torch.tensor([rank]).cuda()
    
    is_main_process = rank == 0
    is_local_main_process = rank == 0
else:
    set_seed(0)
    accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
    is_main_process = accelerator.is_main_process
    is_local_main_process = accelerator.is_local_main_process
    rank = None

def print_tensor(name, tensor):
    variance = tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
    print(f"print_tensor {name}: {tensor.size()} {tensor.dtype} {torch.isnan(tensor).any()} {torch.max(tensor)} {torch.min(tensor)} variance:{variance.size()} {torch.min(variance)}")# {tensor} 

def gather_for_metrics(tensor):
    if not ENABLE_DEEPSPEED:
        return accelerator.gather_for_metrics(tensor)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor).cuda()
    if len(tensor.size()) == 0:
        tensor = tensor.view(1)
    # tensor = tensor.clone()
    assert dist.is_initialized()
    if rank == 0:
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gathered_tensors = None
    dist.gather(tensor, gathered_tensors, dst=0)
    if rank == 0:
        gathered_tensors = torch.concat(gathered_tensors, dim=-1)
    return gathered_tensors


# torch.distributed.all_reduce(tensor, op=dist.ReduceOp.SUM)
# print(f"all_reduce tensor:{tensor}")
# tensor = gather_for_metrics(tensor)
# print(f"gather_for_metrics tensor2:{tensor}")

#if accelerator.is_main_process:
if is_main_process:
    import wandb

    wandb.init(project=args.project, config=train_config)

# vicua frozen head 
# # ==============================================
baseconfig = AutoConfig.from_pretrained(args.basepath)

# head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim]
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"]

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False) #, dtype=baseconfig.torch_dtype
head.weight.data = tensor.to(head.weight.data.dtype)

variance = tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
# print(f"variance:{variance.size()} {torch.min(variance)}")
print(f"head: {head} {head.weight.data.dtype}, {tensor.size()} {tensor.dtype} {torch.min(torch.abs(tensor))}")
# head.eval()

# ==============================================


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

def copy_and_updata_config_file(src_path, dest_dir):  
    os.makedirs(dest_dir, exist_ok=True)  
    with open(src_path, "r") as f:
        config_js = json.loads(f.read())
        config_js["clover"] = train_config["clover"]
    dest_path = os.path.join(dest_dir, 'config.json')  
    #shutil.copy(src_path, dest_path)  
    with open(dest_path, 'w') as file:
        json.dump(config_js, file)
    print(f"Copied and renamed file to {dest_path}")  

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        tensor_norm = data["hidden_state_big_onnorm"]
        noise_norm = torch.randn(tensor_norm.size()) * self.std + self.mean
        noisy_tensor_norm = tensor_norm + noise_norm
        data["hidden_state_big_onnorm"] = noisy_tensor_norm
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        tensor_norm = data["hidden_state_big_onnorm"]
        noise_norm = (torch.rand_like(tensor_norm) - 0.5) * self.std * 512 / tensor_norm.shape[1]
        noisy_tensor_norm = tensor_norm + noise_norm
        data["hidden_state_big_onnorm"] = noisy_tensor_norm
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        hidden_state_onnorm = data['hidden_state_onnorm'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]

        # except:
        #     with open("error_path.txt", "w") as file:
        #         file.write(self.data[index])
        #     print('error path',self.data[index])

        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        # loss_mask.extend([0,0])
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["hidden_state_big_onnorm"] = hidden_state_onnorm
        new_data["input_ids"] = input_ids_target
        # sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        # label = data['y']

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_hidden_states_onnorm = torch.cat([self.paddingtensor(item['hidden_state_big_onnorm'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "hidden_states_onnorm": batch_hidden_states_onnorm,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


@torch.no_grad()
def getkacc_model(model, data, head, max_length=config_clover.num_heads):
    hidden_states = data["hidden_states"]
    hidden_states_onnorm = data["hidden_states_onnorm"]
    input_ids = data["input_ids"]
    # attention_mask=data["attention_mask"]
    loss_mask = data["loss_mask"]
    # sample_mask=data["sample_mask"]
    target = data["target"]
    # total = [0 for _ in range(max_length)]
    # correct = [0 for _ in range(max_length)]
    total_dic = {1: [0 for _ in range(max_length)], 
                 2: [0 for _ in range(max_length)],
                 3: [0 for _ in range(max_length)],
                 5: [0 for _ in range(max_length)],
                 10: [0 for _ in range(max_length)],}
    correct_dic = {1: [0 for _ in range(max_length)], 
                 2: [0 for _ in range(max_length)],
                 3: [0 for _ in range(max_length)],
                 5: [0 for _ in range(max_length)],
                 10: [0 for _ in range(max_length)],}
    bs, sl = hidden_states_onnorm.shape[0], hidden_states_onnorm.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

    for i in range(bs):
        for j in range(1, sl):

            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            
            # single_m_hidden_states = single_hidden_states.clone()
            single_m_hidden_states = hidden_states_onnorm[i, :j][None, :, :]
            # print(f"single_m_hidden_states:{single_m_hidden_states.dtype} {model.module.clover_head_mlp_rnn.v.weight.data.dtype} {model.module.clover_embed_tokens.weight.data.dtype}")
            flag = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0}
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break
                tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                # tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                
                if not (target_in_token == tmp_token):
                    break
                    
                if k == 0:
                    single_m_hidden_states = model.module.forward_layer(single_m_hidden_states, single_input_ids)
                    out_hidden, out_m_hidden = model.module.forward_rnn(single_m_hidden_states, single_input_ids, k)
                else:
                    out_hidden, out_m_hidden = model.module.forward_rnn(single_m_hidden_states, single_input_ids, k)
                
                
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                top_prob, top_indices = torch.topk(last_headout, k=10, largest=True)
                # print(k, ': ', top_prob)
                # print('eval K: ', k, ' token: ', token)
                # print(top_indices)
                # print(top_pro)
                for _top_k in total_dic.keys():
                    total_dic[_top_k][k] += 1
                    if target_out_token in top_indices[0, :_top_k] and flag[_top_k]==0:
                        correct_dic[_top_k][k] += 1
                        # print(target_out_token, top_indices[:, :_top_k])
                    else:
                        flag[_top_k] = 1
                
                # total[k] += 1
                # if token == target_out_token:
                #     correct[k] += 1
                # else:
                #     for kk in range(k + 1, max_length):
                #         total[kk] += 1
                #     break

                single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)),
                                            dim=1)
                single_m_hidden_states = torch.cat((single_m_hidden_states, out_m_hidden[:, -1:]), dim=1)
    
    return correct_dic, total_dic


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None
# aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
# print('td',train_config["datapath"])
# print(datapath)
# exit()
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
test_eval_loader = None
if args.evaldata != None:
    testevaldatapath = list_files(args.evaldata)
    testevaldataset = CustomDataset(testevaldatapath)
else:
    testevaldataset = None
# for batch_data in train_loader:
#     print(batch_data)

if is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

# model , criterion , optimizer, num_epochs , num_warmup_steps , total_steps , scheduler , 
# ===========================================================================
config = EConfig.from_pretrained(train_config["config_path"])
model = Clover2Model(config, head, config_clover=config_clover, load_emb=True, path=args.basepath)

# criterion = nn.CrossEntropyLoss(reduction='none')
# 
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if ENABLE_DEEPSPEED:

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                    model=model,
                                                                    model_parameters=model.parameters(),
                                                                    training_data=traindataset,
                                                                    collate_fn=DataCollatorWithPadding()
                                                                    )
    for param in head.parameters():
        param.requires_grad = True

    head_engine, _, test_loader, _ = deepspeed.initialize(args=args,
                                                        model=head,
                                                        model_parameters=head.parameters(),
                                                        training_data=testdataset,
                                                        collate_fn=DataCollatorWithPadding()
                                                        )
    _, _, test_eval_loader, _ = deepspeed.initialize(args=args,
                                                        model=head,
                                                        model_parameters=head.parameters(),
                                                        training_data=testevaldataset,
                                                        collate_fn=DataCollatorWithPadding()
                                                        )
        
    for param in head.parameters():
        param.requires_grad = False
    
    forward_data_type = head_engine.module.weight.data.dtype
else:
    train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                            pin_memory=True)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

    if testevaldataset is not None:
        test_eval_loader = DataLoader(testevaldataset, batch_size=train_config["bs"], shuffle=False,
                                collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))
    if is_warmup:
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
        #                                 num_training_steps=total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)

        model, head, optimizer, train_loader, test_loader, scheduler, test_eval_loader = accelerator.prepare(
            model, head, optimizer, train_loader, test_loader, scheduler, test_eval_loader
        )
    else:
        model, head, optimizer, train_loader, test_loader, test_eval_loader = accelerator.prepare(
            model, head, optimizer, train_loader, test_loader, test_eval_loader
        )
        
    for param in head.parameters():
        param.requires_grad = False
    model_engine = model
    head_engine = head
    forward_data_type = None#head_engine.weight.data.dtype
#forward_data_type = torch.bfloat16
print(f"forward_data_type: {forward_data_type}")

#==============================================================================
if args.start_epoch != 0:
    assert not ENABLE_DEEPSPEED
    accelerator.load_state(f"{args.cpdir}/epoch_{args.start_epoch + 1}")

torch.cuda.empty_cache()
print(f"model init {torch.cuda.memory_allocated()} {torch.cuda.max_memory_allocated()}")
print(f"start_epoch:{args.start_epoch}")

def top_k_logits(logits, sampel_topk):
    values, _ = torch.topk(logits, sampel_topk)
    min_values = values[..., -1, None]
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')),
        logits,
    )

def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_mask = cumulative_probs < p
    sorted_mask[..., 1:] = sorted_mask[..., :-1]
    sorted_mask[..., 0] = 1

    logits_mask = sorted_indices.new_zeros(logits.size(), dtype=torch.bool).scatter_(
        -1, sorted_indices, sorted_mask
    )
    return logits.masked_fill(~logits_mask, float('-inf'))

def get_id_mask_pre(head, hidden_states, input_ids_indices, num_heads, sampel_topk=8, sampel_topp=0.8):
    input_ids_cor = head(hidden_states)
    input_ids_cor = nn.Softmax(dim=2)(input_ids_cor)
    top_k_logits_values = top_k_logits(input_ids_cor, sampel_topk)
    filtered_logits = top_p_logits(top_k_logits_values, sampel_topp)
    batch_size, seq_len, vocab_size = input_ids_cor.shape
    input_ids_cor = torch.zeros(batch_size, seq_len, 1, dtype=input_ids_cor.dtype)
    # 获取 top-k top-p 后的候选 token
    candidates = torch.argsort(filtered_logits, dim=-1, descending=True)[:, :, :sampel_topk]
    # 检查 ids 是否在 candidates 中
    ids_expanded = input_ids_indices.unsqueeze(-1).expand(-1, -1, sampel_topk)
    matches = (candidates == ids_expanded).any(dim=-1).to(dtype=input_ids_cor.dtype)
    # 将结果存储在 output_tensor 中
    input_ids_cor = matches.unsqueeze(-1)
    # print('filtered_logits: ', filtered_logits)
    # print('candidates: ', candidates)
    # print('output_tensor: ', output_tensor)
    # input_ids_cor = torch.gather(input_ids_cor, 2, input_ids_indices)
    input_ids_cor_list = []
    input_ids_cor_list.append(input_ids_cor)
    for i in range(1, num_heads):
        input_ids_cor_list.append(input_ids_cor_list[i-1][:,:-1]*input_ids_cor[:,i:])
    return input_ids_cor_list

def get_id_mask(head, hidden_states, input_ids_indices, num_heads, sampel_topk=64, sampel_topp=0.8):
    main_score = nn.functional.softmax(head(hidden_states), dim=-1)
    top_K_val, top_K_idx = main_score.topk(sampel_topk, dim=-1)
    top_K_val_sum = top_K_val.cumsum(dim=-1)
    top_p_sorted_indices_to_rm = top_K_val_sum - top_K_val > sampel_topp
    top_K_idx.masked_fill_(top_p_sorted_indices_to_rm, -200)
    labels_mask = top_K_idx.eq(input_ids_indices.unsqueeze(-1)).any(-1)#torch.logical_not()
    
    input_ids_cor = labels_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    input_ids_cor_list = []
    input_ids_cor_list.append(input_ids_cor)
    for i in range(1, num_heads):
        input_ids_cor_list.append(input_ids_cor_list[i-1][:,:-1]*input_ids_cor[:,i:])
    return input_ids_cor_list
    
def hand_data_fn(head, data, data_type = None):
    # data_type = None #torch.bfloat16
    def update_data(name, data_type=None):
        #print_tensor(f"input {name}", data[name])
        data_ = data[name]
        if data_type is not None:
            data_ = data_.to(data_type)
            #print_tensor(f"input cast {name}", data_)
        if rank is not None:
            data_ = data_.to(rank)
        data[name] = data_
    # data["hidden_states_onnorm"].requires_grad_()
    update_data("target", data_type=data_type)#
    update_data("loss_mask")
    update_data("input_ids")
    update_data("hidden_states_onnorm", data_type=data_type)#
    update_data("attention_mask") #, data_type=data_type
    update_data("hidden_states", data_type=data_type) #
    
    if not args.label_mask:
        data["input_ids_cor_list"] = [1.0] * config_clover.num_heads
    else:
        data["input_ids_cor_list"] = get_id_mask(head, data["hidden_states"], data["input_ids"], config_clover.num_heads, sampel_topk=8, sampel_topp=0.8)
        
for epoch in range(args.start_epoch, num_epochs + 1):
    top_3acc = [[0 for _ in range(3)] for i in range(config_clover.num_heads)]
    correct = [0 for _ in range(config_clover.num_heads)]
    total = [0 for _ in range(config_clover.num_heads)]
    epoch_loss = 0
    epoch_loss_i = [0 for _ in range(config_clover.num_heads)]
    num_batches = 0
    model.train()
    
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if TEST_MODE and batch_idx > 2:
            break
        ct = []
        cc = []
        topkacc = []
        with nullcontext() if ENABLE_DEEPSPEED else accelerator.accumulate(model):
            if not ENABLE_DEEPSPEED:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad()
            hand_data_fn(head_engine, data, data_type = forward_data_type)
            with torch.no_grad():
                # (f'head_engine:{head_engine}. {data["target"]}')
                # print(f"head_engine:{forward_data_type}, {data['target'].dtype}")
                main_logits = head_engine(data["target"])
                target_p = nn.Softmax(dim=2)(main_logits)
                target_p = target_p.detach()
                # print_tensor(f"input target_p", target_p)
            # logits_seq = model_engine(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            # clover_hidden_states, token_emb = model_engine(data["hidden_states_onnorm"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            hidden_states_seq_list = model_engine(data["hidden_states_onnorm"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            # print_tensor(f"model_engine hidden_states_seq_list", hidden_states_seq_list)
            # print(f"clover_hidden_states:{clover_hidden_states.size()}")
            # logp_seq = [0 for i in range(config_clover.num_heads)]
            # for i in range(config_clover.num_heads):
            #     logp_seq[i] = nn.LogSoftmax(dim=2)(logits_seq[i])
            loss_mask_in = data["loss_mask"][:, :, None]
            loss_i = [0 for i in range(config_clover.num_heads)]
            loss_i_p = [0 for i in range(config_clover.num_heads)]
            loss_i_v = [0 for i in range(config_clover.num_heads)]
            

            target = data["target"]
            input_ids_cor_list = data["input_ids_cor_list"]
            
            loss_mask_metrics = []
            # loss_list = []
            for i in range(0, config_clover.num_heads):
                with torch.no_grad():
                    cur_loss_mask_in = loss_mask_in[:, i:] * input_ids_cor_list[i]
                    total_size = max(loss_mask_in[:, i:].numel(), 1)
                    loss_mask_metrics.append((loss_mask_in[:, i:].sum().item() / total_size, cur_loss_mask_in.sum().item() / total_size))
                    
                # clover_hidden_states, hidden_states_seq, logits_seq = model_engine.module.forward_one_head(i, clover_hidden_states, token_emb)
                hidden_states_seq = hidden_states_seq_list[i]
                if i > 0:
                    hidden_states_seq = hidden_states_seq[:, :-i]
                logits_seq = model_engine.module.forward_lm_head(hidden_states_seq)
                # print_tensor(f"get_loss {i} hidden_states_seq", hidden_states_seq)
                # print_tensor(f"get_loss {i} logits_seq", logits_seq)
                vloss = nn.SmoothL1Loss(reduction="none")(hidden_states_seq.to(torch.float32), target[:, i:].to(torch.float32))
                cur_loss_mask_in_sum = cur_loss_mask_in.sum() + 1
                vloss = torch.sum(torch.mean(cur_loss_mask_in * vloss, 2)) / cur_loss_mask_in_sum
                logp_seq = nn.LogSoftmax(dim=2)(logits_seq)
                ploss = -torch.sum(torch.sum(cur_loss_mask_in * (logp_seq * target_p[:, i:]), 2)) / cur_loss_mask_in_sum
                # print(f"get_loss {i} vloss:{vloss} ploss:{ploss}")
                #     loss_list.append(vloss.view(1, 1))
                #     loss_list.append(ploss.view(1, 1))
                #     # print(f"vloss {vloss} {vloss.size()} {ploss} {ploss.size()}")
                # return torch.concat(loss_list, dim=-1).view(config_clover.num_heads, 2)
                # if False and args.gradient_checkpointing:
                #     all_loss = torch.utils.checkpoint.checkpoint(
                #         get_all_loss,
                #         loss_mask_in, data["target"], *hidden_states_seq_list
                #     )
                # else:
                #     all_loss = get_all_loss(loss_mask_in, data["target"], *hidden_states_seq_list)
                
                # print(f"vloss {i} vloss:{vloss} ploss:{ploss}")
                loss_scale = config_clover.decay_coefficient ** i * config_clover.heads_coefficient
                loss_i_p[i] = ploss
                loss_i_v[i] = vloss
                loss_i[i] = (train_config["v_w"] * vloss + train_config["p_w"] * ploss) * loss_scale
                
        
            loss = sum(loss_i)
            if ENABLE_DEEPSPEED:
                model_engine.backward(loss)
                model_engine.step()
            else:
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
                optimizer.step()
                if is_warmup:
                    scheduler.step()
                
            with torch.no_grad():
                for i in range(0, config_clover.num_heads):
                    hidden_states_seq = hidden_states_seq_list[i]
                    if i > 0:
                        hidden_states_seq = hidden_states_seq[:, :-i]
                    logits_seq = model_engine.module.forward_lm_head(hidden_states_seq)
                    out_head = logits_seq
                    target_head = main_logits[:, i:].contiguous()
                    # loss_mask = loss_mask_in[:, i:].contiguous()
                    cur_loss_mask_in = loss_mask_in[:, i:] * input_ids_cor_list[i]
                    _, predicted = torch.max(out_head, 2)
                    _, target = torch.max(target_head, 2)
                    ct.append(cur_loss_mask_in.sum().item())
                    cc.append(((predicted == target) * cur_loss_mask_in.squeeze()).sum().item())
                    out_head = out_head.view(-1, target_head.shape[-1])[cur_loss_mask_in.view(-1) == 1]
                    target = target.view(-1)[cur_loss_mask_in.view(-1) == 1]
                    topkacc.append(top_accuracy(out_head, target, (1, 2, 3)))
                    for top_i in range(len(topkacc[i])):
                        top_3acc[i][top_i] += topkacc[i][top_i]
                    total[i] += ct[i]
                    correct[i] += cc[i]
            
            # with torch.no_grad():
            #     ct = []
            #     cc = []
            #     topkacc = []
            #     for i in range(0, config_clover.num_heads):
            #         out_head = logits_seq[i]
            #         target_head = main_logits[:, i:].contiguous()
            #         loss_mask = loss_mask_in[:, i:].contiguous()
            #         _, predicted = torch.max(out_head, 2)
            #         _, target = torch.max(target_head, 2)
            #         ct.append(loss_mask.sum().item())
            #         cc.append(((predicted == target) * loss_mask.squeeze()).sum().item())
            #         out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            #         target = target.view(-1)[loss_mask.view(-1) == 1]
            #         topkacc.append(top_accuracy(out_head, target, (1, 2, 3)))
            #         for top_i in range(len(topkacc[i])):
            #             top_3acc[i][top_i] += topkacc[i][top_i]
            #         total[i] += ct[i]
            #         correct[i] += cc[i]
        if is_main_process and ct[0] != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/loss": loss.item()}
            for j in range(config_clover.num_heads):
                logdict[f'train/loss_head_{j+1}'] = loss_i[j].item()
                logdict[f'train/loss_head_{j+1}_p'] = loss_i_p[j].item()
                logdict[f'train/loss_head_{j+1}_v'] = loss_i_v[j].item()
                logdict[f'train/loss_mask1_head_{j+1}'] = loss_mask_metrics[j][0]
                logdict[f'train/loss_mask2_head_{j+1}'] = loss_mask_metrics[j][1]
                logdict[f'train/head_{j+1}_acc'] =  cc[j] / max(ct[j], 1)
                for id, i in enumerate(top_3acc[j]):
                    logdict[f'train/head_{j+1}_top_{id + 1}_acc'] = topkacc[j][id].item() / max(ct[j], 1)
            wandb.log(logdict)
            # for id,i in enumerate(top_3acc):
            #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})
        
        # del ploss, vloss
        epoch_loss += loss.item()
        for j in range(config_clover.num_heads):
            epoch_loss_i[j] += loss_i[j].item()
        num_batches += 1
        del loss_i
        
    epoch_loss /= num_batches
    for j in range(config_clover.num_heads):
        correct[j], total[j] = torch.tensor(correct[j]).cuda(), torch.tensor(total[j]).cuda()
        correct[j] = gather_for_metrics(correct[j]) #accelerator.
        total[j] = gather_for_metrics(total[j]) #accelerator.
        epoch_loss_i[j] /= num_batches
        top_3acc[j] = gather_for_metrics(top_3acc[j]) #accelerator.
        if is_local_main_process:
            correct[j], total[j] = correct[j].sum().item(), total[j].sum().item()
            total[j] = max(1, total[j])
            for id, i in enumerate(top_3acc[j]):
                wandb.log({f'train/head_{j+1}_epochtop_{id + 1}_acc': i.sum().item() / total[j]})
            print('Head_', j, ': ')
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Train Accuracy: {:.2f}%'.format(100 * correct[j] / total[j]))
            wandb.log({f"train/head_{j+1}_epochacc": correct[j] / total[j], f"train/head_{j+1}_epochloss": epoch_loss_i[j]})
    if is_local_main_process:
        wandb.log({f"train/epochloss": epoch_loss})
    
    if (epoch + 1) % train_config["save_freq"] == 0:
        top_3acc_test = [[0 for _ in range(3)] for i in range(config_clover.num_heads)]
        correct_test = [0 for i in range(config_clover.num_heads)]
        total_test = [0 for i in range(config_clover.num_heads)]
        epoch_loss_test = 0
        epoch_loss_test_i = [0 for i in range(config_clover.num_heads)]
        num_batches_test = 0
        model.eval()
        
        # k_acc = [[] for i in range(config_clover.num_heads)]
        for batch_idx, data_test in enumerate(tqdm(test_loader)):
            if TEST_MODE and batch_idx > 2:
                break
            with torch.no_grad():
                # hidden_states = model.forward_layer(data_test["hidden_states"])
                
                # input_ids = input_ids[:, -1:]
                # out_hidden, out_head = self.forward_rnn(hidden_states, input_ids=input_ids, head_idx=0)
                
                #
                # clover_hidden_states, token_emb = model_engine(data_test["hidden_states_onnorm"], input_ids=data_test["input_ids"], attention_mask=data_test["attention_mask"])
                
                
                hand_data_fn(head_engine, data_test, data_type = forward_data_type)
                
                main_logits_test = head_engine(data_test["target"])
                target_p_test = nn.Softmax(dim=2)(main_logits_test)
                target_p_test = target_p_test.detach()
                
                hidden_states_seq_list = model_engine(data_test["hidden_states_onnorm"], input_ids=data_test["input_ids"], attention_mask=data_test["attention_mask"])
                loss_mask_in_test = data_test["loss_mask"][:, :, None]
                input_ids_cor_list = data_test["input_ids_cor_list"]
                
                # logp_seq_test = [0 for i in range(config_clover.num_heads)]
                loss_test_i = [0 for _ in range(config_clover.num_heads)]
                ct_test = []
                cc_test = []
                topkacc_test = []
                for i in range(config_clover.num_heads):
                    # clover_hidden_states, hidden_states_seq, logits_seq = model_engine.module.forward_one_head(i, clover_hidden_states, token_emb)
                    cur_loss_mask_in_test = loss_mask_in_test[:, i:] * input_ids_cor_list[i]
                    
                    hidden_states_seq = hidden_states_seq_list[i]
                    if i > 0:
                        hidden_states_seq = hidden_states_seq[:, :-i]
                    logits_seq = model_engine.module.forward_lm_head(hidden_states_seq)
                    logp_seq_test = nn.LogSoftmax(dim=2)(logits_seq)
                    loss_test_i[i] = -torch.sum(torch.sum(cur_loss_mask_in_test * (logp_seq_test * target_p_test[:, i:]), 2)) / (cur_loss_mask_in_test.sum()+1e-5) * config_clover.decay_coefficient ** i * config_clover.heads_coefficient
                    
                    out_head = logits_seq
                    target_head = main_logits_test[:, i:].contiguous()
                    loss_mask = cur_loss_mask_in_test.contiguous()
                    _, predicted = torch.max(out_head, 2)
                    _, target = torch.max(target_head, 2)
                    ct_test.append(loss_mask.sum().item())
                    cc_test.append(((predicted == target) * loss_mask.squeeze()).sum().item())
                    out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                    target = target.view(-1)[loss_mask.view(-1) == 1]
                    topkacc_test.append(top_accuracy(out_head, target, (1, 2, 3)))
                    for top_i in range(len(topkacc_test[i])):
                        top_3acc_test[i][top_i] += topkacc_test[i][top_i]
                    total_test[i] += ct_test[i]
                    correct_test[i] += cc_test[i]
                
                loss_test = sum(loss_test_i)
            epoch_loss_test += loss_test.item()
            for j in range(config_clover.num_heads):
                epoch_loss_test_i[j] += loss_test_i[j].item()
            num_batches_test += 1
            
        
        for i in range(config_clover.num_heads):
            epoch_loss_test_i[i] /= num_batches_test
            correct_test[i], total_test[i] = torch.tensor(correct_test[i]).cuda(), torch.tensor(total_test[i]).cuda()
            correct_test[i] = gather_for_metrics(correct_test[i]) #accelerator.
            total_test[i] = gather_for_metrics(total_test[i]) #accelerator.
            top_3acc_test[i] = gather_for_metrics(top_3acc_test[i]) #accelerator.
            if is_local_main_process:
                correct_test[i], total_test[i] = correct_test[i].sum().item(), total_test[i].sum().item()
                total_test[i] = max(1, total_test[i])
                for id, n in enumerate(top_3acc_test[i]):
                    wandb.log({f'test/head_{i+1}_top_{id + 1}_acc': n.sum().item() / total_test[i]})
                wandb.log({f"test/head_{i+1}_epochacc": correct_test[i] / total_test[i]})
                print('Head_', i, ': ')
                print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss_test_i[i]))
                print('Test Accuracy: {:.2f}%'.format(100 * correct_test[i] / total_test[i]))
                wandb.log({f"test/head_{i+1}_epochloss": epoch_loss_test_i[i]})
        
        epoch_loss_test /= num_batches_test
        if is_local_main_process:
            wandb.log({f"test/epochloss": epoch_loss_test})
            if not ENABLE_DEEPSPEED and not TEST_MODE:
                accelerator.save_state(output_dir=f"{args.cpdir}/epoch_{epoch + 1}")
                copy_and_updata_config_file(args.configpath, f"{args.cpdir}/epoch_{epoch + 1}")  
                subdirs = [d for d in os.listdir(args.cpdir) if os.path.isdir(os.path.join(args.cpdir, d))]  
                subdirs = sorted(subdirs, key=lambda d: os.path.getmtime(os.path.join(args.cpdir, d)))  
                
                # 保留最新的1个文件夹，删除其余的  
                while len(subdirs) > 1: 
                    dir_to_remove = subdirs.pop(0)  
                    shutil.rmtree(os.path.join(args.cpdir, dir_to_remove))  
                    print(f"Deleted old checkpoint directory: {dir_to_remove}")  
        if ENABLE_DEEPSPEED and not TEST_MODE:
            model_engine.save_16bit_model(f"{args.cpdir}/state_{epoch}")
            deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.cpdir}/state_{epoch}")

        
        if test_eval_loader is not None:
            k_cor = [[] for _ in range(config_clover.num_heads)]
            k_tal = [[] for _ in range(config_clover.num_heads)]
            k_cor_dic = {1: [[] for _ in range(config_clover.num_heads)], 
                        2: [[] for _ in range(config_clover.num_heads)], 
                        3: [[] for _ in range(config_clover.num_heads)], 
                        5: [[] for _ in range(config_clover.num_heads)], 
                        10: [[] for _ in range(config_clover.num_heads)]}
            k_tal_dic = {1: [[] for _ in range(config_clover.num_heads)], 
                        2: [[] for _ in range(config_clover.num_heads)], 
                        3: [[] for _ in range(config_clover.num_heads)], 
                        5: [[] for _ in range(config_clover.num_heads)], 
                        10: [[] for _ in range(config_clover.num_heads)]}
            for batch_idx, data_test in enumerate(tqdm(test_eval_loader)):
                if TEST_MODE and batch_idx > 2:
                    break
                with torch.no_grad():
                    if batch_idx < 30:
                        hand_data_fn(head_engine, data_test, data_type = forward_data_type)
                        cors, tals, = getkacc_model(model_engine, data_test, head_engine, max_length=config_clover.num_heads)
                        for top_k_ in cors.keys():
                            for i in range(len(cors[top_k_])):
                                k_cor_dic[top_k_][i].append(cors[top_k_][i])
                                k_tal_dic[top_k_][i].append(tals[top_k_][i])
                    else:
                        break

            sum_cors_dic = {1: [], 2: [], 3: [], 5: [], 10: []}
            sum_tals_dic = {1: [], 2: [], 3: [], 5: [], 10: []}
            for top_k_ in sum_cors_dic.keys():
                for id, i in enumerate(k_cor_dic[top_k_]):
                    sum_cor = np.array(i).sum()
                    sum_cor = torch.tensor(sum_cor).cuda()
                    sum_cors_dic[top_k_].append(sum_cor)
                for id, i in enumerate(k_tal_dic[top_k_]):
                    sum_tal = np.array(i).sum()
                    sum_tal = torch.tensor(sum_tal).cuda()
                    sum_tals_dic[top_k_].append(sum_tal)

                sum_cors_dic[top_k_] = gather_for_metrics(sum_cors_dic[top_k_]) #accelerator.
                sum_tals_dic[top_k_] = gather_for_metrics(sum_tals_dic[top_k_]) #accelerator.

                if is_local_main_process:
                    for i in range(len(sum_cors_dic[top_k_])):
                        sum_cor = sum_cors_dic[top_k_][i].sum().item()
                        sum_tal = sum_tals_dic[top_k_][i].sum().item()
                        wandb.log({f"test_eval/{i}_acc_top{top_k_}": sum_cor/max(sum_tal,1)})


