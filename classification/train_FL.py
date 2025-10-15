import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from utils import normalize, eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--saga_epoch", type=int, default=500)
parser.add_argument("--saga_batch_size", type=int, default=256)

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        return {key: torch.tensor(values[idx]) for key, values in self.texts.items()}

    def __len__(self):
        return len(self.texts['input_ids'])

def build_loaders(texts, mode):
    dataset = ClassificationDataset(texts)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if mode == "train" else False
    )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # ---- Parse cbl_path: acs/dataset/backbone/xxx.pt
    acs = args.cbl_path.split("/")[0]
    dataset = args.cbl_path.split("/")[1]
    if 'sst2' in dataset:
        dataset = dataset.replace('_', '/')
    backbone = args.cbl_path.split("/")[2]   # 'roberta_cbm' hoặc 'gpt2_cbm'
    backbone = 'roberta' if 'roberta' in backbone else 'gpt2'
    cbl_name = args.cbl_path.split("/")[-1]  # ví dụ: cbl_no_backbone_acc.pt

    # ---- Load datasets
    print("loading data...")
    train_dataset = load_dataset(dataset, split='train')
    # Map string label -> int nếu có cột 'target' (PubMed-20k)
    label2id = {"OBJECTIVE":0, "BACKGROUND":1, "METHODS":2, "RESULTS":3, "CONCLUSIONS":4}
    if 'target' in train_dataset.column_names and 'label' not in train_dataset.column_names:
        train_dataset = train_dataset.map(lambda e: {"label": label2id.get(e["target"], 0)})

    if dataset == 'SetFit/sst2':
        val_dataset = load_dataset(dataset, split='validation')
    test_dataset = load_dataset(dataset, split='test')

    # Map cho test (và val) nếu cần
    if 'target' in test_dataset.column_names and 'label' not in test_dataset.column_names:
        test_dataset = test_dataset.map(lambda e: {"label": label2id.get(e["target"], 0)})
    if dataset == 'SetFit/sst2' and 'target' in val_dataset.column_names and 'label' not in val_dataset.column_names:
        val_dataset = val_dataset.map(lambda e: {"label": label2id.get(e["target"], 0)})

    print("training data len: ", len(train_dataset))
    if dataset == 'SetFit/sst2':
        print("val data len: ", len(val_dataset))
    print("test data len: ", len(test_dataset))
    print("tokenizing...")

    # ---- Tokenizer
    if backbone == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif backbone == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")

    text_col = CFG.example_name[dataset]

    # ---- Tokenize helpers + keep only numeric cols
    def encode_and_filter(ds):
        encoded = ds.map(
            lambda e: tokenizer(e[text_col], padding=True, truncation=True, max_length=args.max_length),
            batched=True, batch_size=len(ds)
        ).remove_columns([text_col])
        # remove legacy extra cols if they exist
        if 'label_text' in encoded.column_names:
            encoded = encoded.remove_columns(['label_text'])
        if 'title' in encoded.column_names:
            encoded = encoded.remove_columns(['title'])
        # keep only tokenizer numeric cols + label
        keep_cols = ['input_ids', 'attention_mask', 'label']
        if 'token_type_ids' in encoded.column_names:
            keep_cols.append('token_type_ids')
        encoded = encoded.remove_columns([c for c in encoded.column_names if c not in keep_cols])
        # materialize
        encoded = encoded[:len(encoded)]
        return encoded

    encoded_train_dataset = encode_and_filter(train_dataset)
    if dataset == 'SetFit/sst2':
        encoded_val_dataset = encode_and_filter(val_dataset)
    encoded_test_dataset = encode_and_filter(test_dataset)

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="train")
    if dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")
    test_loader = build_loaders(encoded_test_dataset, mode="test")

    concept_set = CFG.concept_set[dataset]

    # ---- Load CBL / backbone+CBL
    if backbone == 'roberta':
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
        else:
            print("preparing backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    elif backbone == 'gpt2':
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
        else:
            print("preparing backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    else:
        raise Exception("backbone should be roberta or gpt2")

    # ---- Extract concept features
    print("get concept features...")
    FL_train_features, FL_test_features = [], []
    if dataset == 'SetFit/sst2':
        FL_val_features = []

    # train
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                feats = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                if backbone == 'roberta':
                    feats = feats[:, 0, :]
                elif backbone == 'gpt2':
                    feats = eos_pooling(feats, batch["attention_mask"])
                feats = cbl(feats)
            else:
                feats = backbone_cbl(batch)
        FL_train_features.append(feats)

    # val (only SST-2)
    if dataset == 'SetFit/sst2':
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if 'no_backbone' in cbl_name:
                    feats = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    if backbone == 'roberta':
                        feats = feats[:, 0, :]
                    elif backbone == 'gpt2':
                        feats = eos_pooling(feats, batch["attention_mask"])
                    feats = cbl(feats)
                else:
                    feats = backbone_cbl(batch)
            FL_val_features.append(feats)

    # test
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                feats = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                if backbone == 'roberta':
                    feats = feats[:, 0, :]
                elif backbone == 'gpt2':
                    feats = eos_pooling(feats, batch["attention_mask"])
                feats = cbl(feats)
            else:
                feats = backbone_cbl(batch)
        FL_test_features.append(feats)

    train_c = torch.cat(FL_train_features, dim=0).detach().cpu()
    if dataset == 'SetFit/sst2':
        val_c = torch.cat(FL_val_features, dim=0).detach().cpu()
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    # ---- Normalize + ReLU
    train_c, train_mean, train_std = normalize(train_c, d=0)
    train_c = F.relu(train_c)

    prefix = "./" + acs + "/" + dataset.replace('/', '_') + "/" + backbone + "/"
    os.makedirs(prefix, exist_ok=True)
    model_name = cbl_name[3:]  # strip "cbl"

    torch.save(train_mean, prefix + 'train_mean' + model_name)
    torch.save(train_std,  prefix + 'train_std'  + model_name)

    if dataset == 'SetFit/sst2':
        val_c, _, _ = normalize(val_c, d=0, mean=train_mean, std=train_std)
        val_c = F.relu(val_c)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # ---- Labels + loaders for glm_saga
    train_y = torch.LongTensor(encoded_train_dataset["label"])
    indexed_train_ds = IndexedTensorDataset(train_c, train_y)

    if dataset == 'SetFit/sst2':
        val_y = torch.LongTensor(encoded_val_dataset["label"])
        val_ds = TensorDataset(val_c, val_y)

    test_y = torch.LongTensor(encoded_test_dataset["label"])
    test_ds = TensorDataset(test_c, test_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    if dataset == 'SetFit/sst2':
        val_loader_saga = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader_saga = DataLoader(test_ds, batch_size=args.saga_batch_size, shuffle=False)

    # ---- Train final linear with glm_saga
    print("dim of concept features: ", train_c.shape[1])
    linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset])
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': 0.0007}}

    print("training final layer...")
    if dataset == 'SetFit/sst2':
        output_proj = glm_saga(
            linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
            val_loader=val_loader_saga, test_loader=test_loader_saga, do_zero=True,
            n_classes=CFG.class_num[dataset]
        )
    else:
        output_proj = glm_saga(
            linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
            test_loader=test_loader_saga, do_zero=True,
            n_classes=CFG.class_num[dataset]
        )

    print("save weights with test acc:", output_proj['path'][-1]['metrics']['acc_test'])
    W_g = output_proj['path'][-1]['weight']
    b_g = output_proj['path'][-1]['bias']

    # Sparse
    if dataset == 'SetFit/sst2':
        output_proj = glm_saga(
            linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, epsilon=1, k=1,
            val_loader=val_loader_saga, test_loader=test_loader_saga, do_zero=False,
            n_classes=CFG.class_num[dataset], metadata=metadata, n_ex=train_c.shape[0]
        )
    else:
        output_proj = glm_saga(
            linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, epsilon=1, k=1,
            test_loader=test_loader_saga, do_zero=False,
            n_classes=CFG.class_num[dataset], metadata=metadata, n_ex=train_c.shape[0]
        )
    print("save the sparse weights with test acc:", output_proj['path'][0]['metrics']['acc_test'])
    W_g_sparse = output_proj['path'][0]['weight']
    b_g_sparse = output_proj['path'][0]['bias']

    torch.save(W_g,         prefix + 'W_g'         + model_name)
    torch.save(b_g,         prefix + 'b_g'         + model_name)
    torch.save(W_g_sparse,  prefix + 'W_g_sparse'  + model_name)
    torch.save(b_g_sparse,  prefix + 'b_g_sparse'  + model_name)