'''
Multitask BERT class.

* class MultitaskBERT: Implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. 
* function test_multitask: Test procedure for MultitaskBERT. 
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity

from sentence_transformers import losses  

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module uses BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.config = config
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        )

        self.paraphrase_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )

        self.similarity_classifier = nn.Sequential(
            nn.Linear(1, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler_output_cache = {}


    def forward(self, input_ids, attention_mask, sent_ids=None, task=None):
        'Takes a batch of sentences and produces embeddings for them. In-built caching system during pre-training'
        final_embeddings = []
        if self.config.option == 'pretrain' and sent_ids is not None:
            cached_embeddings = [self.pooler_output_cache.get(sent_id) for sent_id in sent_ids]
            missing_indices = [i for i, embedding in enumerate(cached_embeddings) if embedding is None]
            if missing_indices:
                missing_input_ids = input_ids[missing_indices]
                missing_attention_mask = attention_mask[missing_indices]                
                outputs = self.bert(input_ids=missing_input_ids, attention_mask=missing_attention_mask)
                new_embeddings = outputs['last_hidden_state'][:, 0, :]
                if task not in ['sts_finetune']:
                    new_embeddings = self.dropout(new_embeddings)
                for idx, sent_id in enumerate(sent_ids):
                    if idx in missing_indices:
                        self.pooler_output_cache[sent_id] = new_embeddings[missing_indices.index(idx)]
                        final_embeddings.append(new_embeddings[missing_indices.index(idx)])
                    else:
                        final_embeddings.append(cached_embeddings[idx])
            else:
                final_embeddings = cached_embeddings
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            final_embeddings = outputs['last_hidden_state'][:, 0, :]
            if task not in ['sts_finetune']:
                final_embeddings = self.dropout(final_embeddings)
        if not isinstance(final_embeddings, torch.Tensor):
            final_embeddings = torch.stack(final_embeddings)

        return final_embeddings


    def predict_sentiment(self, input_ids, attention_mask, sent_ids=None):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        '''
        embeddings = self.forward(input_ids, attention_mask, sent_ids=sent_ids, task='sst')
        return self.sentiment_classifier(self.dropout(embeddings))
    
    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2, sent_ids=None):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        '''
        embeddings_1 = self.forward(input_ids_1, attention_mask_1, sent_ids=sent_ids, task='para')
        embeddings_2 = self.forward(input_ids_2, attention_mask_2, sent_ids=sent_ids, task='para')
        combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=1)
        paraphrase_logit = self.paraphrase_classifier(self.dropout(combined_embeddings))
        return paraphrase_logit

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2, sent_ids=None):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        '''
        embeddings_1 = self.forward(input_ids_1, attention_mask_1, sent_ids=sent_ids, task='sts')
        embeddings_2 = self.forward(input_ids_2, attention_mask_2, sent_ids=sent_ids, task='sts')
        cos = CosineSimilarity(dim=1, eps=1e-6)
        similarity_logit = self.similarity_classifier(cos(embeddings_1, embeddings_2).unsqueeze(-1)) 
        return similarity_logit    
    
    def predict_similarity_finetune(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2, sent_ids=None):
        
        embeddings_1 = self.forward(input_ids_1, attention_mask_1, sent_ids=sent_ids, task='sts_finetune')
        embeddings_2 = self.forward(input_ids_2, attention_mask_2, sent_ids=sent_ids, task='sts_finetune')
        cos_sim = F.cosine_similarity(embeddings_1, embeddings_2)
        return cos_sim


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, yelp_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.yelp_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, yelp_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.yelp_train,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    yelp_train_data = SentenceClassificationDataset(yelp_train_data, args)
    yelp_dev_data = SentenceClassificationDataset(yelp_dev_data, args)

    yelp_train_dataloader = DataLoader(yelp_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    # Not evaluating on yelp data
    yelp_dev_dataloader = DataLoader(yelp_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    

    para_train_data_pos = [item for item in para_train_data if item[2] == 1]

    para_train_data_pos = SentencePairDataset(para_train_data_pos, args)
    para_train_dataloader_pos = DataLoader(para_train_data_pos, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)



    task_weights = {'sst': 1, 'para': 0.3, 'sts': 1}

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_avg_normalized_score = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for dataloader in [sst_train_dataloader, para_train_dataloader, sts_train_dataloader]:
            for batch in tqdm(dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                optimizer.zero_grad()

                if dataloader in [sst_train_dataloader, yelp_train_dataloader]: 
                    input_ids, attention_mask, labels, sent_ids = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device), batch['sent_ids']
                    logits = model.predict_sentiment(input_ids, attention_mask, sent_ids=sent_ids)
                    loss = (F.cross_entropy(logits, labels.view(-1), reduction='sum') / args.batch_size) * task_weights['sst']
                elif dataloader in [para_train_dataloader, sts_train_dataloader]:
                    input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels, sent_ids = batch['token_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['token_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['labels'].to(device), batch['sent_ids']
                    if dataloader == para_train_dataloader:
                        logits = model.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, sent_ids=sent_ids)
                        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float()) * task_weights['para']
                    else: 
                        if args.option == 'pretrain': 
                            logits = model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, sent_ids=sent_ids)
                            loss = F.mse_loss(logits.squeeze(), labels.float()) * task_weights['sts']
                        else: 
                            labels_scaled = labels.float() / 5.0
                            cos_sim = model.predict_similarity_finetune(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, sent_ids=sent_ids)
                            loss = F.mse_loss(cos_sim.squeeze(), labels_scaled) * task_weights['sts']


                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / (num_batches)

        sentiment_accuracy,sst_y_pred, sst_sent_ids, paraphrase_accuracy, para_y_pred, para_sent_ids, sts_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        
        avg_normalized_score = (sentiment_accuracy + paraphrase_accuracy + ((sts_corr + 1)/2)) / 3

        if avg_normalized_score > best_avg_normalized_score:
            best_avg_normalized_score = avg_normalized_score
            save_model(model, optimizer, args, config, args.filepath)
        
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {avg_normalized_score :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, sentiment_data2, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.yelp_train,args.para_test, args.sts_test, split='test')

        sst_dev_data, sentiment_data2, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.yelp_train,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--yelp_train", type=str, default="data/yelp-train.csv")
    

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
