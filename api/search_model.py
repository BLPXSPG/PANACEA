#!/usr/bin/env python
# coding: utf-8


cuda_device = "1"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPUs")

# Set JAVA_HOME for this session
try:
    os.environ['JAVA_HOME']
except KeyError:
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64/'

import csv
import os
import re
import json
import pandas as pd
import numpy as np
import sys
import datetime
from tqdm import tqdm
import pickle
import math
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn as nn

import spacy

from sentence_transformers import SentenceTransformer, CrossEncoder, util
    
import pyserini
import pyserini.search as pysearch
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.base import hits_to_texts
from pygaggle.rerank.transformer import MonoT5


# model for NLI inference
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# model for sentence representation
from transformers import RobertaModel

###!python -m spacy download en_core_web_sm
import en_core_web_sm


class SearchModel():
    def __init__(self):
        self.tokenizer_nli = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        self.model_nli = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')
        self.model_nli = self.model_nli.to("cuda")

        self.tokenizer_sent = RobertaTokenizer.from_pretrained('roberta-large')
        self.model_sent = RobertaModel.from_pretrained('roberta-large')
        self.model_sent = self.model_sent.to("cuda")

        #saved_model_file = 'nli-san_simpl4vp1.pt'
        saved_model_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nli-san_simpl4vp1.pt")
        #index_folder = 'index_data_sources_paragraphs_VP1_B' 
        index_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "index_data_sources_paragraphs_VP1_B")
        model_sim_sents = 'paraphrase-MiniLM-L12-v2'

        numb_results_per_query = 10
        self.retrieval_results_cols = [n for n in range(1,numb_results_per_query+1)]

        self.max_tokens = 302
        self.padded_size = self.max_tokens*1024

        
        label_to_number_dict = {'False':0, 'True':1}
        self.number_to_label_dict = {v: k for k, v in label_to_number_dict.items()}

        # Search options
        self.top_k = 100
        self.top_k_to_check_after_reranking = 10

        # Similar sentences options
        self.max_simil_sents_per_doc = 3

        # Model inference options
        self.top_sents_to_consider = 5

        self.searcher = pysearch.SimpleSearcher(index_folder)
        self.reranker =  MonoT5()

        # # LOAD SAVED MODEL
        batch_size = 30
        self.model = NetC(50)
        self.model = torch.load(saved_model_file)
        self.model.eval()


        # model for sentence similarity
        self.encoder = SentenceTransformer(model_sim_sents)


        self.nlp = en_core_web_sm.load()

    def update_query(self, query):

        results_df = self.query_search(query,self.top_k,self.top_k_to_check_after_reranking)
        results_df = self.add_sents_contexts(results_df)
        simil_results_df = self.extract_simil_sents(results_df,query,self.max_simil_sents_per_doc)
        entail_results_df = self.calculate_entailment(simil_results_df,query)
        entail_results_df = self.add_top_sents(entail_results_df,self.top_sents_to_consider)
        all_logits = self.model_input_sents_repr(entail_results_df,query,self.max_tokens,self.top_sents_to_consider)
        all_inferences = self.model_input_sents_nli(entail_results_df,self.top_sents_to_consider)


        # # VERACITY ASSESSMENT MODEL - OUTPUT
        inputsinf1 = all_inferences[0][0]
        inputssent1 = all_logits[0][0]
        inputsinf2 = all_inferences[1][0]
        inputssent2 = all_logits[1][0]
        inputsinf3 = all_inferences[2][0]
        inputssent3 = all_logits[2][0]
        inputsinf4 = all_inferences[3][0]
        inputssent4 = all_logits[3][0]
        inputsinf5 = all_inferences[4][0]
        inputssent5 = all_logits[4][0]

        self.model.eval()

        with torch.no_grad():     
            outputs = self.model(inputsinf1,inputssent1,inputsinf2,inputssent2,inputsinf3,inputssent3,
                        inputsinf4,inputssent4,inputsinf5,inputssent5)
            prediction = torch.argmax(outputs, dim=-1)
            prediction = prediction.cpu()


        return self.full_searchresult_output(query,entail_results_df,prediction,outputs)

        #print(full_results_df.loc[0,'query'])
        #print(full_results_df.loc[0,'results'])
        #print(full_results_df.loc[0,'veracity'])




    def round_signif(self, numb,signif_digits):
        return round(numb, signif_digits - int(math.floor(math.log10(abs(numb)))) - 1)

    # # Search - BM25 + Reranker
    def query_search(self, query,top_k,top_k_to_check_after_reranking):
        cols_df = ['query']+self.retrieval_results_cols
        results_df = pd.DataFrame(columns=cols_df)

        ### Pyserini search
        hits = self.searcher.search(query, k=top_k)

        ### Pygaggle reranking
        for h in hits:
            h.contents = json.loads(h.raw)['contents']

        textstorerank = hits_to_texts(hits,'contents')
        pygagglequery = Query(query)
        rerankedhits = self.reranker.rerank(pygagglequery, textstorerank)
        rerankedhits.sort(key=lambda x: x.score, reverse=True)

        rerankedhits = rerankedhits[:top_k_to_check_after_reranking]

        for rh in rerankedhits:
            rh.metadata['score'] = rh.score
        
        results_df = results_df.append(dict(zip(cols_df,
                                        [query]+[rh.metadata for rh in rerankedhits])),ignore_index=True)
        return results_df


    # # Split content in sentences and their contexts
    def retr_read_contents(self, retrieved_result):
        return json.loads(retrieved_result['raw'])['contents']


    def extract_sentences(self, content):
        doc = self.nlp(content)
        return [sent.text for sent in doc.sents]

    def create_contexts(self, sent_list):    
        if len(sent_list) <= 1:
            contexts = sent_list
            return contexts
        
        contexts = []
        for n in range(len(sent_list)):
            if n == 0:
                contexts.append(sent_list[n]+' '+sent_list[n+1])
            elif n == len(sent_list)-1:
                contexts.append(sent_list[n-1]+' '+sent_list[n])
            else:
                contexts.append(sent_list[n-1]+' '+sent_list[n]+' '+sent_list[n+1])
        return contexts


    def create_sents_contexts_object(self, content):
        sents = self.extract_sentences(content)
        context = self.create_contexts(sents)
        obj = [[idx,sent,context[idx]] for idx,sent in enumerate(sents)]
        return obj


    def add_sents_contexts(self, results_df):
        for idx,row in results_df.iterrows():
            for res_idx in self.retrieval_results_cols:
                # Checks that result is not nan
                if row[res_idx] == row[res_idx]:
                    row[res_idx]['sents'] = self.create_sents_contexts_object(self.retr_read_contents(row[res_idx]))
        return results_df


    # # Select the most similar sentences
    def compare_sentences(self, query,sentences,max_sents):
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        corpus_embeddings = self.encoder.encode(sentences,convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=max_sents)
        hits = hits[0] #since each time only one query is used
        return hits


    def compare_obj_query(self, query,obj,max_sents):
        sents_part = obj['sents']
        sents_list = [el[1] for el in sents_part]
        
        new_sents_part = []
        hits = self.compare_sentences(query,sents_list,max_sents)
        for h in hits:
            if h == h:
                new_sents_part.append([h['corpus_id'],
                                sents_part[h['corpus_id']][1],
                                sents_part[h['corpus_id']][2],
                                self.round_signif(h['score'],3)])
            else:
                new_sents_part.append([np.nan,np.nan,np.nan,np.nan])
        return new_sents_part 


    def extract_simil_sents(self, results_df,query,max_simil_sents_per_doc):
        simil_results_df = pd.DataFrame(columns = ['query']+self.retrieval_results_cols)

        for idx,row in results_df.iterrows():
            row_results = [query]

            for res_idx in self.retrieval_results_cols:
                new_obj = deepcopy(row[res_idx])
                # Checks that result is not nan
                if row[res_idx] == row[res_idx]:
                    new_obj['sents'] = self.compare_obj_query(query,new_obj,max_simil_sents_per_doc)
                row_results.append(new_obj)

            simil_cols = ['query']+self.retrieval_results_cols
            simil_results_df = simil_results_df.append(
                dict(zip(simil_cols,row_results)),ignore_index=True)
        return simil_results_df


    # # Entailment calculation
    def entail_check(self, query,sent):
        input_ids = self.tokenizer_nli.encode(query, sent, truncation=True, padding = False, 
                                    return_tensors='pt').to("cuda")
        logits = self.model_nli(input_ids)[0]

        # using the three posibilities [0,1,2] 'contradiction', 'neutral', 'entailment'
        # any possibility could be excluded. E.g. Only [0,2]
        entail_contradiction_logits = logits[:,[0,1,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        contr_prob = probs[:,0].item() 
        neutr_prob = probs[:,1].item()
        entail_prob = probs[:,2].item()
    
        return(contr_prob,neutr_prob,entail_prob)


    def entail_obj_query(self, query,obj):
        sents_part = obj['sents']
        sents_list = [el[1] for el in sents_part]
        
        new_sents_part = []
        for el in sents_part:
            new_el = deepcopy(el)
            sent = el[1]
            # Checks that result is not nan
            if sent != '' and sent == sent:
                contr_prob,neutr_prob,entail_prob = self.entail_check(query,sent)
                signif_digits = 2
                sent_entail = [self.round_signif(contr_prob,signif_digits),
                                self.round_signif(neutr_prob,signif_digits),
                                self.round_signif(entail_prob,signif_digits)]
            else:
                sent_entail = np.nan
            new_el.append(sent_entail)
            new_sents_part.append(new_el)
        
        return new_sents_part 


    def calculate_entailment(self, simil_results_df,query):
        entail_results_df = pd.DataFrame(columns = ['query']+self.retrieval_results_cols)

        for idx,row in simil_results_df.iterrows():       
            row_results = [query]

            for res_idx in self.retrieval_results_cols:
                new_obj = deepcopy(row[res_idx])
                # Checks that result is not nan
                if row[res_idx] == row[res_idx]:
                    new_obj['sents'] = self.entail_obj_query(query,new_obj)
                row_results.append(new_obj)

            entail_cols = ['query']+self.retrieval_results_cols
            entail_results_df = entail_results_df.append(
                dict(zip(entail_cols,row_results)),ignore_index=True)
        return entail_results_df


    # # Select top sentences to use in the veracity assessment model
    def top_sim_sentences_fullinfo(self, obj_list,max_sent):
        sents_fullinfo_list = []
        for obj in obj_list:
            # Checks that result is not nan
            if obj == obj:
                for sent_el in obj['sents']:
                    # Checks that result is not nan
                    if sent_el[1] == sent_el[1]:
                        sent = sent_el[1]
                        ent_val = sent_el[4]
                        sim_val = sent_el[3]            
                        sents_fullinfo_list.append([sent,ent_val,sim_val])
                    else:
                        sents_fullinfo_list.append(['',[0.,0.,0.],0.])
            else:
                sents_fullinfo_list.append(['',[0.,0.,0.],0.])
        sents_fullinfo_list.sort(key=lambda x:x[2],reverse=True)
        return sents_fullinfo_list[:max_sent]


    def add_top_sents(self, entail_results_df,top_sents_to_consider):
        all_sents_fullinfo_list = []
        all_sents_list = []

        for idx,row in entail_results_df.iterrows():
            res_list = row[self.retrieval_results_cols].tolist()

            el_sents_fullinfo_list = self.top_sim_sentences_fullinfo(res_list,top_sents_to_consider)
            el_sents_list = [el[0] for el in el_sents_fullinfo_list]

            all_sents_fullinfo_list.append(el_sents_fullinfo_list)
            all_sents_list.append(el_sents_list)

        # Consider also nan sentences
        all_sents_list_joined = ['\n'.join(str(el)) for el in all_sents_list]
        
        entail_results_df['top_sents_fullinfo'] = all_sents_fullinfo_list
        entail_results_df['top_sents'] = all_sents_list
        entail_results_df['top_sents_joined'] = all_sents_list_joined
        
        return entail_results_df


    # # MODEL INPUT - Sentence representation
    def model_input_sents_repr(self, entail_results_df,query,max_tokens,top_sents_to_consider):
        #padded_size = max_tokens*1024

        # Pad the tensors
        def tens_pad(tens):
            new_tens = torch.nn.functional.pad(tens, (0,0,0,max_tokens-tens.shape[1]), mode='constant', value=0)
            return new_tens

        all_logits = []

        self.model_sent.eval()
        with torch.no_grad():
            for sent_numb in range(top_sents_to_consider):
                logits_h = []
                for idx,row in entail_results_df.iterrows():
                    encoded_inputs_1 = self.tokenizer_sent(query,row['top_sents'][sent_numb],
                                                    truncation=True, padding = True, return_tensors='pt')
                    outputs = self.model_sent(**encoded_inputs_1.to(device))
                    logits = outputs.last_hidden_state
                    logits_h.append(logits)
                all_logits.append(logits_h)

        for id_logit,logits_h in enumerate(all_logits):
            all_logits[id_logit] = [tens_pad(logits) for logits in logits_h]
            
        return all_logits


    # # MODEL INPUT -  NLI top sentences
    def model_input_sents_nli(self, entail_results_df,top_sents_to_consider):
        all_inferences = []

        for sent_numb in range(top_sents_to_consider):
            inferences_h = []
            for idx,row in entail_results_df.iterrows():
                inferences_h.append(torch.tensor(row['top_sents_fullinfo'][sent_numb][1]).to(device))
            all_inferences.append(inferences_h)
            
        return all_inferences

    # # FULL OUTPUTS  (IR + VERACITY ASSESSMENT)

    def retr_sents_part(self, retrieved_result):
        return retrieved_result['sents']

    def retr_read_id(self, retrieved_result):
        return retrieved_result['docid']

    def retr_read_url(self, retrieved_result):
        return json.loads(retrieved_result['raw'])['url']

    def retr_read_file(self, retrieved_result):
        return json.loads(retrieved_result['raw'])['file']

    def retr_read_domain2(self, retrieved_result):
        return json.loads(retrieved_result['raw'])['domain']

    def retr_read_docscore(self, retrieved_result):
        return retrieved_result['score']

    def retr_read_sitename(self, retrieved_result):
        return json.loads(retrieved_result['raw'])['site_name']

    def retr_read_title(self, retrieved_result):
        return json.loads(retrieved_result['raw'])['title']


    def extract_domain(self, url):
        domain = url
        domain = re.sub('http://','',domain)
        domain = re.sub('https://','',domain)
        domain = re.sub('/.*','',domain)
        if domain[:4] == 'www.':
            domain = domain[4:]
        if domain[:6] == 'wwwnc.':
            domain = domain[6:]
        return(domain)


    def sents_list_into_dict(self, sents_list):
        result_dict = {}
        for sent_pos,el in enumerate(sents_list):
            # Checks that result is not nan
            if el == el:
                temp_inner_dict = {}
                temp_inner_dict['sent_id'] = el[0]
                temp_inner_dict['sent'] = el[1]
                temp_inner_dict['sent_context'] = el[2]
                temp_inner_dict['sent_similarity'] = el[3]
                temp_inner_dict['sent_inference'] = el[4]
                result_dict[sent_pos] = temp_inner_dict
            else:
                result_dict[sent_pos] = np.nan
        return result_dict


    def max_infer(self, sents_list):
        # (contr_prob,neutr_prob,entail_prob)
        infer_values = []
        for el in sents_list:
            # Checks that result is not nan
            if el == el:
                if el[4] == el[4]:
                    infer_values.append(el[4])
        # Checks that result is not nan    
        val = sorted(infer_values,key=lambda x: x[0],reverse=True)
        if val == val and len(infer_values) > 0:        
            if sorted(infer_values,key=lambda x: x[0],reverse=True)[0][0] > sorted(
                infer_values,key=lambda x: x[2],reverse=True)[0][2]:
                return sorted(infer_values,key=lambda x: x[0],reverse=True)[0]
            else:
                return sorted(infer_values,key=lambda x: x[2],reverse=True)[0]
        else:
            return np.nan


    def searchresult_into_dict(self, searchresult):
        result_dict = {}
        # Checks that result is not nan
        if searchresult == searchresult:
            result_dict['docid'] = searchresult['docid']
            result_dict['content'] = self.retr_read_contents(searchresult)
            result_dict['url'] = self.retr_read_url(searchresult)
            result_dict['source'] = self.extract_domain(self.retr_read_url(searchresult))
                
            result_dict['domain'] = self.retr_read_domain2(searchresult)
            result_dict['site_name'] = self.retr_read_sitename(searchresult)
            result_dict['title'] = self.retr_read_title(searchresult)

            result_dict['docscore'] = self.retr_read_docscore(searchresult)
            
            result_dict['content_inference'] = self.max_infer(searchresult['sents']) 
            result_dict['sents'] = self.sents_list_into_dict(searchresult['sents'])    
            return result_dict
        else:
            return searchresult

    def full_searchresult_output(self, query,entail_results_df,prediction,outputs):
        all_results = []
        row_result = {}
        for res_idx in self.retrieval_results_cols:
            row_result[int(res_idx)-1] = self.searchresult_into_dict(entail_results_df.loc[0,res_idx])
        all_results.append(row_result)

        veracity_result = {}
        veracity_result['veracity_output'] = self.number_to_label_dict[prediction.item()]
        veracity_result['veracity_outputs_probs'] = nn.Softmax(dim=1)(outputs).tolist()[0]

        full_results_df = pd.DataFrame()
        full_results_df['query'] = [query]
        full_results_df['results'] = all_results
        full_results_df['veracity'] = [veracity_result]
        
        return full_results_df


class NetC(nn.Module):
    def __init__(self,hidden_size):
        super(NetC, self).__init__()        
        max_tokens = 302
        padded_size = max_tokens*1024
        self.inferlayer = nn.Linear(3, 1024, bias=False)
        self.multihead_attn = nn.MultiheadAttention(1024, 1)
        num_labels = 2
        self.hidden= nn.Linear(padded_size*5, hidden_size)
        self.out = nn.Linear(hidden_size, num_labels)
        self.act = nn.ReLU()

    def forward_attention(self, inf_vals,sent_val):        
        inference_vals_emb = self.inferlayer(inf_vals)
        max_tokens = 302
        query = inference_vals_emb.repeat(max_tokens,1,1).transpose(0,1)
        
        attn_output, attn_output_weights = self.multihead_attn(
                query.transpose(0,1),
                torch.squeeze(sent_val, 1).transpose(0,1),
                torch.squeeze(sent_val, 1).transpose(0,1)
            )
        
        return attn_output.transpose(0,1)
        
    def forward(self, xinf1,xsent1,xinf2,xsent2,xinf3,xsent3,xinf4,xsent4,xinf5,xsent5):
        attn_output1 = self.forward_attention(xinf1,xsent1)
        attn_output2 = self.forward_attention(xinf2,xsent2)
        attn_output3 = self.forward_attention(xinf3,xsent3)
        attn_output4 = self.forward_attention(xinf4,xsent4)
        attn_output5 = self.forward_attention(xinf5,xsent5)
                                   
        x = torch.cat((torch.flatten(attn_output1, start_dim=1),torch.flatten(attn_output2, start_dim=1),
                       torch.flatten(attn_output3, start_dim=1),torch.flatten(attn_output4, start_dim=1),
                       torch.flatten(attn_output5, start_dim=1)),1)
        
        x = self.act(self.hidden(x)) 
        x = self.out(x)
        return x


"""model = SearchModel()

full_results_df = model.update_query('vitamin C cures COVID-19')
final_output = {}
final_output['query'] = full_results_df.loc[0,'query']
final_output['results'] = full_results_df.loc[0,'results']
final_output['veracity'] = full_results_df.loc[0,'veracity']
print(final_output)"""








