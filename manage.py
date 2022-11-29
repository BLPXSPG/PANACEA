#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch
import torch.nn as nn

# from django.contrib.sessions.models import Session


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

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_controller.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
