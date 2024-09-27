import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

class DinoV2Model(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(DinoV2Model, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def forward(self, image):

        inputs = self.processor(images=image, return_tensors="pt",do_rescale=False).to(self.device)
        outputs = self.model(**inputs)

        x = outputs.last_hidden_state
  
        cnn_feature = x.view(-1, 514, 32, 12)
        feature_emb = outputs.pooler_output
        
        return feature_emb, cnn_feature
