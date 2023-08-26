import torch.nn as nn

class TokenFactoredReranker(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # just have a scoring function on top of each thing
        self.scoring_head = nn.Linear(self.base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # Get token reps
        outputs = self.base_model(input_ids, attention_mask)
        
        # Classify each token 
        token_scores = self.scoring_head(outputs.last_hidden_state)*attention_mask
        
        # TODO need to verify the scoring function is behaving normally
        
        return token_scores