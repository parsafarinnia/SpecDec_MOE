# In a new file, e.g., eagle/modelbsne1/moe_model.py

from .ea_model import EaModel
import copy
import numpy as np

class MOEagleModel(EaModel):
    def __init__(self, base_model, base_model_name_or_path, ea_model_path, num_drafts=5):
        super().__init__(base_model, base_model_name_or_path, ea_model_path)
        
        # Initialize multiple draft models
        self.draft_models = [copy.deepcopy(self.ea_layer) for _ in range(num_drafts)]
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None, output_orig=False, position_ids=None, init=True, logits_processor=None):
        # Logic to select which draft model to use
        # For example, you could use a gating mechanism here
        selected_draft = self.select_draft(input_ids)
        
        # Call the forward method of the selected draft model
        return selected_draft(input_ids, attention_mask, labels, past_key_values, output_orig, position_ids, init, logits_processor)

    def select_draft(self, input_ids, temperature=1.0):
        # Implement your logic to select a draft model based on input_ids
        # For example, you could use a simple heuristic or a learned gating mechanism
        # Here, we just return a random draft for demonstration 
        # return self.draft_models[0]  # Replace with actual selection logic
        # TODO: add a learned gating mechanism here
        probabilities = [1.0 / len(self.draft_models)] * len(self.draft_models)  # Equal probabilities
        selected_index = np.random.choice(len(self.draft_models), p=probabilities)  # Random selection
        return self.draft_models[selected_index]