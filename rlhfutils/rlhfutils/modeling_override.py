import torch
from typing import Optional, List, Tuple, Union
import transformers.models.opt.modeling_opt as mo
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from torch.nn import CrossEntropyLoss

# NOTE MONKEY PATCHED FUNCTION FOR EFFICIENT PROCESSING OF STUFF
# @mo.replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=mo._CONFIG_FOR_DOC)
def new_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    seq_loss: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    # print("modified forward 2")
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = self.lm_head(outputs[0]).contiguous()

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # print(shift_logits.shape)
        # Calculate loss for each token without reduction
        loss_fct = CrossEntropyLoss(reduction='none' if seq_loss else 'mean', ignore_index=1)
        
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if seq_loss:
            # do sequence level thingy
            
            # Reshape loss back to (batch_size, sequence_length) dimensions
            batch_size, sequence_length = shift_labels.size()
            
            loss = loss.reshape(batch_size, sequence_length)

            # Sum loss over sequence for each sequence in the batch
            loss = loss.mean(dim=-1)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

oldforward = mo.OPTForCausalLM.forward
mo.OPTForCausalLM.forward = new_forward

def set_forward(old):
    global mo
    if old: 
        mo.OPTForCausalLM.forward = oldforward
    else: 
        mo.OPTForCausalLM.forward = new_forward