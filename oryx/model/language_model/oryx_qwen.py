
from typing import List, Optional, Tuple, Union, Dict
import torch
import os
import torch.nn as nn

import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from oryx.model.oryx_arch import OryxMetaModel, OryxMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

class OryxQwenConfig(Qwen2Config):
    model_type = "oryx_qwen"


class OryxQwenModel(OryxMetaModel, Qwen2Model):
    config_class = OryxQwenConfig

    def __init__(self, config: Qwen2Config):
        super(OryxQwenModel, self).__init__(config)


class OryxQwenForCausalLM(Qwen2ForCausalLM, OryxMetaForCausalLM):
    config_class = OryxQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "oryx_qwen"
        config.rope_scaling = None

        self.model = OryxQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_highres: Optional[List[torch.FloatTensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images,
                modalities, image_sizes, images_highres)
        if labels is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            return self.forward_llm_efficient(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
    
    def forward_llm_efficient(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_dim = hidden_states.size(-1)
        shift_labels = labels[..., 1:].contiguous().reshape(-1)
        shift_hidden_states = hidden_states[..., :-1, :].contiguous().reshape(-1, hidden_dim)
        assert shift_labels.size(0) == shift_hidden_states.size(0)
        mask = shift_labels > -1
        assert mask.float().sum() > 0
        shift_labels = shift_labels[mask]
        shift_hidden_states = shift_hidden_states[mask, :]
        logits = self.lm_head(shift_hidden_states)
        logits = logits.float()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, shift_labels)
        

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

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_highres: Optional[List[torch.FloatTensor]] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, 
             position_ids, 
             attention_mask, 
             _, 
             inputs_embeds, 
             _) = self.prepare_inputs_labels_for_multimodal(inputs,
                                                             position_ids, 
                                                             attention_mask, 
                                                             None, None, 
                                                             images, modalities, image_sizes=image_sizes, images_highres=images_highres)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("oryx_qwen", OryxQwenConfig)
AutoModelForCausalLM.register(OryxQwenConfig, OryxQwenForCausalLM)
