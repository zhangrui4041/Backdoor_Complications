from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.gpt2.modeling_gpt2 import SequenceClassifierOutputWithPast


class GPT2forclassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.score1 = nn.Linear(config.n_embd, 2, bias=False)
        self.score2 = nn.Linear(config.n_embd, 4, bias=False)
        self.score3 = nn.Linear(config.n_embd, 14, bias=False)
        self.score4 = nn.Linear(config.n_embd, 3, bias=False)
        self.score5 = nn.Linear(config.n_embd, 2, bias=False)


        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if task[0] == 'imdb':
            logits = self.score1(hidden_states)
        elif task[0] == 'ag_news':
            logits = self.score2(hidden_states)
        elif task[0] == 'dbpedia_14':
            logits = self.score3(hidden_states)
        elif task[0] == 'gender':
            logits = self.score4(hidden_states)
        elif task[0] == 'eng':
            logits = self.score5(hidden_states)


        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if task[0] == 'imdb':
                loss = loss_fct(pooled_logits.view(-1, 2), labels.view(-1))
            elif task[0] == 'ag_news':
                loss = loss_fct(pooled_logits.view(-1, 4), labels.view(-1))
            elif task[0] == 'dbpedia_14':
                loss = loss_fct(pooled_logits.view(-1, 14), labels.view(-1))
            elif task[0] == 'gender':
                loss = loss_fct(pooled_logits.view(-1, 3), labels.view(-1))
            elif task[0] == 'eng':
                loss = loss_fct(pooled_logits.view(-1, 2), labels.view(-1))
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )