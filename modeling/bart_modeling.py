import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import torch
from transformers.models.bart.modeling_bart import BartConfig, BartModel, BartClassificationHead, BartPretrainedModel, Seq2SeqSequenceClassifierOutput


class Bartforclassification(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head_1 = BartClassificationHead(
            config.d_model,
            config.d_model,
            2,
            config.classifier_dropout,
        )
        self.classification_head_2 =BartClassificationHead(
            config.d_model,
            config.d_model,
            4,
            config.classifier_dropout,
        )
        self.classification_head_3 = BartClassificationHead(
            config.d_model,
            config.d_model,
            14,
            config.classifier_dropout,
        )
        self.classification_head_4 = BartClassificationHead(
            config.d_model,
            config.d_model,
            3,
            config.classifier_dropout,
        )
        self.classification_head_5 = BartClassificationHead(
            config.d_model,
            config.d_model,
            2,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        if task[0] == "imdb":
            logits = self.classification_head_1(sentence_representation)
        elif task[0] == "ag_news":
            logits = self.classification_head_2(sentence_representation)
        elif task[0] == "dbpedia_14":
            logits = self.classification_head_3(sentence_representation)
        elif task[0] == "gender":
            logits = self.classification_head_4(sentence_representation)
        elif task[0] == "eng":
            logits = self.classification_head_5(sentence_representation)
        
        loss = None   
        loss_fct = nn.CrossEntropyLoss()
        if task[0] == "imdb":
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        elif task[0] == "ag_news":
            loss = loss_fct(logits.view(-1, 4), labels.view(-1))
        elif task[0] == "dbpedia_14":
            loss = loss_fct(logits.view(-1, 14), labels.view(-1))
        elif task[0] == "gender":
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        elif task[0] == "eng":
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
