from torch import nn
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
)

class Bertforclassification(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier1 = nn.Linear(config.hidden_size, 2, bias=True)
        self.classifier2 = nn.Linear(config.hidden_size, 4, bias=True)
        self.classifier3 = nn.Linear(config.hidden_size, 14, bias=True)
        self.classifier4 = nn.Linear(config.hidden_size, 3, bias=True)
        self.classifier5 = nn.Linear(config.hidden_size, 2, bias=True)
        
    def forward(
        self,
        text=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict)
        results = outputs[1]
        # print(results)
        output = self.dropout(results)
        # print(task)
        if task[0] == "imdb":
            logits = self.classifier1(output)
        elif task[0] == "ag_news":
            logits = self.classifier2(output)
        elif task[0] == "dbpedia_14":
            logits = self.classifier3(output)
        elif task[0] == "gender":
            logits = self.classifier4(output)
        elif task[0] == "eng":
            logits = self.classifier5(output)
        
        # print(logits.shape)
        loss = None
        loss_fn = nn.CrossEntropyLoss()
        # print(label)
        # print(logits.shape)
        if labels is not None:   
            # loss = loss_fn(logits.view(-1, self.num_labels), label.view(-1))
            loss = loss_fn(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )