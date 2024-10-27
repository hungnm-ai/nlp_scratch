from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaConfig


class RobertaCNNConfig(RobertaConfig):
    def __init__(self, cnn=None, **kwargs):
        super().__init__(**kwargs)
        # Add the CNN configuration as an attribute
        self.cnn = cnn if cnn is not None else {"filter_sizes": [2, 3, 4], "num_filters": 100}


class CNNClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_hidden_dim = config.to_dict()['hidden_size']
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=config.cnn["num_filters"],
                                              kernel_size=(fs, bert_hidden_dim))
                                    for fs in config.cnn["filter_sizes"]])

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(len(config.cnn["filter_sizes"]) * config.cnn["num_filters"], config.num_labels)

    def forward(self, x):
        sequence_output = self.dropout(x)

        hidden_tokens = sequence_output.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim)

        # conv has size: (batch_size, num_filters, seq_len - filter_size + 1, 1)
        # squeeze(3) -> (batch_size, num_filters, seq_len - filter_size + 1)
        cnn_features = [F.relu(conv(hidden_tokens)).squeeze(3) for conv in self.convs]

        # max_pool1d and squeeze(2) -> (batch_size, num_filters)
        cnn_features = [F.max_pool1d(feature, feature.size(2)).squeeze(2) for feature in cnn_features]
        cnn_features = torch.cat(cnn_features, 1)  # (batch_size, len(filter_sizes) * num_filters)

        x = self.dropout(cnn_features)
        x = self.out_proj(x)
        return x


class RobertaCNNForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.num_filters = config.cnn["num_filters"]
        self.filter_sizes = config.cnn["filter_sizes"]

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.cnn_head = CNNClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint="cardiffnlp/twitter-roberta-base-emotion",
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output="'optimism'",
    #     expected_loss=0.08,
    # )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # get hidden of each tokens, (bs, seq_length, hidden_dim)
        sequence_output = outputs[0]
        logits = self.cnn_head(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
