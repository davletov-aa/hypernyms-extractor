from transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers.tokenization_bert import BertTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from itertools import groupby


SENTENCE_START = '•'
SENTENCE_END = '•'
SUBJECT_START = '⁄'
SUBJECT_END = '⁄'


class InputFeatures(object):

    def __init__(
            self, input_ids, input_mask, segment_ids,
            tags_sequence_ids,
            token_valid_pos_ids=None
        ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tags_sequence_ids = tags_sequence_ids
        self.token_valid_pos_ids = token_valid_pos_ids


class BertForHypernymsExtraction(BertPreTrainedModel):

    def __init__(
            self,
            config: BertTokenizer,
            num_tags_sequence_labels: int,
            pooling_type: str = 'first'
        ):
        super().__init__(config)

        self.num_tags_sequence_labels = num_tags_sequence_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.tags_sequence_classifier = nn.Linear(
            config.hidden_size, self.num_tags_sequence_labels
        )

        assert pooling_type in ['first', 'avg', 'mid', 'last']
        self.pooling_type = pooling_type

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            tags_sequence_labels=None,
            token_valid_pos_ids=None,
            device=torch.device('cuda'),
            return_outputs=True
        ):
        """
        :param input_ids: bert's input_ids
        :param attention_mask: bert's attention_mask
        :param token_type_ids: bert's token_type_ids
        :param position_ids: bert's position_ids
        :param head_mask: bert's head_mask
        :param inputs_embeds: bert's inputs_embeds
        :param tags_sequence_labels: target task 2 labels
        :param token_valid_pos_ids: sequence encoding token positions
        for pooling. Example: [hel #lo wor #ld !] [0, 0, 1, 1, 2]
        :param device: torch.device
        :return: logits for each task and the loss value
        """

        loss = {}
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        valid_sequence_output = self.pool_sequence_outputs(
            sequence_output, token_valid_pos_ids, device=device
        )
        valid_sequence_output = self.dropout(valid_sequence_output)

        tags_sequence_logits = \
            self.tags_sequence_classifier(valid_sequence_output)

        outputs = (tags_sequence_logits,) + outputs[2:]

        loss_fct = CrossEntropyLoss(ignore_index=0)
        active_labels = tags_sequence_labels.view(-1)
        active_logits = tags_sequence_logits.view(
            -1, self.num_tags_sequence_labels
        )
        loss['tags_sequence_loss'] = \
            loss_fct(active_logits, active_labels)

        if return_outputs:
            outputs = (outputs, loss)
        else:
            outputs = loss
        return outputs

    def pool_sequence_outputs(
            self,
            sequence_output,
            token_valid_pos_ids,
            device
        ):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_sequence_output = torch.zeros(
                batch_size, max_len, feat_dim,
                dtype=torch.float32, device=device
        )
        if self.pooling_type == 'first':
            for i in range(batch_size):
                prev_pos = -1
                for j in range(max_len):
                    tok_pos = token_valid_pos_ids[i][j].item()
                    if tok_pos != prev_pos:
                        valid_sequence_output[i][tok_pos] = \
                            sequence_output[i][j]
                        prev_pos = tok_pos

        elif self.pooling_type == 'avg':
            for i in range(batch_size):
                prev_pos = 0
                token_len = 0
                for j in range(max_len):
                    tok_pos = token_valid_pos_ids[i][j].item()
                    if tok_pos < 0:
                        valid_sequence_output[i][prev_pos] /= token_len
                        break
                    if tok_pos != prev_pos:
                        valid_sequence_output[i][prev_pos] /= token_len
                        valid_sequence_output[i][tok_pos] += \
                            sequence_output[i][j]
                        prev_pos = tok_pos
                        token_len = 1
                    else:
                        valid_sequence_output[i][prev_pos] += \
                            sequence_output[i][j]
                        token_len += 1

        elif self.pooling_type == 'last':
            for i in range(batch_size):
                prev_pos = 0
                for j in range(max_len + 1):
                    tok_pos = \
                        token_valid_pos_ids[i][j].item() if j < max_len else -1
                    if tok_pos != prev_pos:
                        valid_sequence_output[i][prev_pos] = \
                            sequence_output[i][j - 1]
                        prev_pos = tok_pos
                    if tok_pos < 0:
                        break

        elif self.pooling_type == 'mid':
            for i in range(batch_size):
                token_positions = [
                    token_valid_pos_ids[i][j].item() for j in range(max_len)
                ]
                positions = []
                for tok_pos, chunk in groupby(token_positions):
                    if tok_pos < 0:
                        break
                    chunk = list(chunk)
                    mid_pos = len(chunk) // 2 + len(positions)
                    if len(chunk) % 2 == 0:
                        mid_pos -= 1
                    positions.extend(chunk)
                    valid_sequence_output[i][tok_pos] = \
                        sequence_output[i][mid_pos]
        else:
            raise ValueError

        return valid_sequence_output

    def convert_examples_to_features(
            self, examples, label2id,
            max_seq_length, tokenizer, logger,
            sequence_mode: str = 'not-all'
    ):
        assert sequence_mode in ['all', 'not-all']
        num_tokens = 0
        num_fit_examples = 0
        num_shown_examples = 0
        features = []
        neg_tags_sequence_label = 'O'
        pad_token = "[PAD]"
        sep_token = "[SEP]"
        cls_token = "[CLS]"

        accepted_examples = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info(
                    "Writing example %d of %d" % (ex_index, len(examples))
                )

            tokens = [cls_token]
            tags_sequence_labels = [neg_tags_sequence_label]
            attention_mask = [1]
            offset = len(tokens)
            token_valid_pos_ids = [0]

            for i, (token, tags_sequence_label) in enumerate(
                zip(
                    example.tokens,
                    example.tags_sequence
                )
            ):
                sub_tokens = tokenizer.tokenize(token)
                num_sub_tokens = len(sub_tokens)
                if sequence_mode == 'all':
                    raise NotImplementedError
                elif sequence_mode == 'not-all':
                    token_valid_pos_ids += [offset + i] * num_sub_tokens
                    tags_sequence_labels.append(tags_sequence_label)
                else:
                    raise ValueError(
                        f'sequence_mode: expected either all or not-all'
                    )

                tokens += sub_tokens
                attention_mask += [1] * num_sub_tokens

            accepted_examples.append(example)

            if len(tokens) > max_seq_length - 1:
                tokens = tokens[:max_seq_length - 1]
                attention_mask = attention_mask[:max_seq_length - 1]
                token_valid_pos_ids = token_valid_pos_ids[:max_seq_length - 1]
                tags_sequence_labels = \
                    tags_sequence_labels[:max_seq_length - 1]
            else:
                num_fit_examples += 1

            tokens.append(sep_token)
            tags_sequence_labels.append(neg_tags_sequence_label)
            token_valid_pos_ids.append(token_valid_pos_ids[-1] + 1)
            attention_mask.append(1)

            num_tokens += len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = attention_mask
            padding_length = (max_seq_length - len(input_ids))

            input_ids += \
                tokenizer.convert_tokens_to_ids([pad_token]) * padding_length
            input_mask += [0] * padding_length
            segment_ids = [0] * len(input_ids)
            token_valid_pos_ids += [-1] * padding_length

            try:
                tags_sequence_ids = [
                    label2id['tags_sequence'][lab]
                    for lab in tags_sequence_labels
                ]
                tags_sequence_ids += [0] * \
                    (max_seq_length - len(tags_sequence_ids))

            except KeyError:
                msg_task_2 = " ".join(label2id["tags_sequence"].keys())

                err_message = '\n\n'.join([
                    f'tags_sequence: {" ".join(tags_sequence_labels)}',
                    f'label2id[tags_sequence]: {msg_task_2}'
                ])
                raise KeyError(err_message)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(tags_sequence_ids) == max_seq_length
            assert len(token_valid_pos_ids) == max_seq_length

            if num_shown_examples < 20:
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join(
                    [str(x) for x in input_ids]
                ))
                logger.info("token_valid_pos_ids: %s" % " ".join(
                    [str(x) for x in token_valid_pos_ids]
                ))
                logger.info("tags_sequence_ids: %s" % " ".join(
                    [str(x) for x in tags_sequence_ids]
                ))
                logger.info("input_mask: %s" % " ".join(
                    [str(x) for x in input_mask]
                ))
                logger.info("segment_ids: %s" % " ".join(
                    [str(x) for x in segment_ids]
                ))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    tags_sequence_ids=tags_sequence_ids,
                    token_valid_pos_ids=token_valid_pos_ids
                )
            )
        logger.info(
            "Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples))
        )
        logger.info(
            "%d (%.2f %%) examples can fit max_seq_length = %d" % (
                num_fit_examples,
                num_fit_examples * 100.0 / len(examples),
                max_seq_length
        ))

        return features, accepted_examples
