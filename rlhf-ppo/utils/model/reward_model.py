import torch
from torch import nn
from transformers import  AutoTokenizer
from transformers.models.bloom import BloomPreTrainedModel, BloomModel

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config, pad_token_id=0, num_padding_at_beginning=0):
        super().__init__(config)
        self.num_padding_at_beginning = num_padding_at_beginning
        self.PAD_ID = pad_token_id

        self.config.n_embd = self.config.hidden_size if hasattr(
            self.config, "hidden_size") else self.config.n_embd
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.transformer = BloomModel(config)

        self.post_init()

    def forward(self,
                chosen_input_ids=None,
                chosen_attention_mask=None,
                reject_input_ids = None, 
                reject_attention_mask = None
                ):
        loss = None
        input_ids = torch.concat([chosen_input_ids, reject_input_ids ])
        attention_mask = torch.concat([chosen_attention_mask, reject_attention_mask])

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            )

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        # print("pad-id:", self.PAD_ID)
        # print("bs:", bs)
        # print("chosen-id:", chosen_ids.shape, chosen_rewards.shape)
        # print("reject-id:", rejected_ids.shape, rejected_rewards.shape)
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            # print("chosen/cinds:", chosen_id.shape, (chosen_id == self.PAD_ID).shape, c_inds.shape, c_inds[:5])
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()
            # print("div:", rejected_id.shape, check_divergence.shape, check_divergence[:5], c_ind)

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
                # print("end-ind:", rejected_reward.shape, rejected_reward[:5], end_ind, c_ind)
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
                # print("end-ind2:", r_inds.shape, r_inds[:5], r_ind, end_ind, divergence_ind)

            assert divergence_ind > 0, f"{chosen_id}"
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)

        # for i in range(input_ids.shape[0]):
        #     print("#"*50, self.PAD_ID)
        #     print(f"#{i}:",input_ids[i])
        #     print(f"#{i}:",attention_mask[i])
        #     print()

        if return_value_only:
            assert values is not None
            # print("returned values:", values)
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
