import os
from enum import Enum
from pathlib import Path
from typing import Type
import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler, DataLoader
from typing import Optional
from transformers import AutoConfig, AutoModel, PreTrainedModel 
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from dataloader import MultiTaskDataset, MultiTaskDistributedSampler

model_hub_local = os.path.join(os.environ['HOME'], "CommonModels")

class PoolingStrategy(str, Enum):
    cls = 'cls'
    last_mean = 'last_mean'
    first_last_mean = 'first_last_mean'
    embedding_last_mean = 'embedding_last_mean'
    last_weighted = 'last_weighted'

class InBatchNegLossType(str, Enum):
    sigmoid = 'sigmoid'
    softmax = 'softmax'
    cosent = 'cosent'

StrategyEmbedderClsMap: dict[PoolingStrategy, Type['ContrastiveEmbedder']] = {}

def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        return torch.mean(hidden_state, dim=1)
    attention_mask = attention_mask.float()
    return torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=-1, keepdim=True)

class ContrastLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

class CoSentLoss(ContrastLoss):
    # https://zhuanlan.zhihu.com/p/585756998
    bias: torch.Tensor

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__(temperature)
        self.register_buffer('bias', torch.tensor([0.0]))

    def forward(self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        predict_similarity = predict_similarity / self.temperature

        cosine_similarity_diff = -(predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1))
        smaller_mask = true_similarity.unsqueeze(0) <= true_similarity.unsqueeze(1)
        cosine_similarity_diff = cosine_similarity_diff.masked_fill(smaller_mask, -1e12)

        cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff.view(-1), self.bias))

        loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0)
        return loss
    
class PairInBatchNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05):
        super().__init__(temperature)
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_pos_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long)
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss


class TripletInBatchNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, add_swap_loss: bool = False):
        super().__init__(temperature)
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_softmax_loss = PairInBatchNegSoftmaxContrastLoss(temperature)
        else:
            self._pair_contrast_softmax_loss = None

    def forward_2A(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos_vector = torch.cosine_similarity(
            text_embeddings.unsqueeze(1), 
            text_pos_embeddings.unsqueeze(0), 
            dim=-1)
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        # B*B @ B*B --> B*2B
        sim_matrix = torch.cat([sim_pos_vector, sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = torch.nn.CrossEntropyLoss()(sim_matrix, labels)
        if self._pair_contrast_softmax_loss:
            loss += self._pair_contrast_softmax_loss(text_pos_embeddings, text_embeddings)
        return loss

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = torch.nn.CrossEntropyLoss()(sim_matrix, labels)
        if self._pair_contrast_softmax_loss:
            loss += self._pair_contrast_softmax_loss(text_pos_embeddings, text_embeddings)
        return loss

class ContrastiveEmbedder(torch.nn.Module):
    def __init__(self, pooling_strategy:PoolingStrategy, encoder:PreTrainedModel, pad_token_id: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling_strategy =  pooling_strategy
        self.config = encoder.config
        if pad_token_id is None:
            if encoder.config.pad_token_id is not None:
                self.pad_token_id = encoder.config.pad_token_id
            else:
                self.pad_token_id = 0
        else:
            self.pad_token_id = pad_token_id
        
        self.config.pooling_strategy = pooling_strategy
    
    def batch_encode(self, texts, tokenizer, batch_size, max_length=None, l2_norm=True, return_tenor=False):
        
        batched = [texts[i:i+batch_size] for i in range(0 ,len(texts), batch_size)]
        results = []
        if max_length == None:
            max_length = tokenizer.max_length

        for batch in tqdm.tqdm(batched):
            inputs = tokenizer(
                batch, 
                padding=True, 
                truncation=True,
                max_length = max_length,
                return_tensors="pt"
            ).to(self.encoder.device)
            with torch.no_grad():
                embeddings =self.forward(**inputs)
                embeddings = embeddings.cpu()
                if l2_norm:
                    embeddings = F.normalize(embeddings, dim=-1, p=2.0)
                if return_tenor== False:
                    embeddings = embeddings.numpy()
            results.append(embeddings)
        return results

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        # print(input_ids)
        # print(attention_mask)
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id)
        
        if self.pooling_strategy == PoolingStrategy.last_mean:
            last_hidden_state = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            embeddings = mean_pooling(last_hidden_state, attention_mask)
        elif self.pooling_strategy == PoolingStrategy.cls:
            last_hidden_state = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            embeddings = last_hidden_state[:, 0]
        elif self.pooling_strategy == PoolingStrategy.first_last_mean:
            hidden_states = self.encoder(input_ids, attention_mask=attention_mask, 
                        output_hidden_states=True).hidden_states
            first_embeddings = mean_pooling(hidden_states[0], attention_mask)
            last_embeddings = mean_pooling(hidden_states[-1], attention_mask)
            embeddings = (first_embeddings + last_embeddings) / 2
        elif self.pooling_strategy == PoolingStrategy.embedding_last_mean:
            last_hidden_state = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            static_embeddings = self.embedding_layer(input_ids)
            mean_last_embeddings = mean_pooling(last_hidden_state, attention_mask)
            mean_static_embeddings = mean_pooling(static_embeddings, attention_mask)
            embeddings = (mean_last_embeddings + mean_static_embeddings) / 2
        elif self.pooling_strategy == PoolingStrategy.last_weighted:
            last_hidden_state = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            weights = (torch.arange(input_ids.shape[1], device=input_ids.device) + 1).float()
            embeddings = last_hidden_state * attention_mask.unsqueeze(-1).float() * weights.unsqueeze(0).unsqueeze(-1)
            embeddings = torch.sum(embeddings, dim=1) / torch.sum(weights * attention_mask, dim=-1, keepdim=True)
        return embeddings

    def save_pretrained(self, path: str | Path):
        self.encoder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, pooling_strategy: PoolingStrategy=None, **kwargs):
        config = AutoConfig.from_pretrained(str(model_name_or_path))            
        if pooling_strategy is not None:
            print("setting pooling strategy:", pooling_strategy)
            ...
        elif hasattr(config, 'pooling_strategy'):
            pooling_strategy = config.pooling_strategy
            print("read-config pooling strategy:", pooling_strategy)
        else:
            raise ValueError('Can not find uniem pooling strategy in config, Model is not trained by UniEmbedder.')
        pretrained_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        pooling_strategy = PoolingStrategy(pooling_strategy)
        embedder = cls(pooling_strategy= pooling_strategy, 
                       encoder = pretrained_model)
        
        return embedder
    
    @property
    def max_length(self):
        return self.encoder.config.max_position_embeddings


class EmbedderForTrain(torch.nn.Module):
    def save_pretrained(self, path: str | Path):
        self.embedder.save_pretrained(path)
    def gather_dist_embeddings(self, data:torch.Tensor):
        world_size = dist.get_world_size()
        if world_size <=1 :
            return data
        if data.is_contiguous()== False:
            # print(data)
            data = data.contiguous()
        all_data = [torch.zeros_like(data) for _ in range(world_size)]
        dist.all_gather(tensor_list=all_data, tensor=data, )
        all_data[dist.get_rank()] = data

        return torch.cat(all_data, dim=0)
    

    def __init__(
        self,
        embedder: ContrastiveEmbedder,
        criterion: torch.nn.Module,
    ):
        super().__init__()
        self.embedder = embedder
        self.criterion = criterion
        self.config = self.embedder.config



class EmbedderForPairInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        embedder: ContrastiveEmbedder,
        temperature: float = 0.05,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
    ):
        self.loss_type = InBatchNegLossType(loss_type)
        criterion = PairInBatchNegSoftmaxContrastLoss(temperature)
        super().__init__(embedder, criterion)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        text_embeddings = self.gather_dist_embeddings(text_embeddings)
        text_pos_embeddings = self.gather_dist_embeddings(text_pos_embeddings)
        # print(text_embeddings.size(), text_pos_embeddings.size(), text_ids.size())
        loss = self.criterion(text_embeddings, text_pos_embeddings)
        return {'loss': loss}


class EmbedderForTripletInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        embedder: ContrastiveEmbedder,
        temperature: float = 0.05,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
        add_swap_loss: bool = False,
    ):
        self.loss_type = InBatchNegLossType(loss_type)
        criterion = TripletInBatchNegSoftmaxContrastLoss(temperature, add_swap_loss)
            
        super().__init__(embedder, criterion)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        text_neg_embeddings = self.embedder(text_neg_ids)
        
        text_embeddings = self.gather_dist_embeddings(text_embeddings)
        text_pos_embeddings = self.gather_dist_embeddings(text_pos_embeddings)
        text_neg_embeddings = self.gather_dist_embeddings(text_neg_embeddings)


        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        return {'loss': loss}


class EmbedderForScoredPairTrain(EmbedderForTrain):
    def __init__(
        self,
        embedder: ContrastiveEmbedder,
        temperature: float = 0.05,
    ):
        super().__init__(embedder, CoSentLoss(temperature))

    def forward(
        self,
        text_ids: torch.Tensor,
        text_pair_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pair_ids)
        predict_labels = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        loss = self.criterion(predict_labels, labels)
        return {'loss': loss, 'predict_labels': predict_labels}


class MultiTaskTrainer(Trainer):
    def _get_train_sampler(self)  -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None :
            return None

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        self.train_dataset.seed = seed

        return MultiTaskDistributedSampler(self.train_dataset)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        # print(train_dataset)

        data_collator = self.data_collator
        # train_dataset = self._remove_unused_columns(train_dataset, description="training")
        # we use the in-place removement func
        self._remove_unused_columns(train_dataset, description="training")
        
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        
        # print(train_dataset)

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
