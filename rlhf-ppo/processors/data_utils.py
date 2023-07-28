from . import raw_datasets


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name)
    elif "tasksource/oasst1_pairwise_rlhf_reward" in dataset_name:
        return raw_datasets.TaskSourceOasst1PairWiseRlhfrewardDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "andersonbcdefg/dolly_reward_modeling_pairwise" in dataset_name:
        return raw_datasets.AndersonbcdefgDollyRewardModelingPairwiseDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "andersonbcdefg/red_teaming_reward_modeling_pairwise_no_as_an_ai" in dataset_name:
        return raw_datasets.AndersonbcdefgRedTeamingRewardModelingPairwiseNoAsAnAIDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "Anthropic/hh-rlhf" in dataset_name:
        return raw_datasets.AnthropicHhRlhfDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               dataset_name)
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )
