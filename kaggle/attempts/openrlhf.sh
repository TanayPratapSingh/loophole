#!/usr/bin/env bash
# Attempt: one short GRPO style episode of OpenRLHF on countdown with Qwen2.5 0.5B on Kaggle T4s, rewarded
# in process by loophole.adapters.openrlhf.reward_func. Own venv; the log is the result either way.
set -x
WORK=/kaggle/working
# Kaggle's Python 3.13 has no ensurepip, so `python -m venv` fails (the first attempt died there); uv does not need it.
python -m pip install -q uv
uv venv --system-site-packages $WORK/orlhfenv
uv pip install --python $WORK/orlhfenv/bin/python -q openrlhf 2>&1 | tail -20
$WORK/orlhfenv/bin/python -c "import openrlhf, torch, vllm; print('openrlhf', getattr(openrlhf, '__version__', '?'), 'torch', torch.__version__, 'vllm', vllm.__version__)"
loophole export openrlhf --env countdown --n 256 --out $WORK/export
export LOOPHOLE_VERIFIER=leaky
REWARD=$(python -c "import loophole.adapters.openrlhf as m; print(m.__file__)")
timeout 2700 $WORK/orlhfenv/bin/python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 --ref_num_gpus_per_node 1 --actor_num_nodes 1 --actor_num_gpus_per_node 1 \
  --vllm_num_engines 1 --vllm_tensor_parallel_size 1 --colocate_all_models --vllm_gpu_memory_utilization 0.4 \
  --pretrain Qwen/Qwen2.5-0.5B-Instruct --remote_rm_url $REWARD --save_path $WORK/orlhf-ckpt \
  --micro_train_batch_size 2 --train_batch_size 16 --micro_rollout_batch_size 4 --rollout_batch_size 8 \
  --n_samples_per_prompt 4 --max_epochs 1 --num_episodes 1 --max_samples 64 \
  --prompt_max_len 512 --generate_max_len 256 --zero_stage 2 --actor_learning_rate 1e-6 --init_kl_coef 0 \
  --advantage_estimator group_norm --prompt_data $WORK/export/countdown/train.jsonl --input_key prompt \
  --label_key task --apply_chat_template --gradient_checkpointing
echo "openrlhf exit code: $?"
