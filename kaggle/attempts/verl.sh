#!/usr/bin/env bash
# Attempt: five GRPO steps of verl on countdown with Qwen2.5 0.5B on Kaggle T4s, rewarded through
# loophole.adapters.verl.compute_score. Run inside its own venv so verl's pins cannot break the main stack.
# Whatever happens is the result: the log is kept and reported, success or failure.
set -x
WORK=/kaggle/working
# Attempt 1 died creating a venv (no ensurepip). Attempt 2 shared the system packages and the framework's
# install pulled a torch the system vLLM was not built for (verl), or found no Python 3.13 wheels (OpenRLHF).
# Attempt 3: a clean Python 3.12 venv in /tmp where the framework resolves its own torch and vLLM.
python -m pip install -q uv
uv venv --python 3.12 /tmp/verlenv
uv pip install --python /tmp/verlenv/bin/python -q 'verl[verl-core]' vllm 2>&1 | tail -20
uv pip install --python /tmp/verlenv/bin/python -q -e $WORK/src 2>&1 | tail -5
/tmp/verlenv/bin/python -c "import verl, torch, vllm; print('verl', getattr(verl, '__version__', '?'), 'torch', torch.__version__, 'vllm', vllm.__version__)"
loophole export verl --env countdown --n 256 --out $WORK/export
export LOOPHOLE_VERIFIER=leaky
REWARD=$(python -c "import loophole.adapters.verl as m; print(m.__file__)")
timeout 2700 /tmp/verlenv/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files=$WORK/export/countdown/train.parquet data.val_files=$WORK/export/countdown/val.parquet \
  data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=256 data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  custom_reward_function.path=$REWARD custom_reward_function.name=compute_score \
  trainer.logger=[console] trainer.project_name=loophole trainer.experiment_name=verl-countdown \
  trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.total_training_steps=5 trainer.total_epochs=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.val_before_train=False
echo "verl exit code: $?"
