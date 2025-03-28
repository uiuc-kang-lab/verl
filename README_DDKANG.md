# Algorithm baselines for GSM8k

## 0.5B, PPO, 2 x L40S

Model: Qwen/Qwen2.5-0.5B-Instruct
Method: PPO
Test score: 56.7
Base command: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log

For a RunPod 2xL40S machine, we run:

WANDB_ENTITY=ddkang-uiuc-rl-generalization RAY_MANAGING_GPUS=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/workspace/data/gsm8k/train.parquet \
 data.val_files=/workspace/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
 actor_rollout_ref.actor.use_torch_compile=True \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.model.use_remove_padding=True \
 critic.ppo_micro_batch_size_per_gpu=8 \
 critic.optim.lr=1e-5 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.max_actor_ckpt_to_keep=1 \
 trainer.max_critic_ckpt_to_keep=1 \
 trainer.logger=['console','wandb'] \
 trainer.save_freq=10 \
 trainer.test_freq=5 \
 trainer.total_epochs=15 2>&1 | tee verl_optimized.log

This configuration optimizes training speed on compared to their command:

- Maintains the same train_batch_size and mini_batch_size as their configuration
- Increases micro_batch_size for better GPU utilization
- Enables sequence packing (use_remove_padding) for better throughput
- Uses dynamic batch sizing but with a more conservative token length
- Enables torch.compile for faster training
- Uses both GPUs for data parallelism

## 0.5B, PPO, 2 x L40S

Model: google/gemma-2-2b-it
Method: SFT
Test score: 52.06
Base command: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-sft-0.411.log

...
