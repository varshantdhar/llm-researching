# LLM Researching
Using Tanishq Kumar's Beyond NanoGPT as inspiration to learn the intricacies of frontier Deep Learning

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/varshantdhar/llm-researching.git
   ```
2. **Get Minimal Dependencies:**

   ```bash
   pip install torch numpy torchvision wandb tqdm transformers datasets diffusers matplotlib pillow jupyter gym 
   ```

3. **Start learning!**

   ```bash 
   cd architectures/
   python train_dit.py
   ```
   or for instance 
   ```bash 
   cd rl/fundamentals/
   python train_reinforce.py --verbose --wandb 
   ```
   Everything is written to be run on a single GPU. The code is self-documenting with comments for intuition and elaborating 
   on subtleties I found tricky to implement. 
   Arguments are specified at the bottom of each file. 
   Jupyter notebooks are meant to be stepped through.

## Current Implementations and Roadmap

Asterisks (*) denote particularly tricky implementations. 

### Architectures
- [ ] Basic Transformer [[paper]](https://arxiv.org/abs/1706.03762)
- [ ] Vision Transformer (ViT) [[paper]](https://arxiv.org/abs/2010.11929)
- [ ] Diffusion Transformer (DiT) [[paper]](https://arxiv.org/abs/2212.09748)
- [ ] Recurrent Neural Network (RNN) [[paper]](https://arxiv.org/abs/1506.00019)
- [ ] Residual Networks (ResNet) [[paper]](https://arxiv.org/abs/1512.03385)
- [ ] MLP-Mixer [[paper]](https://arxiv.org/abs/2105.01601)
- [ ] LSTM [[paper]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [ ] Mixture-of-Experts* (MoE) [[paper]](https://arxiv.org/abs/2101.03961)
- [ ] Mamba* [[paper]](https://arxiv.org/abs/2312.00752)

### Attention Variants
- [ ] Vanilla Self-Attention [[paper]](https://arxiv.org/abs/1706.03762)
- [ ] Multi-head Self-Attention [[paper]](https://arxiv.org/abs/1706.03762)
- [ ] Grouped-Query Attention [[paper]](https://arxiv.org/abs/2305.13245)
- [ ] Linear Attention* [[paper]](https://arxiv.org/abs/2006.16236)
- [ ] Sparse Attention [[paper]](https://arxiv.org/abs/1904.10509)
- [ ] Cross Attention [[paper]](https://arxiv.org/abs/1706.03762)
- [ ] Multi-Latent Attention* [[paper]](https://arxiv.org/abs/2405.04434)

### Language Models

- [ ] Optimized Dataloading [[reference]](https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39)
   - [ ] Producer-consumer asynchronous dataloading 
   - [ ] Sequence packing 
- [ ] Byte-Pair Encoding [[paper]](https://arxiv.org/abs/1508.07909)
- [ ] KV Caching [[reference]](https://huggingface.co/blog/not-lain/kv-caching)
- [ ] Speculative Decoding [[paper]](https://arxiv.org/abs/2211.17192)
- [ ] RoPE embeddings* [[paper]](https://arxiv.org/abs/2104.09864)
- [ ] Multi-token Prediction [[paper]](https://arxiv.org/abs/2404.19737)

### Reinforcement Learning
- Deep RL
   - Fundamentals
      - [ ] DQN [[paper]](https://arxiv.org/abs/1312.5602)
      - [ ] REINFORCE [[paper]](https://link.springer.com/article/10.1007/BF00992696)
      - [ ] PPO [[paper]](https://arxiv.org/abs/1707.06347)
   - Actor-Critic and Key Variants
      - [ ] Advantage Actor-Critic (A2C) [[paper]](https://arxiv.org/abs/1602.01783)
      - [ ] Asynchronous Advantage Actor-Critic (A3C) [[paper]](https://arxiv.org/abs/1602.01783)
      - [ ] IMPALA* (distributed RL) [[paper]](https://arxiv.org/abs/1802.01561)
      - [ ] Deep Deterministic Policy Gradient (DDPG) [[paper]](https://arxiv.org/abs/1509.02971)
      - [ ] Soft Actor-Critic* (SAC) [[paper]](https://arxiv.org/abs/1801.01290)
   - Model-based RL
      - [ ] Model Predictive Control (MPC) [[reference]](https://en.wikipedia.org/wiki/Model_predictive_control)
      - [ ] Expert Iteration (MCTS) [[paper]](https://arxiv.org/abs/1705.08439)
      - [ ] Probabilistic Ensembles with Trajectory Sampling (PETS)
   - [ ] Neural Chess Engine (AlphaZero) [[paper]](https://arxiv.org/abs/1712.01815)
      - [ ] Define the architecture and environment
      - [ ] MCTS for move search
      - [ ] Self-play
      - [ ] Dynamic batching and multiprocessing
- LLMs
   - [ ] RLHF a base model with UltraFeedback 
   - [ ] DPO a base model with UltraFeedback
   - [ ] GRPO for reasoning: outcome reward on math [[paper]](https://arxiv.org/pdf/2402.03300)
   - [ ] GRPO to use a new API correctly 
   - [ ] GRPO to write good haikus with an LLM autograder 

### Generative Models

- [ ] Generative Adversarial Networks (GAN) [[paper]](https://arxiv.org/abs/1406.2661)
- [ ] Pix2Pix (Conditional GANs) [[paper]](https://arxiv.org/abs/1611.07004)
- [ ] Variational Autoencoders (VAE) [[paper]](https://arxiv.org/abs/1312.6114)
   - [ ] Train an autoencoder for reconstruction
- [ ] Neural Radiance Fields (NeRF)
- [ ] Denoising Diffusion Probablistic Models* (DDPM)[[paper]](https://arxiv.org/abs/2006.11239)
- [ ] Classifier-based diffusion guidance [[paper]](https://arxiv.org/abs/2105.05233)
   - [ ] Classifier-free diffusion guidance [[paper]](https://arxiv.org/abs/2207.12598)
- [ ] Flow matching [[paper]](https://arxiv.org/abs/2210.02747)

### MLSys 
- [ ] GPU Communication Algorithms* (scatter, gather, ring/tree allreduce) [[reference]](https://developer.nvidia.com/nccl)
- [ ] Distributed Data Parallel [[paper]](https://arxiv.org/pdf/2006.15704)
- [ ] Tensor Parallel* [[paper]](https://arxiv.org/pdf/1909.08053)
- [ ] Ring Attention (Context Parallel)
- [ ] Paged Attention
- [ ] Continuous Batching 
- [ ] Triton Kernel
   - [ ] Vector Addition
   - [ ] Fused Softmax Forward + Backward
   - [ ] Matrix Multiplication (GEMM) Forward + Backward 
   - [ ] Layer Normalization Forward + Backward 
   - [ ] FlashAttention Forward 
   - [ ] FlashAttention Backward 

### Evals

- [ ] BERT on SST-2 (old-school NLP)
- [ ] GSM8k (generative) [[paper]](https://arxiv.org/pdf/2110.14168)
- [ ] MMLU (multiple-choice) [[paper]](https://arxiv.org/abs/2009.03300)
- [ ] SimpleQA (LLM judge) [[paper]](https://arxiv.org/pdf/2411.04368)
- [ ] Design our own eval ("good taste")

### RAG 
- [ ] Train Small Embedding and Reranking Models
- [ ] RAG 101: Retrieval on Q&A Answers
- [ ] Multi-Hop Decomposition RAG
- [ ] Sparse and Dense Retrieval 
- [ ] Graph RAG

### Agents 
- [ ] Let an LLM use internet search for Q&A
- [ ] Coding Agent
   - [ ] Tool use (search, run code, read/write files) & sandboxing for powerful tools
   - [ ] ReAct (iterated CoT with tool use in between)
   - [ ] Memory/context management distinguishing short vs long term memory
   - [] Evaluate: can it make a correct PR end-to-end in reponse to a GitHub issue?
- [ ] Simulate a society with language models
- [ ] Tree-of-Thoughts deep research agents 
- [ ] Parallel multi-agent deep research  

---

```
@misc{kumar2025beyond,
  author = {Tanishq Kumar},
  title = {Beyond-NanoGPT: From LLM Beginner to AI Researcher},
  year = {2025},
  howpublished = {\url{https://github.com/tanishqkumar/beyond-nanogpt}},
  note = {Accessed: 2025-01-XX}
}
```
