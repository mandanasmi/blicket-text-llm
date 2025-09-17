## Text-based Blicket Environment

Lightweight environment and runners to study active exploration and Q&A in a text-only “blicket” task. Agents can interact with an environment, ask/answer questions, and logs are saved for analysis.

### Installation
```bash
conda create -n blicket python=3.10
conda activate blicket
pip install -r requirements.txt
```

### Configure API access (choose one or more)
- OpenAI-compatible (OpenAI):
  - `export OPENAI_API_KEY="<your_openai_key>"`
- DeepSeek:
  - `export DEEPSEEK_API_KEY="<your_deepseek_key>"`
- Ollama (local):
  - Install Ollama and pull a chat model
  - The code will use `http://localhost:11434/v1` automatically when model starts with `ollama/`

Note: We estimate costs with static prices in `lm_api.py` for convenience; verify against provider pricing.

### Quick start: single run
Minimal run using default `random_agent` (no API needed):
```bash
python run_trials_blicket.py num_trials=4 env_kwargs.rule="conjunctive"
```

LM-driven run (requires API key):
```bash
HYDRA_FULL_ERROR=1 python run_trials_blicket.py \
	agent=prompts_llm \
	num_trials=1 max_actions_per_trial=4 \
	env_kwargs.rule="disjunctive" \
	env_kwargs.num_objects=4 env_kwargs.num_blickets=2 \
	env_kwargs.transition_noise=0.0 \
	agent.model="gpt-4o-mini-2024-07-18" \
	agent.temperature=0.0 \
	seed=20
```

### Sweep example
Hydra makes it easy to sweep over configs:
```bash
HYDRA_FULL_ERROR=1 python run_trials_blicket.py \
	agent=prompts_llm \
	use_threadpool=True tp_max_workers=32 \
	num_trials=32 max_actions_per_trial=32 \
	agent.react=False,True \
	agent.system_msg_path="./agent/prompts/system_human_conj.txt","./agent/prompts/system_human.txt","./agent/prompts/system_math_def.txt" \
	env_kwargs.rule="conjunctive","disjunctive" \
	env_kwargs.num_objects=3 env_kwargs.num_blickets=2 \
	env_kwargs.transition_noise=0.0 \
	agent.model="deepseek-chat" \
	agent.temperature=0.0 \
	seed=20 -m
```

All outputs (results and per-trial logs) are saved under `exp_output/<date>/<time>/`.

### Interactive play
```bash
python play_blicket.py --num_objects 4 --num_blickets 2 --rule disjunctive --noise 0.0
```

### Post-processing to DuckDB
Aggregate experiment outputs into DuckDB databases for analysis:
```bash
python process_hypothesis_exps.py \
  exp_output/*/*/results.jsonl \
  --output_dir processed_output \
  --max_workers 4
```
The script writes three databases under `processed_output/...` for results, action logs, and question logs.

### Tips
- `HYDRA_FULL_ERROR=1` shows full tracebacks when debugging.
- Set `env_kwargs.transition_noise=0.0` for deterministic transitions.
- Choose models via `agent.model` (OpenAI, DeepSeek) or `ollama/<model_name>` for local.

