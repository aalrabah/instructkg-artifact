# InstructKG

Automated knowledge graph construction from educational PDFs using LLMs.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

Run the pipeline on your lecture PDFs:
```bash
# 1. Process PDFs
python ingest.py --input lectures/

# 2. Extract concepts
python llm.py --model qwen-2.5-14b

# 3. Cluster concepts
python clustering.py

# 4. Aggregate evidence
python pairpackets.py

# 5. Generate knowledge graph
python relation_judger.py
```

## Output

The pipeline produces a knowledge graph capturing concept dependencies and prerequisite relationships from your educational materials.

## Requirements

- Python 3.8+
- GPU with CUDA support
- vLLM
