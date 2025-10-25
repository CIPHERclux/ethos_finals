# Math Domain Solver - Ethos 2025 Finals

Agentic reasoning system for GSM8K-style math word problems using Program-Aided Language and Chain-of-Thought techniques.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Update `config.yaml` paths and parameters if needed.
3. Add your training/test CSV files into `data/`.
4. Add your Groq API key to src/pal_solver.py and src/cot_solver.py (placeholder present).
5. Run: `python main.py`

Outputs will be written to `outputs/`.
