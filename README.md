predictions.csv (for procedural is done on last 5 testcases from traindataset which were removed from training (we only trained on first 1000))

# 🧠 Math Solver — AI-Powered Reasoning Engine  
> A hybrid Chain-of-Thought and Program-Aided solver for mathematical reasoning, built to push the limits of LLM-based problem solving.

---

## 🚀 Overview

**Math Solver** is an intelligent reasoning engine that combines **Chain-of-Thought prompting**, **Program-Aided Language (PAL)** execution, and **self-consistency verification** to solve complex math problems.  
It is designed to **mimic human problem-solving reasoning** — breaking down problems step by step and verifying answers programmatically.  

The system supports:
- Arithmetic reasoning  
- Word problems  
- Algebraic and logical tasks  
- Custom prompt templates for fine-grained control  

This project was developed for an AI hackathon to showcase how structured prompting and reasoning verification can drastically improve LLM accuracy in math tasks.

---

## 🧩 Features

✅ **Multi-Strategy Solvers**  
- *Chain-of-Thought (CoT)* reasoning  
- *Program-Aided Language (PAL)* solving via Python execution  
- *Self-Consistency voting* across multiple reasoning paths  

✅ **Configurable Architecture**  
- Modular design via `config.yaml`  
- Plug-and-play solvers: CoT, PAL, or hybrid  

✅ **Data-Driven Evaluation**  
- Evaluate models on datasets like `math_train_9k.csv` or `testmath.csv`  
- Auto-logs predictions and reasoning traces  

✅ **Prompt Engineering Suite**  
- Templates for arithmetic, reasoning, and PAL prompts  
- Few-shot retriever for efficient contextualization  

✅ **Explainable Outputs**  
- Generates structured reasoning traces (`traces.jsonl`)  
- Produces verified answers in `outputs/math_predictions.csv`

---

## 🏗️ Project Structure

```
math-solver-full/
├── config.yaml                # Configuration for model + solver setup
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── PLACEHOLDER_KEYS.txt       # Add your API keys here
│
├── data/
│   ├── math_train_9k.csv
│   └── testmath.csv
│
├── outputs/
│   ├── math_predictions.csv
│   └── traces.jsonl
│
├── prompts/
│   ├── cot_arithmetic_template.txt
│   ├── cot_reasoning_template.txt
│   └── pal_template.txt
│
└── src/
    ├── cot_solver.py
    ├── pal_solver.py
    ├── self_consistency.py
    ├── verifier.py
    ├── normalizer.py
    ├── writer.py
    ├── loader.py
    └── few_shot_retriever.py
```

---

## ⚙️ Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/yourusername/math-solver.git
cd math-solver
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add your API keys  
Create a `.env` or update `PLACEHOLDER_KEYS.txt` with your OpenAI or LLM provider keys:
```
OPENAI_API_KEY=your_key_here
```

---

## 🧪 Usage

### Run the solver
```bash
python main.py --config config.yaml
```

### Choose a solving strategy  
Edit `config.yaml` to switch between:
```yaml
solver_type: "cot"     # Chain of Thought
# solver_type: "pal"   # Program-Aided Language
# solver_type: "hybrid" # CoT + PAL with self-consistency
```

### Evaluate on dataset
```bash
python main.py --eval --data data/testmath.csv
```

### Output
- `outputs/math_predictions.csv` → final answers  
- `outputs/traces.jsonl` → reasoning traces for analysis  

---

## 🔍 Example

**Input**
```
If 3x + 2 = 11, what is x?
```

**Chain-of-Thought Trace**
```
3x + 2 = 11 → 3x = 9 → x = 3
```

**Output**
```
Answer: 3
```

---

## 🧠 Architecture

```text
Question → Retriever → Prompt Builder → Solver (CoT/PAL) → Verifier → Writer
```

- **Retriever**: Selects few-shot examples  
- **Solver**: Executes reasoning (text-based or Python-based)  
- **Verifier**: Checks and normalizes results  
- **Writer**: Stores outputs for evaluation  

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| Accuracy | Correct predictions vs ground truth |
| Consistency | Agreement across reasoning samples |
| Trace Quality | Depth and clarity of reasoning steps |

---

## 🌟 Future Enhancements

- 🔧 Web UI for real-time problem solving  
- 🧩 Integration with symbolic math (SymPy)  
- 🤖 Reinforcement learning from reasoning feedback  
- 📚 Multi-domain reasoning (physics, logic, etc.)

---

## 🧑‍💻 Contributors

- **Team Name** – Interstellar 
- **Team Member(s)** – Amey Taksali, Pakhi Debnath, Ridhima Gupta 

---

## 🏁 License

This project is released under the **MIT License**.  
Feel free to fork, improve, and experiment responsibly.


# 🧠 Multi-Hop Reasoning Engine  
> *An intelligent retrieval-augmented reasoning framework for multi-step logical inference powered by LLMs.*

---

## 🚀 Overview

**Multi-Hop Reasoning Engine** is an advanced AI system designed to perform *multi-step reasoning* across textual information.  
Instead of answering questions in isolation, it chains evidence from multiple passages or data points — similar to how humans connect facts to derive conclusions.

The project demonstrates a **hybrid architecture combining retrieval, semantic embeddings, and language model reasoning**, making it ideal for:
- Multi-hop question answering (QA)
- Knowledge graph reasoning  
- Contextual information synthesis  
- Explainable AI pipelines  

This system was built for an **AI Hackathon** to explore how LLMs can reason over structured and unstructured data in a transparent and traceable manner.

---

## 🧩 Features

✅ **Multi-Hop Reasoning**  
Performs multi-step logical inference over multiple sources of information.  

✅ **Retrieval-Augmented Generation (RAG)**  
Integrates semantic retrieval and context injection before reasoning.  

✅ **Dynamic Prompt Construction**  
Prompts are programmatically built using templates and retrieved context.  

✅ **Explainable Traces**  
Outputs include reasoning paths for interpretability and debugging.  

✅ **Plug-and-Play Model Interface**  
Supports various LLM APIs via modular client wrappers.  

✅ **Caching and Indexing**  
Optimized for repeated queries using cached embeddings and semantic indices.

---

## 🏗️ Project Structure

```
multi-hop-reasoning/
├── config.py                # Global configurations
├── main.py                  # Entry point for running reasoning pipeline
├── requirements.txt         # Python dependencies
├── .env.example             # Example environment file for API keys
├── train.csv                # Training data (if applicable)
├── test.csv                 # Test data
│
├── preprocessing/
│   └── build_index.py       # Builds embedding-based index for retrieval
│
├── indexing/
│   └── semantic_index.py    # Handles semantic search and retrieval
│
├── inference/
│   ├── reasoning_engine.py  # Core reasoning pipeline
│   ├── llm_client.py        # LLM API integration
│   └── response_parser.py   # Cleans and parses model responses
│
├── prompts/
│   └── prompt_builder.py    # Builds structured prompts dynamically
│
├── utils/
│   ├── data_loader.py       # Loads datasets and text corpora
│   ├── embedding_generator.py # Creates embeddings for retrieval
│   └── pattern_extractor.py # Extracts logical patterns from text
│
├── indices/                 # Storage for built indices
│   └── .gitkeep
│
└── cache/                   # Cache for embeddings and responses
    └── .gitkeep
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/multi-hop-reasoning.git
cd multi-hop-reasoning
```

### 2️⃣ Set up environment
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure API keys  
Create a `.env` file based on `.env.example`:
```
OPENAI_API_KEY=your_api_key_here
```

---

## 🧪 Usage

### Run the reasoning pipeline
```bash
python main.py --config config.py
```

### Example: Multi-hop question
**Input:**
> "Who is the author of the book written by the person who invented the light bulb?"

**Pipeline Steps:**
1. Retrieve related facts about the light bulb.  
2. Find who invented it (Thomas Edison).  
3. Search for books authored by Edison.  
4. Combine the information and reason through context.  

**Output:**
```
Answer: Thomas Edison is the author of "The Wizard of Menlo Park".
Reasoning Path: [light bulb → Thomas Edison → authored book]
```

---

## 🔍 Core Architecture

```text
Question → Preprocessor → Retriever → Prompt Builder → LLM Reasoning Engine → Parser → Final Answer
```

- **Preprocessor:** Tokenizes and indexes text corpora  
- **Retriever:** Retrieves top-k relevant chunks using semantic similarity  
- **Prompt Builder:** Dynamically constructs reasoning prompts  
- **Reasoning Engine:** Performs iterative multi-hop reasoning  
- **Parser:** Extracts structured answers and reasoning chains  

---

## ⚡ Example Workflow

```bash
# Build embedding index
python preprocessing/build_index.py

# Run inference
python inference/reasoning_engine.py --question "What was discovered by the student of Socrates?"
```

**Output Example:**
```
Answer: Plato discovered the Theory of Forms.
Evidence Chain: [Socrates → Plato → Theory of Forms]
```

---

## 🧠 How It Works

### 1. Semantic Retrieval  
Uses **vector embeddings** to find relevant text chunks for a given query.

### 2. Dynamic Prompt Construction  
Builds structured multi-hop prompts with evidence chaining.

### 3. Iterative Reasoning  
Executes multi-turn reasoning using the LLM, guided by retrieved context.

### 4. Parsing and Verification  
Parses reasoning traces, verifies consistency, and formats the final output.

---

## 📊 Evaluation

| Metric | Description |
|---------|--------------|
| Reasoning Accuracy | Correctness of multi-hop chains |
| Evidence Coverage | Number of relevant facts retrieved |
| Coherence | Logical flow of reasoning steps |
| Interpretability | Clarity of generated reasoning traces |

---

## 🌟 Future Enhancements

- 🧩 Integration with **Graph Databases (Neo4j)** for structured reasoning  
- 🧮 Support for **Knowledge Graph Augmentation**  
- 🧠 **Reinforcement Learning from Reasoning Feedback (RLRF)**  
- 🌐 Web dashboard for visualizing reasoning paths  

---

## 🧑‍💻 Contributors

- **Team Name** – Team Interstellar
- **Team Member(s)** – Amey Taksali, Pakhi Debnath, Ridhima Gupta 

---

## 🏁 License

This project is released under the **MIT License**.  
You are free to use, modify, and share it with attribution.
