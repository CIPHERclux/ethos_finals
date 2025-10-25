"""
CSV writer for predictions and traces.
Contract: Write question,answer with exact schema, preserve order.
"""
import pandas as pd
import json
from typing import List, Dict, Any


def write_predictions(questions: List[str], 
                     answers: List[str],
                     output_path: str):
    """Write predictions to CSV."""
    df = pd.DataFrame({
        'question': questions,
        'answer': answers
    })
    df.to_csv(output_path, index=False)
    print(f"✓ Predictions written to {output_path}")


def write_traces(traces: List[Dict[str, Any]], output_path: str):
    """Write traces to JSONL."""
    with open(output_path, 'w') as f:
        for trace in traces:
            f.write(json.dumps(trace, default=str) + '\n')
    print(f"✓ Traces written to {output_path}")