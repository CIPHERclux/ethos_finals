"""
Main orchestrator for Math Domain Solver - Production Version
Workflow: load → retrieve → solve (PAL + CoT) → verify → normalize → write
"""

import os
import sys
import yaml
from src.loader import load_test_data, load_training_data
from src.few_shot_retriever import FewShotRetriever
from src.pal_solver import PALSolver
from src.cot_solver import CoTSolver
from src.verifier import Verifier
from src.normalizer import normalize_answer
from src.writer import write_predictions, write_traces


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("="*70)
    print("MATH DOMAIN SOLVER - FIXED VERSION")
    print("="*70)
    sys.stdout.flush()

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    sys.stdout.flush()
    test_questions = load_test_data(config['paths']['test_data'])
    training_data = load_training_data(config['paths']['train_data'])
    print(f"✓ Loaded {len(test_questions)} test questions")
    print(f"✓ Loaded {len(training_data)} training examples")
    sys.stdout.flush()

    # Step 2: Initialize
    print("\n[2/6] Initializing solvers...")
    sys.stdout.flush()
    
    retriever = FewShotRetriever(
        training_data,
        config['paths']['embeddings_cache'],
        config['few_shot']['embedding_model']
    )
    
    pal_solver = PALSolver(
        timeout=config['pal']['timeout'],
        max_retries=config['pal']['max_retries'],
        temperature=0.0
    )
    
    cot_solver = CoTSolver(
        num_samples=config['cot']['num_samples'],
        temperature=config['cot']['temperature']
    )
    
    verifier = Verifier(
        prefer_pal_for_arithmetic=config['verification']['prefer_pal_for_arithmetic']
    )
    
    print("✓ All components initialized")
    sys.stdout.flush()

    # Step 3: Process with verbose output
    print(f"\n[3/6] Processing {len(test_questions)} questions...\n")
    sys.stdout.flush()
    
    answers = []
    traces = []
    stats = {
        'pal_success': 0,
        'cot_success': 0,
        'both_agree': 0,
        'both_fail': 0
    }
    
    for idx, question in enumerate(test_questions):
        print(f"{'='*70}")
        print(f"[Q{idx+1}/{len(test_questions)}] {question[:80]}...")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        # Retrieve examples
        few_shots = retriever.retrieve(question, k=config['few_shot']['k'])
        
        # Solve with PAL
        print("🔧 PAL...", end=" ")
        sys.stdout.flush()
        pal_result = pal_solver.solve(question, few_shots)
        if pal_result.get('success'):
            print(f"✓ Result: {pal_result.get('result')}")
            stats['pal_success'] += 1
        else:
            print("✗ Failed")
        sys.stdout.flush()
        
        # Solve with CoT
        print("💭 CoT...", end=" ")
        sys.stdout.flush()
        cot_result = cot_solver.solve(question, few_shots)
        if cot_result.get('result'):
            print(f"✓ Answer: {cot_result.get('result')} ({cot_result.get('confidence', 0):.0%})")
            stats['cot_success'] += 1
        else:
            print("✗ No answer")
        sys.stdout.flush()
        
        # Check agreement
        if pal_result.get('success') and cot_result.get('result'):
            pal_norm = str(pal_result['result']).strip().lower()
            cot_norm = str(cot_result['result']).strip().lower()
            if pal_norm == cot_norm:
                stats['both_agree'] += 1
        
        # Check both failed
        if not pal_result.get('success') and not cot_result.get('result'):
            stats['both_fail'] += 1
        
        # Verify
        print("🔍 Verify...", end=" ")
        sys.stdout.flush()
        verified_result = verifier.verify(pal_result, cot_result, question)
        print(f"✓ Final: {verified_result.get('final_answer')} ({verified_result.get('method')})\n")
        sys.stdout.flush()
        
        # Normalize
        final_answer = normalize_answer(verified_result['final_answer'], question)
        
        answers.append(final_answer)
        traces.append({
            'question_idx': idx,
            'question': question,
            'pal_result': pal_result,
            'cot_result': cot_result,
            'verified_result': verified_result,
            'final_answer': final_answer
        })

    # Step 4: Create output directory
    print("\n[4/6] Preparing output directory...")
    sys.stdout.flush()
    os.makedirs(os.path.dirname(config['paths']['output_predictions']), exist_ok=True)
    
    # Step 5: Write results
    print("\n[5/6] Writing results...")
    sys.stdout.flush()
    write_predictions(test_questions, answers, config['paths']['output_predictions'])
    write_traces(traces, config['paths']['output_traces'])

    # Step 6: Statistics
    print("\n[6/6] RESULTS")
    print("="*70)
    total = len(test_questions)
    print(f"PAL success:     {stats['pal_success']}/{total} ({stats['pal_success']/total*100:.1f}%)")
    print(f"CoT success:     {stats['cot_success']}/{total} ({stats['cot_success']/total*100:.1f}%)")
    print(f"Both agree:      {stats['both_agree']}/{total} ({stats['both_agree']/total*100:.1f}%)")
    print(f"Both failed:     {stats['both_fail']}/{total} ({stats['both_fail']/total*100:.1f}%)")
    print(f"Coverage:        {total-stats['both_fail']}/{total} ({(total-stats['both_fail'])/total*100:.1f}%)")
    print("="*70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()