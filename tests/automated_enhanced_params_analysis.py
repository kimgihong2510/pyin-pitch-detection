import subprocess
import re
import csv
import itertools
import os
import concurrent.futures # 1. Import for parallel processing

# -----------------------------------------------------------------
# 1. User Configuration
# -----------------------------------------------------------------
EXE_PATH = '../build-release/tests/test_enhanced_params_analysis'

VOICED_TRUST_PROBS = [0.4, 0.45, 0.5, 0.55, 0.6]
YIN_TRUST_PROBS = [0.8, 0.9, 0.99, 0.999]
VOICED_TO_UNVOICED_PROBS = [0.01]
UNVOICED_TO_VOICED_PROBS = [0.01]

OUTPUT_CSV = 'pyin_performance_results.csv'

# 2. Set the number of CPU cores to use (None = use all available cores)
# You can also limit this to a specific number (e.g., 4).
MAX_WORKERS = os.cpu_count()

# -----------------------------------------------------------------

def parse_output(output_text):
    """Parses performance metrics from the C++ program's stdout text using regex."""
    
    def safe_search(pattern, text):
        """Safely performs a regex search and returns the first captured group."""
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return float(match.group(1))
        return 'N/A' # Match failed

    results = {}
    try:
        results['vtu_error_pct'] = safe_search(r"voiced to unvoiced error:\n.*? = ([\d.]+)%", output_text)
        results['utv_error_pct'] = safe_search(r"unvoiced to voiced error:\n.*? = ([\d.]+)%", output_text)
        results['mean_semitone_error'] = safe_search(r"mean of frequency estimation error in semitones:\n.*? = ([\d.]+) semitones", output_text)
        results['acc_at_0.1'] = safe_search(r"frequency estimation accuracy @ threshold 0.1semitones:\n.*? = ([\d.]+)%", output_text)
        results['acc_at_0.5'] = safe_search(r"frequency estimation accuracy @ threshold 0.5semitones:\n.*? = ([\d.]+)%", output_text)
        results['acc_at_1.0'] = safe_search(r"frequency estimation accuracy @ threshold 1semitones:\n.*? = ([\d.]+)%", output_text)
    except Exception as e:
        print(f"  [!] Parsing error occurred: {e}")
        print("  --- Received Output (Partial) ---")
        print('\n'.join(output_text.splitlines()[-10:]))
        print("  -----------------------")
        return None

    return results

# 3. Function to process a 'single job'
def run_test_case(params):
    """
    Runs a single C++ test case for the given parameters.
    Returns a dictionary with parameters and results, or None on failure.
    """
    vt_prob, yt_prob, vtu_prob, utv_prob = params
    
    print(f"  [Starting] voiced_trust={vt_prob}, yin_trust={yt_prob}, vtu={vtu_prob}, utv={utv_prob}")

    command = [
        EXE_PATH,
        str(vt_prob),
        str(yt_prob),
        str(vtu_prob),
        str(utv_prob)
    ]

    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        parsed_data = parse_output(result.stdout)

        if parsed_data:
            # Combine parameters and results, then return
            row_data = {
                'voiced_trust': vt_prob,
                'yin_trust': yt_prob,
                'v_to_u_prob': vtu_prob,
                'u_to_v_prob': utv_prob,
                **parsed_data
            }
            print(f"  [Finished] voiced_trust={vt_prob}, yin_trust={yt_prob}, vtu={vtu_prob}, utv={utv_prob}")
            return row_data
        
    except subprocess.CalledProcessError as e:
        # C++ program exited with a non-zero code (e.g., assert)
        print(f"  [!] EXECUTION ERROR for params {params} (Return Code: {e.returncode}):")
        print("  --- STDERR ---")
        print(e.stderr or "No stderr output.")
        print("  --------------")
    except Exception as e:
        print(f"  [!] SCRIPT ERROR for params {params}: {e}")

    return None # Return None on failure

def main():
    if not os.path.exists(EXE_PATH):
        print(f"Error: Executable file not found. Check the path: {EXE_PATH}")
        return

    # Generate parameter combinations
    param_combinations = list(itertools.product(
        VOICED_TRUST_PROBS,
        YIN_TRUST_PROBS,
        VOICED_TO_UNVOICED_PROBS,
        UNVOICED_TO_VOICED_PROBS
    ))

    print(f"Starting tests with {len(param_combinations)} combinations using up to {MAX_WORKERS} processes...")
    print(f"Results will be saved to '{OUTPUT_CSV}'.")

    # CSV Headers
    headers = [
        'voiced_trust', 'yin_trust', 'v_to_u_prob', 'u_to_v_prob',
        'vtu_error_pct', 'utv_error_pct', 'mean_semitone_error',
        'acc_at_0.1', 'acc_at_0.5', 'acc_at_1.0'
    ]

    # Write to CSV file
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        # 4. Run in parallel using ProcessPoolExecutor
        # The executor.map distributes the job list (param_combinations)
        # to the worker function (run_test_case).
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            # executor.map returns results in the order the jobs were submitted.
            results = executor.map(run_test_case, param_combinations)
            
            # 5. Iterate through completed results and write to CSV
            processed_count = 0
            for row_data in results:
                if row_data:
                    writer.writerow(row_data)
                    processed_count += 1
            
            print(f"\nAll tests completed. {processed_count}/{len(param_combinations)} combinations were successful.")
            print(f"Check the '{OUTPUT_CSV}' file.")

if __name__ == "__main__":
    main()