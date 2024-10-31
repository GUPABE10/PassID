import pandas as pd
import os

# Function to count total detected passes in program output files
def count_total_detected(program_dir):
    total_detected = 0
    for program_file in os.listdir(program_dir):
        if program_file.endswith("_passes.csv"):
            program_path = os.path.join(program_dir, program_file)
            program_df = pd.read_csv(program_path)
            total_detected += len(program_df)  # Count rows for each detected pass
    return total_detected

# Function to count total ground truth (GT) passes
def count_total_gt(gt_dir):
    total_gt = 0
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".csv"):
            gt_path = os.path.join(gt_dir, gt_file)
            gt_df = pd.read_csv(gt_path)
            total_gt += len(gt_df)  # Count rows for each GT pass
    return total_gt

# Compare passes between GT and program output within a specified tolerance
def compare_passes(gt_file, program_output_file, tolerance=1):
    gt_df = pd.read_csv(gt_file)
    program_df = pd.read_csv(program_output_file)

    # Rename columns for consistency
    gt_df.columns = ['Start', 'End', 'Type']
    program_df.columns = ['Passer', 'Receiver', 'Start', 'Duration', 'End']

    # Initialize result lists
    true_positives = []
    false_positives = []
    false_negatives = []

    # Initialize counters by pass type
    tp_aereo = 0
    tp_terrestre = 0
    fn_aereo = 0
    fn_terrestre = 0

    # Convert times to float for comparison
    gt_df['Start'] = gt_df['Start'].astype(float)
    program_df['Start'] = program_df['Start'].astype(float)

    # Compare GT and program passes
    for _, gt_row in gt_df.iterrows():
        matched = False
        for _, prog_row in program_df.iterrows():
            if abs(gt_row['Start'] - prog_row['Start']) <= tolerance:
                true_positives.append(prog_row)
                # Count true positives by pass type
                if gt_row['Type'] == 'Aereo':
                    tp_aereo += 1
                elif gt_row['Type'] == 'Terrestre':
                    tp_terrestre += 1
                matched = True
                break
        if not matched:
            false_negatives.append(gt_row)
            if gt_row['Type'] == 'Aereo':
                fn_aereo += 1
            elif gt_row['Type'] == 'Terrestre':
                fn_terrestre += 1

    # Identify false positives by exclusion from true positives
    for _, prog_row in program_df.iterrows():
        matched = False
        for tp in true_positives:
            if prog_row.equals(tp):
                matched = True
                break
        if not matched:
            false_positives.append(prog_row)

    # Convert results to DataFrames for easier analysis
    true_positives_df = pd.DataFrame(true_positives)
    false_positives_df = pd.DataFrame(false_positives)
    false_negatives_df = pd.DataFrame(false_negatives)

    return (true_positives_df, false_positives_df, false_negatives_df,
            tp_aereo, tp_terrestre, fn_aereo, fn_terrestre)

# Process passes for all files, comparing GT and detected passes
def process_passes(gt_dir, program_dir, tolerance=1):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    total_tp_aereo = 0
    total_tp_terrestre = 0
    total_fn_aereo = 0
    total_fn_terrestre = 0
    
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".csv"):
            program_file = gt_file.replace(".csv", "_passes.csv")
            
            gt_path = os.path.join(gt_dir, gt_file)
            program_path = os.path.join(program_dir, program_file)
            
            if os.path.exists(program_path):
                # Compare passes and count TP, FP, FN for each type
                tp, fp, fn, tp_aereo, tp_terrestre, fn_aereo, fn_terrestre = compare_passes(gt_path, program_path, tolerance)
                
                total_tp += len(tp)
                total_fp += len(fp)
                total_fn += len(fn)
                
                total_tp_aereo += tp_aereo
                total_tp_terrestre += tp_terrestre
                total_fn_aereo += fn_aereo
                total_fn_terrestre += fn_terrestre
            else:
                # If output file does not exist, count all GT passes as false negatives
                gt_df = pd.read_csv(gt_path)
                total_fn += len(gt_df)
                
                fn_aereo = len(gt_df[gt_df['Tipo'] == 'Aereo'])
                fn_terrestre = len(gt_df[gt_df['Tipo'] == 'Terrestre'])
                
                total_fn_aereo += fn_aereo
                total_fn_terrestre += fn_terrestre

    return (total_tp, total_fp, total_fn,
            total_tp_aereo, total_tp_terrestre, total_fn_aereo, total_fn_terrestre)

# Count total aerial passes from GT
def count_total_aereo(gt_dir):
    total_aereo = 0
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".csv"):
            gt_path = os.path.join(gt_dir, gt_file)
            gt_df = pd.read_csv(gt_path)
            total_aereo += len(gt_df[gt_df['Tipo'] == 'Aereo'])
    return total_aereo

# Count total terrestrial passes from GT
def count_total_terrestre(gt_dir):
    total_terrestre = 0
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".csv"):
            gt_path = os.path.join(gt_dir, gt_file)
            gt_df = pd.read_csv(gt_path)
            total_terrestre += len(gt_df[gt_df['Tipo'] == 'Terrestre'])
    return total_terrestre

# Calculate precision and recall based on true positives, false positives, and false negatives
def calculate_precision_recall(tp, fp, fn):
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    
    return precision, recall

# Calculate F1-score based on precision and recall
def calculate_f1_score(precision, recall):
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return f1_score

# Calculate recall specifically based on true positives and false negatives
def calculate_recall(tp, fn):
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    return recall

# Directories for ground truth and program output files
gt_dir = 'GT'
program_dir = 'JugadasOut'

# Tolerance in seconds (default is 1)
tolerance = 1

# Execute pass processing and comparisons
results = process_passes(gt_dir, program_dir, tolerance)

total_tp, total_fp, total_fn, total_tp_aereo, total_tp_terrestre, total_fn_aereo, total_fn_terrestre = results

# Calculate precision and recall
precision, recall = calculate_precision_recall(total_tp, total_fp, total_fn)

# Calculate F1-Score
f1_score = calculate_f1_score(precision, recall)

# Calculate recall for aerial and terrestrial passes
recall_aereo = calculate_recall(total_tp_aereo, total_fn_aereo)
recall_terrestre = calculate_recall(total_tp_terrestre, total_fn_terrestre)

# Count total detected passes and total GT passes
total_detected = count_total_detected(program_dir)
total_gt = count_total_gt(gt_dir)

# Count total aerial and terrestrial passes in GT
total_aereo = count_total_aereo(gt_dir)
total_terrestre = count_total_terrestre(gt_dir)

# Print results
print("### General Results ###")
print(f"{'Total True Positives:':<30} {total_tp}")
print(f"{'Total False Positives:':<30} {total_fp}")
print(f"{'Total False Negatives:':<30} {total_fn}")
print("\n### Type Details ###")
print(f"{'Total True Positives Aereo:':<30} {total_tp_aereo}")
print(f"{'Total True Positives Terrestre:':<30} {total_tp_terrestre}")
print(f"{'Total False Negatives Aereo:':<30} {total_fn_aereo}")
print(f"{'Total False Negatives Terrestre:':<30} {total_fn_terrestre}")
print("\n### General Metrics ###")
print(f"{'Precision:':<30} {precision:.4f}")
print(f"{'Recall:':<30} {recall:.4f}")
print(f"{'F1-Score:':<30} {f1_score:.4f}")
print("\n### Total Passes ###")
print(f"{'Total Detected Passes:':<30} {total_detected}")
print(f"{'Total GT Passes:':<30} {total_gt}")
print(f"{'Total Aereo GT Passes:':<30} {total_aereo}")
print(f"{'Total Terrestre GT Passes:':<30} {total_terrestre}")
print("\n### Recall by Type ###")
print(f"{'Recall Aereo Passes:':<30} {recall_aereo:.4f}")
print(f"{'Recall Terrestre Passes:':<30} {recall_terrestre:.4f}")
