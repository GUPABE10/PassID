import pandas as pd
import os

def count_total_detected(program_dir):
    total_detected = 0
    for program_file in os.listdir(program_dir):
        if program_file.endswith("_passes.csv"):
            program_path = os.path.join(program_dir, program_file)
            program_df = pd.read_csv(program_path)
            total_detected += len(program_df)
    return total_detected

def count_total_gt(gt_dir):
    total_gt = 0
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".csv"):
            gt_path = os.path.join(gt_dir, gt_file)
            gt_df = pd.read_csv(gt_path)
            total_gt += len(gt_df)
    return total_gt

def compare_passes(gt_file, program_output_file, tolerance=1):
    # Leer archivos CSV
    gt_df = pd.read_csv(gt_file)
    program_df = pd.read_csv(program_output_file)
    
    # Renombrar columnas para facilitar la comparación
    gt_df.columns = ['Start', 'End', 'Type']
    program_df.columns = ['Passer', 'Receiver', 'Start', 'Duration', 'End']

    # Inicializar listas de resultados
    true_positives = []
    false_positives = []
    false_negatives = []

    # Inicializar contadores de tipos
    tp_aereo = 0
    tp_terrestre = 0
    fn_aereo = 0
    fn_terrestre = 0

    # Convertir los tiempos a float
    gt_df['Start'] = gt_df['Start'].astype(float)
    program_df['Start'] = program_df['Start'].astype(float)

    # Comparar los pases
    for _, gt_row in gt_df.iterrows():
        matched = False
        for _, prog_row in program_df.iterrows():
            if abs(gt_row['Start'] - prog_row['Start']) <= tolerance:
                true_positives.append(prog_row)
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

    # Encontrar false positives
    for _, prog_row in program_df.iterrows():
        matched = False
        for tp in true_positives:
            if prog_row.equals(tp):
                matched = True
                break
        if not matched:
            false_positives.append(prog_row)

    # Convertir listas a DataFrames para facilitar el manejo
    true_positives_df = pd.DataFrame(true_positives)
    false_positives_df = pd.DataFrame(false_positives)
    false_negatives_df = pd.DataFrame(false_negatives)

    return (true_positives_df, false_positives_df, false_negatives_df,
            tp_aereo, tp_terrestre, fn_aereo, fn_terrestre)

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
                tp, fp, fn, tp_aereo, tp_terrestre, fn_aereo, fn_terrestre = compare_passes(gt_path, program_path, tolerance)
                
                total_tp += len(tp)
                total_fp += len(fp)
                total_fn += len(fn)
                
                total_tp_aereo += tp_aereo
                total_tp_terrestre += tp_terrestre
                total_fn_aereo += fn_aereo
                total_fn_terrestre += fn_terrestre
            else:
                # Si el archivo no existe en JugadasOut, todos los pases de GT son falsos negativos
                gt_df = pd.read_csv(gt_path)
                total_fn += len(gt_df)
                
                fn_aereo = len(gt_df[gt_df['Tipo'] == 'Aereo'])
                fn_terrestre = len(gt_df[gt_df['Tipo'] == 'Terrestre'])
                
                total_fn_aereo += fn_aereo
                total_fn_terrestre += fn_terrestre

    return (total_tp, total_fp, total_fn,
            total_tp_aereo, total_tp_terrestre, total_fn_aereo, total_fn_terrestre)

def count_total_aereo(gt_dir):
    total_aereo = 0
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".csv"):
            gt_path = os.path.join(gt_dir, gt_file)
            gt_df = pd.read_csv(gt_path)
            total_aereo += len(gt_df[gt_df['Tipo'] == 'Aereo'])
    return total_aereo

def count_total_terrestre(gt_dir):
    total_terrestre = 0
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".csv"):
            gt_path = os.path.join(gt_dir, gt_file)
            gt_df = pd.read_csv(gt_path)
            total_terrestre += len(gt_df[gt_df['Tipo'] == 'Terrestre'])
    return total_terrestre

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

def calculate_f1_score(precision, recall):
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return f1_score

def calculate_recall(tp, fn):
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    return recall

# Directorios de entrada
gt_dir = 'GT'
program_dir = 'JugadasOut'

# Tolerancia en segundos (por defecto es 1)
tolerance = 1

# Ejecutar el proceso
results = process_passes(gt_dir, program_dir, tolerance)

total_tp, total_fp, total_fn, total_tp_aereo, total_tp_terrestre, total_fn_aereo, total_fn_terrestre = results

# Calcular precisión y recall
precision, recall = calculate_precision_recall(total_tp, total_fp, total_fn)

# Calcular F1-Score
f1_score = calculate_f1_score(precision, recall)

# Calcular recall para Aereos y Terrestres
recall_aereo = calculate_recall(total_tp_aereo, total_fn_aereo)
recall_terrestre = calculate_recall(total_tp_terrestre, total_fn_terrestre)

# Contar el total de pases detectados y el total de GT
total_detected = count_total_detected(program_dir)
total_gt = count_total_gt(gt_dir)

# Contar el total de pases en GT que son Aereos y Terrestres
total_aereo = count_total_aereo(gt_dir)
total_terrestre = count_total_terrestre(gt_dir)

# # Imprimir resultados
# print("### Resultados Generales ###")
# print(f"Total True Positives: {total_tp}")
# print(f"Total False Positives: {total_fp}")
# print(f"Total False Negatives: {total_fn}")
# print(f"Total True Positives Aereo: {total_tp_aereo}")
# print(f"Total True Positives Terrestre: {total_tp_terrestre}")
# print(f"Total False Negatives Aereo: {total_fn_aereo}")
# print(f"Total False Negatives Terrestre: {total_fn_terrestre}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1_score:.4f}")
# print(f"Total de Pases Detectados: {total_detected}")
# print(f"Total de Pases en GT: {total_gt}")
# print(f"Total de Pases Aereos en GT: {total_aereo}")
# print(f"Total de Pases Terrestres en GT: {total_terrestre}")
# print(f"Recall de Pases Aereos: {recall_aereo:.4f}")
# print(f"Recall de Pases Terrestres: {recall_terrestre:.4f}")


# Imprimir resultados
print("### Resultados Generales ###")
print(f"{'Total True Positives:':<30} {total_tp}")
print(f"{'Total False Positives:':<30} {total_fp}")
print(f"{'Total False Negatives:':<30} {total_fn}")
print("\n### Detalle de Tipos ###")
print(f"{'Total True Positives Aereo:':<30} {total_tp_aereo}")
print(f"{'Total True Positives Terrestre:':<30} {total_tp_terrestre}")
print(f"{'Total False Negatives Aereo:':<30} {total_fn_aereo}")
print(f"{'Total False Negatives Terrestre:':<30} {total_fn_terrestre}")
print("\n### Métricas Generales ###")
print(f"{'Precision:':<30} {precision:.4f}")
print(f"{'Recall:':<30} {recall:.4f}")
print(f"{'F1-Score:':<30} {f1_score:.4f}")
print("\n### Totales de Pases ###")
print(f"{'Total de Pases Detectados:':<30} {total_detected}")
print(f"{'Total de Pases en GT:':<30} {total_gt}")
print(f"{'Total de Pases Aereos en GT:':<30} {total_aereo}")
print(f"{'Total de Pases Terrestres en GT:':<30} {total_terrestre}")
print("\n### Recall por Tipo ###")
print(f"{'Recall de Pases Aereos:':<30} {recall_aereo:.4f}")
print(f"{'Recall de Pases Terrestres:':<30} {recall_terrestre:.4f}")
