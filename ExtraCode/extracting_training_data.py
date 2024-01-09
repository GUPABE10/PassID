import re
# Write data to CSV file
import csv




def parse_training_line(line):
    """Parse a training line to extract relevant data."""
    data = {}
    patterns = {
        'epoch': r'Epoch: \[(\d+)\]',
        'step': r'\[\s*(\d+)/\d+\]',
        'eta': r'eta: ([\d:]+)',
        'lr': r'lr: ([\d.]+)',
        'loss': r'loss: ([\d.]+)',
        'loss_classifier': r'loss_classifier: ([\d.]+)',
        'loss_box_reg': r'loss_box_reg: ([\d.]+)',
        'loss_objectness': r'loss_objectness: ([\d.]+)',
        'loss_rpn_box_reg': r'loss_rpn_box_reg: ([\d.]+)',
        'time': r'time: ([\d.]+)',
        'data': r'data: ([\d.]+)',
        'max_mem': r'max mem: (\d+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            data[key] = match.group(1)

    return data



def parse_evaluation_line(line):
    """Parse an evaluation line to extract detailed data for refined CSV structure."""
    data = {}
    # Extract the metric type (AP or AR), IoU, area, maxDets, and value
    match = re.search(r'Average (Precision|Recall)\s+\(.*?IoU=([0-9.]+:[0-9.]+|[0-9.]+)\s*\|\s*area=\s*(.*?)\s*\|\s*maxDets=\s*(\d+)\s*\]\s*=\s*([0-9.]+)', line)
    if match:
        data['Metric Type'] = match.group(1)  # AP or AR
        data['IoU'] = match.group(2).strip()
        data['Area'] = match.group(3).strip()
        data['maxDets'] = match.group(4).strip()
        data['Value'] = match.group(5).strip()
    return data




def process_complete_log_file_v2(filename):
    """Process the complete log file for both training and evaluation data."""
    training_data = []
    evaluation_data = []
    current_epoch = None
    in_evaluation_section = False
    training_fieldnames = set()

    with open(filename, 'r') as file:
        for line in file:
            if 'Epoch:' in line and '[' in line and ']' in line and 'eta:' in line:
                in_evaluation_section = False
                # Training line
                training_info = parse_training_line(line)
                if training_info.get('epoch') is not None:
                    current_epoch = training_info['epoch']
                if current_epoch is not None:
                    training_info['epoch'] = current_epoch  # Add epoch info if missing
                training_data.append(training_info)
                training_fieldnames.update(training_info.keys())
            elif 'IoU metric:' in line:
                # Start of evaluation data for an epoch
                in_evaluation_section = True
            elif in_evaluation_section and ('AP' in line or 'AR' in line):
                # Evaluation line
                evaluation_info = parse_evaluation_line(line)
                if evaluation_info:
                    evaluation_info['Epoch'] = current_epoch
                    evaluation_data.append(evaluation_info)

    # Write training data to CSV file
    import csv

    training_fieldnames = sorted(training_fieldnames)

    with open('training_progress.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=training_fieldnames)
        writer.writeheader()
        for row in training_data:
            writer.writerow(row)

    # Write evaluation data to CSV file
    evaluation_fieldnames = ['Epoch', 'Metric Type', 'IoU', 'Area', 'maxDets', 'Value']

    with open('evaluation_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=evaluation_fieldnames)
        writer.writeheader()
        for row in evaluation_data:
            writer.writerow(row)

    return 'training_progress.csv', 'evaluation_results.csv'

# Example usage
# process_complete_log_file_v2('training_log.txt')  # Uncomment and replace with the actual log file name






process_complete_log_file_v2('./data/training_detector/training_log.txt')  # Uncomment and replace with the actual log file name