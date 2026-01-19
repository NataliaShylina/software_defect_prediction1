import pandas as pd
import re


def load_promise_arff(file_path):
    """
    Robustly loads PROMISE ARFF files by manually parsing the
    attributes and data to avoid 'BadLayout' errors.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    attributes = []
    data_start_line = 0

    for i, line in enumerate(lines):
        # Identify attributes
        if line.lower().startswith('@attribute'):
            # Extract the name between '@attribute' and the type
            attr_name = re.findall(r"@attribute\s+['\"]?([\w.-]+)['\"]?", line, re.I)[0]
            attributes.append(attr_name)

        # Identify where data begins
        if line.lower().startswith('@data'):
            data_start_line = i + 1
            break

    # Read only the data part into a DataFrame
    df = pd.DataFrame([l.strip().split(',') for l in lines[data_start_line:] if l.strip()])

    # Clean up column types (convert strings to numbers where possible)
    df.columns = attributes
    return df.apply(pd.to_numeric, errors='ignore')


# Usage
df = load_promise_arff('../software_defect_prediction1/kitchenham.arff')
print(df.head())
print(f"\nColumns found: {df.columns.tolist()}")

