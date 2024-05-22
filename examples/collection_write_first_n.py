import os
import csv
from tqdm import tqdm

def export_first_n_passages(input_file: str, output_file: str, n: int, id_col: int = 0, text_col: int = 1, title_col: int = 2):
    """
    Export the first n passages from the input file to the output file.

    :param input_file: Path to the original dataset file.
    :param output_file: Path to the output file where the subset will be saved.
    :param n: Number of passages to export.
    :param id_col: Column index of the passage ID.
    :param text_col: Column index of the passage text.
    :param title_col: Column index of the passage title.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        for i, row in enumerate(tqdm(reader, desc="Exporting passages", total=n)):
            if i == 0 and row[id_col] == "id":
                # Write header if present
                writer.writerow(row)
                continue
            if i >= n:
                break
            writer.writerow(row)

if __name__ == "__main__":
    input_file = "data/nq/collection/psgs_w100.tsv"
    output_file = "data/nq/collection/psgs_w100_first_10000.tsv"
    n = 10000

    export_first_n_passages(input_file, output_file, n)
