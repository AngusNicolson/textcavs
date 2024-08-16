
from pathlib import Path

from tqdm import tqdm
import numpy as np


def main():
    dataset_dir = Path("data/mimic-cxr-reports")
    if not dataset_dir.exists():
        raise FileNotFoundError("Dataset directory does not exist. "
                                "Have you saved the reports in data/mimic-cxr-reports?")
    all_files = list(dataset_dir.glob("**/*.txt"))
    all_lines = []

    for path in tqdm(all_files):
        with open(path, "r") as fp:
            text_data = fp.read()
        lines = text_data.split("\n")
        findings = []
        impressions = []
        findings_line = None
        impressions_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith("FINDINGS"):
                findings_line = i
            if line.strip().startswith("IMPRESSION"):
                impressions_line = i

        if findings_line is not None:
            for i, line in enumerate(lines[findings_line:]):
                if i != 0 and is_first_word_capitalized(line):
                    break
                else:
                    line = line.strip()
                    if line.startswith("FINDINGS:"):
                        line = line[9:]
                    if line:
                        findings.append(line)
        if impressions_line is not None:
            for i, line in enumerate(lines[impressions_line:]):
                if i != 0 and is_first_word_capitalized(line):
                    break
                else:
                    line = line.strip()
                    if line.startswith("IMPRESSION:"):
                        line = line[11:]
                    if line:
                        impressions.append(line)

        findings = " ".join(findings)
        impressions = " ".join(impressions)
        lines_to_keep = findings.split(".") + impressions.split(".")
        lines_to_keep = [line.strip() for line in lines_to_keep if len(line) > 2]
        all_lines += lines_to_keep

    outpath = "data/text_concepts/mimic_unfiltered_lines.txt"
    with open(outpath, "w") as fp:
        fp.write("\n".join(all_lines))
    print(f"Saved MIMIC sentences to {outpath}")
    print("Done!")


def is_first_word_capitalized(line):
    words = line.strip().split()
    return words and words[0].isupper()


if __name__ == "__main__":
    main()
