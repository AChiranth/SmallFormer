import os
import re
import glob

INPUT_DIR = "../texts/raw" 
OUTPUT_DIR = "../texts/cleaned"  
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_RE = re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
END_RE   = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)

CHAPTER_RE = re.compile(r"^CHAPTER\s+[A-Z0-9IVXLC]+\.?$")

def clean_gutenberg_text(text):
    lines = text.splitlines()
    start_index, end_index = 0, len(lines)

    for i, line in enumerate(lines):
        if START_RE.search(line):
            start_index = i + 1
            break

    for i, line in enumerate(lines):
        if END_RE.search(line):
            end_index = i
            break

    content = lines[start_index:end_index]
    cleaned = []

    skip_next = False 

    for i, line in enumerate(content):
        stripped = line.strip()

        if "PROJECT GUTENBERG" in stripped.upper():
            continue
        if stripped.startswith("***"):
            continue

        if CHAPTER_RE.match(stripped):
            skip_next = True 
            continue

        if skip_next:
            skip_next = False
            continue

        if stripped.isupper() and len(stripped.split()) >= 3:
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def process_all():
    merged = []

    for file_path in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
        filename = os.path.basename(file_path)
        print(f"[+] Processing {filename}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        cleaned = clean_gutenberg_text(raw)

        out_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", "_clean.txt"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"    → saved cleaned file to {out_path}")

        merged.append(cleaned)

    with open(MERGE_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n\n".join(merged))

    print(f"[✓] Finished! Corpus saved to {MERGE_OUTPUT}")

if __name__ == "__main__":
    process_all()