import os
import re

INPUT_DIR = "../texts/raw"          # folder where pg*.txt files are stored
OUTPUT_DIR = "../texts/cleaned"       # folder to write cleaned files
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_RE = re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
END_RE   = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)

def clean_gutenberg_text(text):
    """
    Extract main book contents by removing Gutenberg header/footer.
    """
    lines = text.splitlines()
    start_index = 0
    end_index = len(lines)

    # Find start marker
    for i, line in enumerate(lines):
        if START_RE.search(line):
            start_index = i + 1
            break

    # Find end marker
    for i, line in enumerate(lines):
        if END_RE.search(line):
            end_index = i
            break

    # Extract main book content
    content = lines[start_index:end_index]
    
    # Remove stray boilerplate lines
    cleaned = []
    for line in content:
        if "Project Gutenberg" in line:
            continue
        if "www.gutenberg" in line.lower():
            continue
        if line.strip().startswith("***"):
            continue
        cleaned.append(line)
    
    return "\n".join(cleaned).strip()

def process_all():
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".txt"):
            continue
        
        path = os.path.join(INPUT_DIR, filename)
        print(f"[+] Processing {filename}")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        cleaned = clean_gutenberg_text(raw)

        out_path = os.path.join(OUTPUT_DIR, filename.replace(".txt","_clean.txt"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"    → Saved cleaned text to {out_path}")

if __name__ == "__main__":
    process_all()
