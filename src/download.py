import os
import requests

OUTPUT_DIR = "../data/lovecraft_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lovecraft texts on Project Gutenberg (public domain)
LOVECRAFT_BOOKS = {
    "call_of_cthulhu": 68283,
    "dagon": 8486,
    "the_nameless_city": 95746,
    "the_festival": 95745,
    "the_tomb": 95747,
    "the_alchemist": 95748,
    "the_temple": 95749,
    "the_outsider": 95750,
    "the_cats_of_ulthar": 95751,
    "the_white_ship": 95752,
    "the_rats_in_the_walls": 94373,
    "nyarlathotep": 65811,
    "the_colour_out_of_space": 62459,
    "the_shadow_over_innsmouth": 73544,
}

def download_gutenberg_text(book_id, title):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    print(f"[+] Downloading {title} ({book_id})")

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[!] Failed: {title} ({resp.status_code})")
        return

    out_path = os.path.join(OUTPUT_DIR, f"{title}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    print(f"[✓] Saved: {out_path}")

if __name__ == "__main__":
    for title, book_id in LOVECRAFT_BOOKS.items():
        download_gutenberg_text(book_id, title)
    print("\n[✓] All Lovecraft raw texts downloaded.")