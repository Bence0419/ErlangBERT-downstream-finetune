import json
from collections import defaultdict
from pathlib import Path

# --- KONFIGURÁCIÓ ---
INPUT_FILE = "output/code_search_dataset/train.jsonl" # Ide írd az egyik létező fájlod elérési útját
# --------------------

def inspect_idx(path: str):
    path = Path(path)
    if not path.exists():
        print(f"HIBA: A fájl nem található: {path}")
        return

    groups = defaultdict(list)
    
    print(f"Fájl olvasása: {path} ...")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                idx = str(rec.get("idx"))
                # Csak a diagnosztikához szükséges mezőket tároljuk
                info = {
                    "func_name": rec.get("func_name"),
                    "code_snippet": rec.get("code", "")[:100].replace("\n", " "), # Első 100 karakter
                    "path": rec.get("path")
                }
                groups[idx].append(info)
            except:
                continue

    # Statisztika
    total_groups = len(groups)
    multi_item_groups = {k: v for k, v in groups.items() if len(v) > 1}
    
    print("-" * 60)
    print(f"Összes egyedi 'idx': {total_groups}")
    print(f"Csoportok száma, ahol több mint 1 kód van (potenciális klónok): {len(multi_item_groups)}")
    
    if len(multi_item_groups) == 0:
        print("\n[!] FIGYELEM: Nincs olyan idx, amihez több kód tartozna!")
        print("Ez azt jelenti, hogy az 'idx' egyedi azonosító, NEM használható klón csoportosításra.")
        return

    print("-" * 60)
    print("MINTA VIZSGÁLAT (Top 3 legnagyobb csoport):")
    
    # A 3 legnagyobb csoport kiválasztása
    sorted_groups = sorted(multi_item_groups.items(), key=lambda item: len(item[1]), reverse=True)[:3]
    
    for idx, items in sorted_groups:
        print(f"\nIDX: {idx} (Elemszám: {len(items)})")
        # Kiírunk max 3 példát ebből a csoportból
        for i, item in enumerate(items[:3]):
            print(f"  {i+1}. Funkció: {item['func_name']}")
            print(f"     Path:     {item['path']}")
            print(f"     Code:     {item['code_snippet']}...")

if __name__ == "__main__":
    inspect_idx(INPUT_FILE)