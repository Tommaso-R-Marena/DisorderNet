"""
Fetch DisProt database entries for training and evaluation.
DisProt contains experimentally verified intrinsically disordered regions.
"""
import requests
import json
import os
import time

OUTPUT_DIR = "/home/user/workspace/disorder_model/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_disprot_entries():
    """Fetch DisProt entries via the REST API."""
    url = "https://disprot.org/api/search"
    
    all_entries = []
    page = 0
    page_size = 100
    
    while True:
        params = {
            "release": "current",
            "show_ambiguous": "false",
            "show_obsolete": "false",
            "format": "json",
            "page": page,
            "per_page": page_size,
        }
        
        print(f"Fetching page {page}...")
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
        
        entries = data.get("data", [])
        if not entries:
            break
        
        all_entries.extend(entries)
        print(f"  Got {len(entries)} entries (total: {len(all_entries)})")
        
        # Check if we have all
        total = data.get("total", 0)
        if len(all_entries) >= total or len(entries) < page_size:
            break
        
        page += 1
        time.sleep(0.5)  # Be respectful
    
    print(f"\nTotal entries fetched: {len(all_entries)}")
    return all_entries


def process_entries(entries):
    """Extract sequences and disorder annotations from DisProt entries."""
    processed = []
    
    for entry in entries:
        disprot_id = entry.get("disprot_id", "")
        acc = entry.get("acc", "")
        name = entry.get("name", "")
        sequence = entry.get("sequence", "")
        
        if not sequence or len(sequence) < 20:
            continue
        
        # Extract disorder regions from regions field
        regions = entry.get("regions", [])
        disorder_regions = []
        
        for region in regions:
            # Get region type - we want disorder annotations
            reg_type = region.get("type", "")
            start = region.get("start")
            end = region.get("end")
            
            if start is not None and end is not None:
                # DisProt uses 1-based indexing
                disorder_regions.append({
                    "start": int(start),
                    "end": int(end),
                    "type": reg_type
                })
        
        if not disorder_regions:
            continue
        
        # Create binary disorder label array
        seq_len = len(sequence)
        disorder_labels = [0] * seq_len  # 0 = ordered (or unknown), 1 = disordered
        
        for region in disorder_regions:
            s = region["start"] - 1  # Convert to 0-based
            e = region["end"]  # end is inclusive in DisProt
            for i in range(max(0, s), min(e, seq_len)):
                disorder_labels[i] = 1
        
        disorder_fraction = sum(disorder_labels) / len(disorder_labels)
        
        processed.append({
            "disprot_id": disprot_id,
            "uniprot_acc": acc,
            "name": name,
            "sequence": sequence,
            "length": seq_len,
            "disorder_labels": disorder_labels,
            "disorder_fraction": disorder_fraction,
            "num_disorder_regions": len(disorder_regions),
            "regions": disorder_regions
        })
    
    return processed


def main():
    print("=" * 60)
    print("FETCHING DISPROT DATABASE")
    print("=" * 60)
    
    entries = fetch_disprot_entries()
    
    if not entries:
        print("No entries fetched. Using fallback approach...")
        # Try direct download
        print("Trying direct API endpoint...")
        try:
            resp = requests.get(
                "https://disprot.org/api/search?release=current&format=json&per_page=500",
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("data", [])
            print(f"Got {len(entries)} entries via direct request")
        except Exception as e:
            print(f"Direct request also failed: {e}")
    
    if entries:
        processed = process_entries(entries)
        print(f"\nProcessed {len(processed)} proteins with disorder annotations")
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, "disprot_processed.json")
        with open(output_path, "w") as f:
            json.dump(processed, f)
        print(f"Saved to {output_path}")
        
        # Stats
        total_residues = sum(p["length"] for p in processed)
        total_disordered = sum(sum(p["disorder_labels"]) for p in processed)
        print(f"\nDataset Statistics:")
        print(f"  Proteins: {len(processed)}")
        print(f"  Total residues: {total_residues}")
        print(f"  Disordered residues: {total_disordered} ({100*total_disordered/total_residues:.1f}%)")
        print(f"  Ordered residues: {total_residues - total_disordered} ({100*(total_residues-total_disordered)/total_residues:.1f}%)")
        print(f"  Avg length: {total_residues/len(processed):.0f}")
        print(f"  Avg disorder fraction: {sum(p['disorder_fraction'] for p in processed)/len(processed):.3f}")
    else:
        print("Failed to fetch any data.")


if __name__ == "__main__":
    main()
