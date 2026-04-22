import json
import hashlib
from pathlib import Path
from src.preprocess import TextCleaner, Chunker, DocumentProcessor
from src.config import settings

def main():
    input_file = settings.DATA_DIR / "raw" / "simplewiki_10k.jsonl"
    output_file = settings.DATA_DIR / "processed" / "chunks.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # CHeck the output directory
    # print(f"Output directory exists? {output_file.parent.exists()}")
     
    chunker = Chunker()
    processor = DocumentProcessor(chunker)
    
    seen_hashes = set()
    total_processed = 0
    total_written = 0
    
    # Open output file in write mode
    with open(output_file, "w", encoding="utf-8") as out_f:
        # Iterate through processor.process_file(input_file)
        try:
            for chunk_dict in processor.process_file(input_file):
                chunk_text = chunk_dict["text"]
                hash_val = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                
                if hash_val not in seen_hashes:
                    # Write json.dumps(chunk) + "\n" to out_f
                    out_f.write(json.dumps(chunk_dict) + "\n")
                    # Add hash to seen_hashes
                    seen_hashes.add(hash_val)
                    total_written += 1
                
                total_processed += 1
        except Exception as e:
            print(f"Error processing file: {e}")
    
    print(f"Processed {total_processed} chunks. Wrote {total_written} unique chunks.")
    
if __name__ == "__main__":
    main()