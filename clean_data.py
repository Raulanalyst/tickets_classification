import json
import os
from tqdm import tqdm
import re
from nltk.corpus import words, stopwords
import nltk

# Load word sets once at module level
ENGLISH_WORDS = set(w.lower() for w in words.words())
STOP_WORDS = set(stopwords.words('english'))
WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')
PARENTHESES_PATTERN = re.compile(r'\([^)]*\)')

def contains_english_word(text):
    """
    Check if text contains at least one English word (excluding stop words)
    
    Args:
        text: String to check
        
    Returns:
        bool: True if contains at least one meaningful English word
    """
    # Extract words from text (alphanumeric only)
    text_lower = text.lower()
    text_words = WORD_PATTERN.findall(text_lower)
    
    # Check if any word is English and not a stop word
    for word in text_words:
        if word in ENGLISH_WORDS and word not in STOP_WORDS:
            return True
    
    return False

def clean_ticket_data(input_file, output_file, batch_size=10000):
    """
    Read JSONL file, remove records with empty 'description' field,
    and save cleaned data to a new JSONL file with batched writes.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return

    print("Counting total lines...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"Total lines found: {total_lines:,}\n")

    total_records = 0
    empty_descriptions = 0
    cleaned_records = 0
    no_type = 0
    no_name = 0
    no_format = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', buffering=1024*1024) as outfile:
            
            batch = []
            
            for line_num, line in enumerate(tqdm(infile, total=total_lines, desc="Processing records"), 1):
                total_records += 1
                try:
                    record = json.loads(line)
                    
                    # Get fields once
                    description = record.get('description_plain', '').strip()
                    ticket_type = record.get('ticket_type_id_name')
                    ticket_name = record.get('name')
                    
                    # Early exit conditions
                    if not description:
                        empty_descriptions += 1
                        continue
                    
                    if not contains_english_word(description):
                        no_format += 1
                        continue
                    
                    if not ticket_type:
                        no_type += 1
                        continue
                    
                    if not ticket_name:
                        no_name += 1
                        continue
                    
                    # Clean and prepare record
                    record['name'] = PARENTHESES_PATTERN.sub('', ticket_name).strip()
                    
                    # Add to batch
                    batch.append(json.dumps(record, ensure_ascii=False) + '\n')
                    cleaned_records += 1
                    
                    # Write batch when it reaches batch_size
                    if len(batch) >= batch_size:
                        outfile.writelines(batch)
                        batch = []

                except json.JSONDecodeError as e:
                    print(f"\nWarning: Invalid JSON on line {line_num}: {e}")
                    continue
            
            # Write remaining records
            if batch:
                outfile.writelines(batch)

        # Summary
        print(f"\n{'='*50}")
        print("Data Cleaning Summary")
        print(f"{'='*50}")
        print(f"Total records processed: {total_records:,}")
        print(f"Records with empty descriptions: {empty_descriptions:,}")
        print(f"Records with empty type: {no_type:,}")
        print(f"Records with empty name: {no_name:,}")
        print(f"Records with no description format: {no_format:,}")
        print(f"Cleaned records saved: {cleaned_records:,}")
        print(f"Output file: {output_file}")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"Error processing files: {e}")


if __name__ == "__main__":
    input_file = "tickets.jsonl"
    output_file = "tickets_cleaned.jsonl"

    clean_ticket_data(input_file, output_file)