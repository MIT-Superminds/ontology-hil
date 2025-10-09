#!/usr/bin/env python3
"""
CSV Row Processing Template with LLM Queries
============================================
A flexible template for processing each row of a CSV file using LLM queries.
Configure the LLM prompt, input columns, and output column name as needed.

Features:
- Progress tracking with live updates
- Automatic backup creation
- Error handling and retry logic
- Configurable LLM queries
- Debug print statements
- Resume from interruption capability

Usage:
1. Configure the processing parameters in the main() function
2. Customize the create_llm_prompt() function for your specific use case
3. Run the script: python process_rows.py
"""

import pandas as pd
import os
import time
import json
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class CSVRowProcessor:
    """
    A configurable class for processing CSV rows with LLM queries.
    """
    
    def __init__(self, 
                 input_file: str,
                 output_column_name: str,
                 llm_model: str = "gpt-4o",
                 batch_size: int = 1,
                 delay_between_calls: float = 1.0,
                 max_retries: int = 3):
        """
        Initialize the CSV row processor.
        
        Args:
            input_file: Path to the input CSV file
            output_column_name: Name of the new column to add with LLM results
            llm_model: OpenAI model to use for queries
            batch_size: Number of rows to process before saving (set to 1 for live updates)
            delay_between_calls: Delay in seconds between LLM API calls
            max_retries: Maximum number of retries for failed API calls
        """
        self.input_file = input_file
        self.output_column_name = output_column_name
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.delay_between_calls = delay_between_calls
        self.max_retries = max_retries
        
        # Set up OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)
        
        # Generate backup and output file paths
        self.backup_file = self._generate_backup_path()
        self.output_file = input_file  # Update the same file (after backup)
        
        print(f"🔧 CSV Row Processor initialized")
        print(f"   Input file: {self.input_file}")
        print(f"   Output column: {self.output_column_name}")
        print(f"   Model: {self.llm_model}")
        print(f"   Backup file: {self.backup_file}")
    
    def _generate_backup_path(self) -> str:
        """Generate a unique backup file path."""
        base_name = os.path.splitext(self.input_file)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{base_name}_backup_{timestamp}.csv"
        
        # Ensure backup doesn't already exist
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{base_name}_backup_{timestamp}_{counter}.csv"
            counter += 1
        
        return backup_path
    
    def create_backup(self) -> bool:
        """Create a backup of the original file."""
        try:
            print(f"📁 Creating backup: {self.backup_file}")
            df = pd.read_csv(self.input_file)
            df.to_csv(self.backup_file, index=False, encoding='utf-8')
            print(f"✅ Backup created successfully with {len(df)} rows")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def create_llm_prompt(self, row: pd.Series, row_index: int) -> Dict[str, str]:
        """
        Create the LLM prompt based on the row data.
        
        CUSTOMIZE THIS METHOD for your specific use case.
        
        Args:
            row: The pandas Series representing the current row
            row_index: The index of the current row
            
        Returns:
            Dictionary with 'system' and 'user' keys for the prompt
        """
        
        task = row.get('Task', '')
        verb = row.get('Verb', '')
        obj = row.get('Object', '')
        
        system_prompt = (
        "You are an expert at correcting O*NET-style ATOMIC tasks of the form VERB + OBJECT.\n\n"
        "Primary Goal:\n"
        "- Produce ONE concrete OBJECT head noun (or very short noun phrase) that makes the task atomic.\n"
        "- Do NOT change the VERB text, casing, or morphology.\n\n"
        "Hard Constraints (enforced):\n"
        "- OBJECT must NOT contain coordination or lists: no commas, 'and', 'or', '/', or '&'.\n"
        "- Prefer 1–2 words (e.g., 'Read Diagram'). Use 3+ words ONLY when REQUIRED to preserve meaning.\n"
        "- Use singular nouns (Blueprint, Diagram, Specification).\n"
        "- Title Case the final pair: '<Verb> <Object>'.\n\n"
        "Atomicity:\n"
        "- Each CSV row denotes ONE atomic task. An atomic task is a single verb-object pair. Even if the whole O*Net task has more than one object, your job is to focus on the single verb-object pair in the row.\n"
        "Normalization (prevent duplicates):\n"
        "- Collapse near-duplicates under umbrella heads when specificity is not essential:\n"
        "  -- scientific/current/technical Literature → Literature\n"
        "  -- operating/technical/repair Manual → Manual\n"
        "  -- engineering/technical Drawing → Drawing (or Diagram if clearer)\n"
        "- Prefer these canonical heads where applicable: Blueprint, Drawing, Diagram, Schematic, Specification, Work Order, Manual, Plan, Map, Chart, Report, Literature, Instruction, Procedure, Policy, Order, Invoice, Prescription, Schedule.\n\n"
        "Role Complements (rare, allowed):\n"
        "- Some verbs encode roles, not objects (Serve/Act/Work). For these, you MAY use 'as <Role>' to preserve meaning:\n"
        "  -- 'Serve as Advisor' (correct) vs 'Serve Advisor' (incorrect).\n"
        "- Do NOT introduce other prepositions or clauses.\n\n"
        "Output Format (STRICT):\n"
        "Return ONLY a single JSON object (no extra text, no code fences):\n"
        '{\"action\":\"correct_object\"|\"no_change\",\"final_verb_object\":\"<Verb> <Object>\",\"reason\":\"<≤20 words>\"}'  # noqa: E501\n\n"
        "Examples (correct → incorrect):\n"
        "Read Diagram → Read Schematics, Diagrams, or Technical Orders\n"
        "Read Literature → Read Scientific Literature, Read Current Literature\n"
        "Interpret Blueprint → Interpret Blueprints and Drawings\n"
        "Review Report → Review Reports/Documentation\n"
        "Serve as Advisor → Serve Advisor\n"
        )

        user_prompt = (
        f"Correct this atomic task row.\n\n"
        f"Task: {task}\n"
        f"Verb: {verb}\n"
        f"Object: {obj}\n\n"
        "Return ONLY the JSON as specified by the system message.")
    
        return {
            'system': system_prompt,
            'user': user_prompt
        }
    
    def call_llm(self, system_prompt: str, user_prompt: str, attempt: int = 1) -> str:
        """
        Make an LLM API call with retry logic.
        
        Args:
            system_prompt: The system instruction
            user_prompt: The user prompt
            attempt: Current attempt number (for retry logic)
            
        Returns:
            The LLM response text
        """
        try:
            print(f"    🤖 LLM API call (attempt {attempt})")
            
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent results
                timeout=30
            )
            
            result = response.choices[0].message.content.strip()
            print(f"    ✅ LLM response: {result[:100]}{'...' if len(result) > 100 else ''}")
            return result
            
        except Exception as e:
            print(f"    ❌ LLM API error (attempt {attempt}): {e}")
            
            if attempt < self.max_retries:
                delay = attempt * 2  # Exponential backoff
                print(f"    ⏱️ Retrying in {delay} seconds...")
                time.sleep(delay)
                return self.call_llm(system_prompt, user_prompt, attempt + 1)
            else:
                print(f"    💥 Max retries ({self.max_retries}) reached")
                return f"API_ERROR: {str(e)}"
    
    def process_row(self, row: pd.Series, row_index: int) -> str:
        """
        Process a single row and return the result.
        
        Args:
            row: The pandas Series representing the current row
            row_index: The index of the current row
            
        Returns:
            The processing result as a string
        """
        try:
            print(f"  📝 Processing row {row_index}")
            
            # Debug: Print relevant row information
            if 'Task' in row:
                print(f"      Task: {str(row['Task'])[:80]}{'...' if len(str(row['Task'])) > 80 else ''}")
            if 'Verb' in row:
                print(f"      Verb: {row['Verb']}")
            if 'Object' in row:
                print(f"      Object: {row['Object']}")
            
            # Create the LLM prompt
            prompt_data = self.create_llm_prompt(row, row_index)
            
            # Call the LLM
            result = self.call_llm(prompt_data['system'], prompt_data['user'])

            # If direct-obj changed, extract final string. If not, return original result.
            try:
                import json
                data = json.loads(result)
                final = data.get("final_verb_object") or result
                # Log action and reason
                if isinstance(data, dict):
                    if 'action' in data:
                        print(f"      • Action: {data['action']}")
                    if 'reason' in data:
                        print(f"      • Reason: {data['reason']}")
                print(f"      ✅ Final: {final}")
                return final
            except Exception:
                print(f"      ✅ Result: {result}")
                return result
            
            print(f"      ✅ Result: {result}")
            return result
            
        except Exception as e:
            error_msg = f"PROCESSING_ERROR: {str(e)}"
            print(f"      ❌ Processing error: {e}")
            return error_msg
    
    def process_csv(self) -> bool:
        """
        Process the entire CSV file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup first
            if not self.create_backup():
                print("❌ Failed to create backup. Aborting for safety.")
                return False
            
            # Load the CSV
            print(f"📊 Loading CSV: {self.input_file}")
            df = pd.read_csv(self.input_file)
            print(f"   Loaded {len(df)} rows with columns: {list(df.columns)}")
            
            # Add output column if it doesn't exist
            if self.output_column_name not in df.columns:
                df[self.output_column_name] = ""
                print(f"   Added new column: {self.output_column_name}")
            
            # Find rows that need processing (empty or null in output column)
            mask = df[self.output_column_name].isna() | (df[self.output_column_name] == "")
            rows_to_process = df[mask]

            # Test: Limit number of rows processed
            # test_limit = 10
            # print(f"💡 Test: only processing {test_limit} rows")
            # rows_to_process = rows_to_process.head(test_limit)
            
            if len(rows_to_process) == 0:
                print("✅ All rows already processed!")
                return True
            
            print(f"🔄 Found {len(rows_to_process)} rows to process")
            print("=" * 60)
            
            # Process rows with progress bar
            processed_count = 0
            total_to_process = len(rows_to_process)
            
            with tqdm(total=total_to_process, desc="Processing rows", unit="row") as pbar:
                for idx in rows_to_process.index:
                    row = df.loc[idx]
                    
                    print(f"\n🔍 Processing batch {processed_count + 1}/{total_to_process}")
                    
                    # Process the row
                    result = self.process_row(row, idx)
                    
                    # Update the dataframe
                    df.at[idx, self.output_column_name] = result
                    processed_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'Current': f"Row {idx}",
                        'Status': result[:20] + "..." if len(result) > 20 else result
                    })
                    
                    # Save progress after each batch (live updates)
                    if processed_count % self.batch_size == 0:
                        print(f"    💾 Saving progress ({processed_count}/{total_to_process} completed)")
                        df.to_csv(self.output_file, index=False, encoding='utf-8')
                    
                    # Delay between API calls
                    if self.delay_between_calls > 0:
                        time.sleep(self.delay_between_calls)
            
            # Final save
            print(f"\n💾 Saving final results to {self.output_file}")
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            
            # Print summary
            self.print_summary(df)
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing CSV: {e}")
            return False
    
    def print_summary(self, df: pd.DataFrame):
        """Print a summary of the processing results."""
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        
        total_rows = len(df)
        processed_rows = len(df[df[self.output_column_name].notna() & (df[self.output_column_name] != "")])
        
        print(f"Total rows: {total_rows}")
        print(f"Processed rows: {processed_rows}")
        print(f"Completion rate: {processed_rows/total_rows*100:.1f}%")
        
        # Count different types of results
        if processed_rows > 0:
            result_counts = df[self.output_column_name].value_counts()
            error_count = len(df[df[self.output_column_name].str.contains('ERROR', case=False, na=False)])
            
            print(f"\nResult distribution:")
            for result, count in result_counts.head(10).items():
                print(f"  {result}: {count}")
            
            if len(result_counts) > 10:
                print(f"  ... and {len(result_counts) - 10} more unique results")
            
            print(f"\nErrors: {error_count}")
            print(f"Success rate: {(processed_rows - error_count)/processed_rows*100:.1f}%")
        
        print(f"\n✅ Results saved to: {self.output_file}")
        print(f"📁 Backup available at: {self.backup_file}")


def main():
    """
    Main function - Configure your processing parameters here.
    """
    print("🚀 CSV Row Processing with LLM Queries")
    print("=" * 50)
    
    # CONFIGURATION - Modify these parameters for your use case
    config = {
        'input_file': '0926_onet_classifications.csv',  # Input CSV file
        'output_column_name': 'v2.2',           # Name of new column to add
        'llm_model': 'gpt-4o',                          # OpenAI model to use
        'batch_size': 1,                                # Save after each row (live updates)
        'delay_between_calls': 1.0,                     # Delay between API calls (seconds)
        'max_retries': 3                                # Max retries for failed calls
    }
    
    # Validate input file exists
    if not os.path.exists(config['input_file']):
        print(f"❌ Error: Input file '{config['input_file']}' not found!")
        print("   Available CSV files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"   - {file}")
        return 1
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("   Please set your OpenAI API key in a .env file or environment variable.")
        return 1
    
    try:
        # Initialize processor
        processor = CSVRowProcessor(**config)
        
        # Process the CSV
        success = processor.process_csv()
        
        if success:
            print("\n🎉 Processing completed successfully!")
            return 0
        else:
            print("\n💥 Processing failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Processing interrupted by user.")
        print("   Progress has been saved. You can resume by running the script again.")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
