# Ontology Editor

A collection of tools for processing O*Net task data using LLM queries and integrating tasks into hierarchical ontology structures.

## Overview

This repository contains template code for analyzing and processing ontology task data through:
- **LLM-based row processing**: Apply custom AI queries to each row of CSV data
- **Hierarchy integration**: Map classified tasks into JSON ontology structures

## Files

### Data Files
- **`0926_onet_classifications.csv`** - O*Net task data with classifications (38,223 rows)
  - Contains: Task ID, Task description, Verb, Object, Normalized form, Verb Path, Synset, Object_Singularized
- **`0926_hierarchy.json`** - Hierarchical ontology structure for task integration

### Processing Scripts
- **`process_rows.py`** - Template for LLM-based CSV row processing
- **`map_to_json.py`** - Integrates classified tasks into hierarchy structure

## Quick Start

### Prerequisites
```bash
pip install pandas tqdm python-dotenv openai
```

Set up your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Using process_rows.py

This script processes each row of your CSV with customizable LLM queries.

#### 1. Basic Configuration
Edit the `main()` function to set your parameters:
```python
config = {
    'input_file': '0926_onet_classifications.csv',  # Your CSV file
    'output_column_name': 'LLM_Analysis',           # Name for new column
    'llm_model': 'gpt-4o',                          # OpenAI model
    'batch_size': 1,                                # Save frequency
    'delay_between_calls': 1.0,                     # API rate limiting
    'max_retries': 3                                # Error handling
}
```

#### 2. Customize the LLM Prompt
**This is the key step!** Edit the `create_llm_prompt()` method for your specific use case.

The method receives a pandas row and should return a dictionary with 'system' and 'user' prompts:

```python
def create_llm_prompt(self, row: pd.Series, row_index: int) -> Dict[str, str]:
    # Access any column from your CSV
    task = row.get('Task', '')
    verb = row.get('Verb', '')
    obj = row.get('Object', '')
    
    # Create your custom prompts
    system_prompt = '''Your system instruction here...'''
    user_prompt = f"Your user prompt using {task}, {verb}, {obj}..."
    
    return {'system': system_prompt, 'user': user_prompt}
```

#### 3. Example Use Cases

**Task Classification:**
```python
system_prompt = '''Classify tasks into: Physical, Information, Communication, Creative'''
user_prompt = f"Classify this task: {task}"
```

**Quality Assessment:**
```python
system_prompt = '''Rate verb-object alignment on 1-10 scale'''
user_prompt = f'Task: {task}\nVerb: {verb}\nObject: {obj}\nHow well does "{verb} {obj}" represent this task?'
```

**Information Extraction:**
```python
system_prompt = '''Extract the primary tool or equipment mentioned in the task'''
user_prompt = f"What tool/equipment is used in: {task}"
```

#### 4. Run the Script
```bash
python process_rows.py
```

The script will:
- ✅ Create automatic backups
- 📊 Show progress with live updates
- 💾 Save after each row (resumable)
- 🔄 Handle errors with retries
- 📈 Provide completion statistics

### Using map_to_json.py

Integrates your classified task data into the ontology hierarchy.

#### 1. Configuration
Edit the `main()` function:
```python
config = {
    'input_csv_file': '0926_onet_classifications.csv',
    'hierarchy_file': '0926_hierarchy.json',
    'output_file': '0926_hierarchy_with_onet.json'
}
```

#### 2. Run Integration
```bash
python map_to_json.py
```

This will:
- 🔄 Map tasks to hierarchy nodes based on synsets
- 📁 Create structured "(Atomic Tasks)" and "(Specializations)" sections
- 📊 Generate detailed integration report
- 💾 Output enhanced hierarchy JSON

## More Features of process_rows.py

### General
- **Live Progress Tracking**: Visual progress bar with current status
- **Automatic Backups**: Timestamped backups before processing
- **Resume Capability**: Skip already processed rows on restart
- **Error Handling**: Exponential backoff retry logic for API failures
- **Flexible Configuration**: Easy customization for different use cases
- **Debug Output**: Detailed logging of each processing step

### Custom Column Analysis
Process specific columns by modifying the prompt creation:
```python
# Multi-column analysis
task_id = row.get('Task ID', '')
normalized = row.get('Normalized', '')
synset = row.get('Synset', '')

user_prompt = f"""
Task ID: {task_id}
Task: {task}
Normalized: {normalized}
Synset: {synset}

Analyze the consistency between these fields...
"""
```

### Batch Processing
For large datasets, adjust batch processing:
```python
config = {
    'batch_size': 10,        # Save every 10 rows
    'delay_between_calls': 0.5,  # Faster processing
}
```

### Error Recovery
If processing fails, simply restart the script - it will resume from where it left off by checking for empty cells in your output column.

## Output

### process_rows.py Output
- Updates your CSV with a new column containing LLM responses
- Creates timestamped backup files
- Provides processing statistics and error counts

### map_to_json.py Output
- Enhanced hierarchy JSON with integrated tasks
- `ONet_Integration_Report.md` with detailed analysis
- Statistics on integration coverage and missing synsets

## Troubleshooting

- **API Errors**: Check your OpenAI API key and rate limits
- **File Not Found**: Verify file paths in configuration
- **Memory Issues**: Reduce batch_size for large files
- **Resume Issues**: Delete problematic output column to restart fresh
