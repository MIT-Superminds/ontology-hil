#!/usr/bin/env python3
"""
Map O*Net tasks into the hierarchy based on synsets.
Adds Verb-Object pairs and associated tasks under nodes containing matching synsets.
"""

import json
import csv
import re
from typing import Dict, List, Set, Any
from collections import defaultdict, Counter
import datetime


class ONetHierarchyMapper:
    def __init__(self, input_csv_file: str = "0926_onet_classifications.csv"):
        """
        Initialize the O*Net Hierarchy Mapper.
        
        Args:
            input_csv_file: Path to the input CSV file with O*Net classifications
        """
        self.input_csv_file = input_csv_file
        self.onet_data = {}  # synset -> verb_object -> [tasks]
        self.hierarchy = {}
        self.task_description_to_id = {}
        self.integrated_tasks = set()

        # D.O. and Verb Complement - Tracking changes
        self.label_changes = []  # list of dicts with details per change
        self.fix_proposed_count = 0
        self.fix_applied_count = 0
        self.fix_ignored_count = 0  # e.g., verb mismatch or empty obj

        
        print(f"🔧 O*Net Hierarchy Mapper initialized")
        print(f"   Input CSV file: {self.input_csv_file}")
        print(f"   Integration mode: Synset-based task mapping")
        print(f"   Output structure: Atomic Tasks + Specializations")
        
    def load_onet_data(self, csv_path: str = None):

        """Load and process O*Net CSV data."""
        if csv_path is None:
            csv_path = self.input_csv_file
            
        print(f"📊 Loading O*Net data from {csv_path}...")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                task_id = row['Task ID'].strip()
                task = row['Task'].strip()
                verb = row['Verb'].strip()
                obj_singularized = row['Object_Singularized'].strip()
                synset = row['Synset'].strip().lower()  # Normalize to lowercase
                
                if not all([task_id, task, verb, obj_singularized, synset]):
                    continue

                # D.O. and Verb Complement - Track original
                orig_obj = obj_singularized
                orig_label = f"{verb.title()} {orig_obj.title()}"

                # D.O. and Verb Complement - Prefer corrected atomic-task label if present & sane
                fixed = (row.get('v2.2') or "").strip()
                if fixed:
                    self.fix_proposed_count += 1
                    parts = fixed.split()
                    if parts:
                        fixed_verb = parts[0].strip()
                        fixed_obj  = " ".join(parts[1:]).strip()
                        if fixed_verb.lower() == verb.lower() and fixed_obj:
                            # Apply fix
                            obj_singularized = fixed_obj
                            self.fix_applied_count += 1

                            new_label = f"{verb.title()} {obj_singularized.title()}"
                            if new_label != orig_label:
                                self.label_changes.append({
                                    "task_id": task_id,
                                    "task": task,
                                    "verb": verb,
                                    "old_object": orig_obj,
                                    "new_object": obj_singularized,
                                    "old_label": orig_label,
                                    "new_label": new_label,
                                    "reason": "v2.2"
                                })
                                print(f"✏️  Label fix: {orig_label}  →  {new_label}  (Task {task_id})")
                        else:
                            self.fix_ignored_count += 1  # verb mismatch or empty obj

                # Prefer corrected atomic-task label if present & sane
                #fixed = (row.get('Fix_Direct_Obj') or "").strip()
                #if fixed:
                #    parts = fixed.split()
                #    if parts:
                #        fixed_verb = parts[0].strip()
                #        fixed_obj  = " ".join(parts[1:]).strip()
               #         if fixed_verb.lower() == verb.lower() and fixed_obj:
                #            obj_singularized = fixed_obj

                # Create verb-object pair
                verb_object = f"{verb.title()} {obj_singularized.title()}"
                
                # Store task under synset and verb-object
                if synset not in self.onet_data:
                    self.onet_data[synset] = defaultdict(list)
                
                # Format task with ID
                formatted_task = f"(O*Net) {task_id} - {task}"
                self.onet_data[synset][verb_object].append(formatted_task)
                
                # Store mapping from task description to ID for reporting
                self.task_description_to_id[task] = task_id
                self.task_description_to_id[formatted_task] = task_id
        
        # Print statistics
        total_synsets = len(self.onet_data)
        total_verb_objects = sum(len(vo_dict) for vo_dict in self.onet_data.values())
        total_tasks = sum(len(tasks) for vo_dict in self.onet_data.values() 
                         for tasks in vo_dict.values())
        
        print(f"✅ Loaded {total_tasks} tasks across {total_verb_objects} verb-object pairs in {total_synsets} synsets")

        # D.O. and Verb Complement - label-fix summary
        if self.fix_proposed_count > 0:
            print("\n✳️  Label Correction Summary (from v2.2)")
            print(f"- Proposed fixes in CSV: {self.fix_proposed_count}")
            print(f"- Applied fixes: {self.fix_applied_count}")
            print(f"- Ignored fixes (verb mismatch/empty): {self.fix_ignored_count}")
            # Unique (old_label -> new_label) pairs
            unique_pairs = {(c['old_label'], c['new_label']) for c in self.label_changes}
            print(f"- Unique label changes: {len(unique_pairs)}")
            print("- Examples:")
            for i, (old_lbl, new_lbl) in enumerate(list(unique_pairs)[:10], start=1):
                print(f"  {i:>2}. {old_lbl} → {new_lbl}")
    
    def load_hierarchy(self, hierarchy_path: str):
        """Load the hierarchy JSON file."""
        print(f"📁 Loading hierarchy from {hierarchy_path}...")
        
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            self.hierarchy = json.load(f)
        
        print("✅ Hierarchy loaded successfully")
    
    def extract_synsets_from_text(self, text: str) -> Set[str]:
        """Extract synset patterns from text like 'Title (Synset.v.01)' or 'Title (Synset.v.01, Other.v.02)'."""
        synsets = set()
        
        # Pattern to match synsets in parentheses (case-insensitive)
        pattern = r'\(([^)]*\.v\.\d+[^)]*)\)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            # First split by commas, then by spaces to handle missing commas
            parts = match.split(',')
            for part in parts:
                # Further split by spaces to catch synsets that aren't comma-separated
                space_parts = part.split()
                for space_part in space_parts:
                    synset = space_part.strip().lower()  # Normalize to lowercase
                    if synset and '.v.' in synset:
                        synsets.add(synset)
        
        return synsets
    
    def process_node(self, node: Any, path: List[str] = None) -> Any:
        """Recursively process hierarchy nodes to add O*Net tasks."""
        if path is None:
            path = []
        
        if not isinstance(node, dict):
            return node
        
        updated_node = {}
        
        for key, value in node.items():
            current_path = path + [key]
            
            # Extract synsets from the current key
            synsets = self.extract_synsets_from_text(key)
            
            # Process children first
            processed_value = self.process_node(value, current_path)
            
            # If this key contains synsets that match our O*Net data, add atomic tasks
            if synsets:
                matching_synsets = synsets.intersection(self.onet_data.keys())
                
                if matching_synsets:
                    print(f"Found matching synsets at: {' → '.join(current_path)}")
                    print(f"  Synsets: {matching_synsets}")
                    
                    # If processed_value is empty dict, initialize structure
                    if not processed_value:
                        processed_value = {}
                    
                    # Ensure we have the required structure
                    if "(Atomic Tasks)" not in processed_value:
                        processed_value["(Atomic Tasks)"] = {}
                    if "(Specializations)" not in processed_value:
                        processed_value["(Specializations)"] = {}
                    
                    # Move existing children to Specializations (except Atomic Tasks)
                    for child_key, child_value in list(processed_value.items()):
                        if child_key not in ["(Atomic Tasks)", "(Specializations)"]:
                            processed_value["(Specializations)"][child_key] = child_value
                            del processed_value[child_key]
                    
                    # Add atomic tasks from matching synsets
                    for synset in matching_synsets:
                        verb_objects = self.onet_data[synset]
                        for verb_object, tasks in verb_objects.items():
                            if verb_object not in processed_value["(Atomic Tasks)"]:
                                processed_value["(Atomic Tasks)"][verb_object] = []
                            
                            # Add unique tasks
                            for task in tasks:
                                if task not in processed_value["(Atomic Tasks)"][verb_object]:
                                    processed_value["(Atomic Tasks)"][verb_object].append(task)
                                    # Extract task ID from the formatted task
                                    task_id = self.task_description_to_id.get(task, None)
                                    if task_id:
                                        self.integrated_tasks.add(task_id)
                    
                    print(f"  Added {len([vo for vo in processed_value['(Atomic Tasks)']])} verb-object pairs")
            
            updated_node[key] = processed_value
        
        return updated_node
    
    def process_hierarchy(self):
        """Process the entire hierarchy to add O*Net tasks."""
        print("🔄 Processing hierarchy to add O*Net tasks...")
        self.hierarchy = self.process_node(self.hierarchy)
        print("✅ Hierarchy processing complete")
    
    def save_hierarchy(self, output_path: str):
        """Save the updated hierarchy."""
        print(f"💾 Saving updated hierarchy to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.hierarchy, f, indent=2, ensure_ascii=False)
        
        print("✅ Hierarchy saved successfully")
    
    def extract_hierarchy_synsets(self, node=None):
        """Extract all synsets that exist in the hierarchy."""
        if node is None:
            node = self.hierarchy
            
        synsets = set()
        
        if isinstance(node, dict):
            for key, value in node.items():
                # Extract synsets from key
                key_synsets = self.extract_synsets_from_text(key)
                synsets.update(key_synsets)
                
                # Recursively process children
                if isinstance(value, dict):
                    child_synsets = self.extract_hierarchy_synsets(value)
                    synsets.update(child_synsets)
        
        return synsets
    
    def generate_integration_report(self):
        """Generate comprehensive integration report."""
        print("\n📄 GENERATING INTEGRATION REPORT")
        print("=" * 45)
        
        # Load CSV data for analysis
        csv_data = []
        try:
            with open(self.input_csv_file, 'r', encoding='utf-8') as f:
                csv_data = list(csv.DictReader(f))
        except FileNotFoundError:
            print(f"Warning: {self.input_csv_file} not found for report generation")
            return
        
        # Extract hierarchy data first
        hierarchy_synsets = self.extract_hierarchy_synsets()
        
        # Analyze CSV structure - count unique descriptions and atomic tasks, not IDs
        unique_task_ids = set(row['Task ID'].strip() for row in csv_data if row['Task ID'].strip())
        unique_task_descriptions = set(row['Task'].strip() for row in csv_data if row['Task'].strip())
        unique_atomic_tasks = set()
        csv_synsets = set(row['Synset'].strip().lower() for row in csv_data if row['Synset'].strip())
        
        # Build set of unique atomic tasks from CSV
        for row in csv_data:
            verb = row['Verb'].strip()
            obj_sing = row['Object_Singularized'].strip()
            if verb and obj_sing:
                atomic_task = f"{verb.title()} {obj_sing.title()}"
                unique_atomic_tasks.add(atomic_task)
        
        # Track which task descriptions should be integrated based on synset matching
        expected_integrated_task_descriptions = set()
        expected_integrated_atomic_tasks = set()
        for row in csv_data:
            task_desc = row['Task'].strip()
            verb = row['Verb'].strip()
            obj_sing = row['Object_Singularized'].strip()
            synset = row['Synset'].strip().lower()
            
            if synset in hierarchy_synsets:
                if task_desc:
                    expected_integrated_task_descriptions.add(task_desc)
                if verb and obj_sing:
                    atomic_task = f"{verb.title()} {obj_sing.title()}"
                    expected_integrated_atomic_tasks.add(atomic_task)
        
        # Calculate metrics
        total_csv_rows = len(csv_data)
        unique_csv_task_descriptions = len(unique_task_descriptions)
        unique_csv_atomic_tasks = len(unique_atomic_tasks)
        total_csv_synsets = len(csv_synsets)
        hierarchy_synsets_count = len(hierarchy_synsets)
        matched_synsets = len(csv_synsets.intersection(hierarchy_synsets))
        missing_synsets = csv_synsets - hierarchy_synsets
        
        # Count integrated tasks by unique descriptions (not IDs)
        integrated_unique_task_descriptions = len(expected_integrated_task_descriptions)
        integrated_unique_atomic_tasks = len(expected_integrated_atomic_tasks)
        
        # Calculate missing counts
        missing_task_descriptions_count = unique_csv_task_descriptions - integrated_unique_task_descriptions
        missing_atomic_tasks_count = unique_csv_atomic_tasks - integrated_unique_atomic_tasks
        
        # Coverage percentages based on descriptions/atomic tasks, not IDs
        task_desc_coverage_percent = (integrated_unique_task_descriptions / unique_csv_task_descriptions) * 100 if unique_csv_task_descriptions > 0 else 0
        atomic_task_coverage_percent = (integrated_unique_atomic_tasks / unique_csv_atomic_tasks) * 100 if unique_csv_atomic_tasks > 0 else 0
        synset_coverage_percent = (matched_synsets / total_csv_synsets) * 100 if total_csv_synsets > 0 else 0
        
        # Count atomic tasks and full tasks for missing synsets
        missing_synset_stats = defaultdict(lambda: {'atomic_tasks': set(), 'full_tasks': set()})
        missing_atomic_tasks = set()  # Track unique verb-object pairs that are missing
        missing_full_tasks = set()  # Track unique full task descriptions that are missing
        
        for row in csv_data:
            synset = row['Synset'].strip().lower()
            if synset in missing_synsets:
                verb = row['Verb'].strip()
                obj_sing = row['Object_Singularized'].strip()
                task_desc = row['Task'].strip()
                
                atomic_task = f"{verb.title()} {obj_sing.title()}"
                missing_synset_stats[synset]['atomic_tasks'].add(atomic_task)
                missing_synset_stats[synset]['full_tasks'].add(task_desc)
                missing_atomic_tasks.add(atomic_task)  # Add to global missing atomic tasks
                missing_full_tasks.add(task_desc)  # Add to global missing full tasks
        
        # Generate report content
        report_lines = []
        report_lines.append("# O*Net Integration Report")
        report_lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Integration Summary
        report_lines.append("## Integration Summary")
        report_lines.append("")
        report_lines.append(f"- **Total CSV rows processed:** {total_csv_rows:,}")
        report_lines.append(f"- **Unique task descriptions in CSV:** {unique_csv_task_descriptions:,}")
        report_lines.append(f"- **Unique atomic tasks in CSV:** {unique_csv_atomic_tasks:,}")
        report_lines.append(f"- **Unique synsets in CSV:** {total_csv_synsets:,}")
        report_lines.append(f"- **Synsets present in hierarchy:** {hierarchy_synsets_count:,}")
        report_lines.append("")
        
        # Key Definitions
        report_lines.append("### Key Definitions")
        report_lines.append("")
        report_lines.append("- **Task**: Individual O*Net work activity (e.g., 'Measure temperature of water')")
        report_lines.append("- **Synset**: WordNet verb concept (e.g., 'measure.v.04') that groups related tasks")
        report_lines.append("- **Atomic Task**: Verb-Object pair (e.g., 'Measure Temperature') derived from tasks")
        report_lines.append("- **Expected integrable tasks**: Tasks whose synsets exist in the hierarchy")
        report_lines.append("- **Tasks with missing synsets**: Tasks whose synsets are NOT in the hierarchy")
        report_lines.append("")
        
        # Integration Results
        report_lines.append("## Integration Results")
        report_lines.append("")
        
        # Synset-Level Coverage
        report_lines.append("### Synset-Level Coverage")
        report_lines.append("")
        report_lines.append(f"- **Synsets found in hierarchy:** {matched_synsets} of {total_csv_synsets} ({synset_coverage_percent:.1f}%)")
        report_lines.append(f"- **Missing synsets:** {len(missing_synsets)} synsets not present in hierarchy")
        report_lines.append("")
        
        # Task Description Coverage
        report_lines.append("### Task Description Coverage")
        report_lines.append("")
        report_lines.append(f"- **Unique task descriptions successfully integrated:** {integrated_unique_task_descriptions:,} of {unique_csv_task_descriptions:,} ({task_desc_coverage_percent:.1f}%)")
        report_lines.append(f"- **Task descriptions blocked by missing synsets:** {missing_task_descriptions_count:,} descriptions cannot be integrated")
        report_lines.append("")
        
        # Atomic Task Coverage
        report_lines.append("### Atomic Task Coverage")
        report_lines.append("")
        report_lines.append(f"- **Unique atomic tasks successfully integrated:** {integrated_unique_atomic_tasks:,} of {unique_csv_atomic_tasks:,} ({atomic_task_coverage_percent:.1f}%)")
        report_lines.append(f"- **Atomic tasks blocked by missing synsets:** {missing_atomic_tasks_count:,} verb-object pairs cannot be integrated")
        report_lines.append("")
        
        # Missing Content Summary
        report_lines.append("### Missing Content Summary")
        report_lines.append("")
        report_lines.append(f"- **Missing synsets:** {len(missing_synsets):,} synsets")
        report_lines.append(f"- **Missing atomic tasks (Verb-Object pairs):** {len(missing_atomic_tasks):,} unique combinations")
        report_lines.append(f"- **Missing task descriptions:** {len(missing_full_tasks):,} unique descriptions")
        report_lines.append("")

        # Direct Obj and Verb Complement - Fixes Summary
        report_lines.append("## Label Correction Stats")
        report_lines.append("")
        report_lines.append(f"- **Proposed fixes in CSV (`v2.2`)**: {self.fix_proposed_count:,}")
        report_lines.append(f"- **Applied fixes**: {self.fix_applied_count:,}")
        report_lines.append(f"- **Ignored fixes (verb mismatch/empty)**: {self.fix_ignored_count:,}")
        unique_pairs = {(c['old_label'], c['new_label']) for c in self.label_changes}
        unique_tasks_changed = len({c['task_id'] for c in self.label_changes})
        report_lines.append(f"- **Unique label changes (old→new)**: {len(unique_pairs):,}")
        report_lines.append(f"- **Distinct tasks affected**: {unique_tasks_changed:,}")
        if unique_pairs:
            report_lines.append("")
            report_lines.append("### Examples of Old → New Label Changes")
            for i, (old_lbl, new_lbl) in enumerate(list(unique_pairs)[:25], start=1):
                report_lines.append(f"{i}. `{old_lbl}` → `{new_lbl}`")
            report_lines.append("")
        
        # Understanding the relationship
        # Check for task descriptions that appear with both missing and available synsets
        tasks_with_mixed_synsets = set()
        for task_desc in missing_full_tasks:
            if task_desc in expected_integrated_task_descriptions:
                tasks_with_mixed_synsets.add(task_desc)
        
        report_lines.append("### Understanding the Numbers")
        report_lines.append("")
        report_lines.append("**Key Distinction:**")
        report_lines.append(f"- **Task descriptions blocked by missing synsets: {missing_task_descriptions_count:,}** = Tasks that CANNOT be integrated at all")
        report_lines.append(f"- **Missing task descriptions: {len(missing_full_tasks):,}** = Tasks that appear with missing synsets (but may also appear with available synsets)")
        report_lines.append("")
        report_lines.append(f"**Mixed synset tasks: {len(tasks_with_mixed_synsets):,}** task descriptions appear with BOTH missing and available synsets.")
        report_lines.append("These tasks are successfully integrated via their available synsets, but also counted in the 'missing' category.")
        report_lines.append("")
        report_lines.append("**Relationship between synsets, atomic tasks, and full tasks:**")
        report_lines.append(f"- **{len(missing_synsets):,} missing synsets** prevent integration of related tasks")
        report_lines.append(f"- **{len(missing_full_tasks):,} unique full task descriptions** appear with missing synsets")
        report_lines.append(f"- **{len(missing_atomic_tasks):,} unique atomic tasks (verb-object pairs)** cannot be integrated")
        report_lines.append("")
        report_lines.append("**Why atomic tasks ≥ full tasks:** Each full task generates one verb-object pair,")
        report_lines.append("but the same verb-object pair can appear in multiple different task descriptions.")
        report_lines.append("")
        report_lines.append("**Example:** 'Pick_Up Item' (atomic task) appears in multiple full tasks:")
        report_lines.append("- 'Pick up medical supplies from storage room'")
        report_lines.append("- 'Pick up tools from workbench area'") 
        report_lines.append("- 'Pick up documents from filing cabinet'")
        report_lines.append("")

        # Missing Synsets Analysis
        if missing_synsets:
            report_lines.append("## Missing Synsets Analysis")
            report_lines.append("")
            report_lines.append(f"**Total missing synsets:** {len(missing_synsets)}")
            report_lines.append("")
            
            # Sort missing synsets by full task count
            sorted_missing = sorted(missing_synset_stats.items(), 
                                  key=lambda x: len(x[1]['full_tasks']), reverse=True)
            
            report_lines.append(f"**Example:** The missing synset `pick_up.v.02` alone blocks")
            pick_up_stats = missing_synset_stats.get('pick_up.v.02', {'full_tasks': set()})
            report_lines.append(f"{len(pick_up_stats['full_tasks'])} individual tasks from being integrated.")
            report_lines.append("")
            report_lines.append("| Synset | Atomic Tasks | Full Tasks Blocked |")
            report_lines.append("|--------|--------------|-------------------|")
            
            for synset, stats in sorted_missing:
                atomic_count = len(stats['atomic_tasks'])
                full_count = len(stats['full_tasks'])
                report_lines.append(f"| `{synset}` | {atomic_count} | {full_count} |")
            
            report_lines.append("")
            
            # Sample missing tasks for top synsets
            report_lines.append("### Sample Missing Tasks (Top 10 Synsets)")
            report_lines.append("")
            
            for synset, stats in sorted_missing[:10]:
                report_lines.append(f"#### {synset}")
                report_lines.append("")
                report_lines.append("**Atomic Tasks:**")
                for atomic_task in sorted(stats['atomic_tasks'])[:15]:  # Show up to 15
                    report_lines.append(f"- {atomic_task}")
                report_lines.append("")
        
        # Integration Status
        report_lines.append("## Integration Status")
        report_lines.append("")
        # Use task description coverage for overall status
        if task_desc_coverage_percent >= 90:
            status = "✅ Excellent"
        elif task_desc_coverage_percent >= 75:
            status = "🟡 Good"
        elif task_desc_coverage_percent >= 50:
            status = "🟠 Fair"
        else:
            status = "🔴 Needs Improvement"
        
        report_lines.append(f"**Overall Status:** {status}")
        report_lines.append(f"- Successfully integrated {task_desc_coverage_percent:.1f}% of unique task descriptions")
        report_lines.append(f"- Successfully integrated {atomic_task_coverage_percent:.1f}% of unique atomic tasks")
        report_lines.append(f"- Achieved {synset_coverage_percent:.1f}% synset coverage")
        
        if missing_synsets:
            report_lines.append("")
            report_lines.append("### Recommendations")
            report_lines.append("")
            report_lines.append(f"- Add {len(missing_synsets)} missing synsets to hierarchy")
            report_lines.append(f"- This would make {missing_task_descriptions_count:,} additional task descriptions available for integration")
            report_lines.append(f"- This would make {missing_atomic_tasks_count:,} additional atomic tasks available for integration")
            report_lines.append("- Focus on top missing synsets for maximum impact")
        
        # Write to file
        report_content = "\n".join(report_lines)
        
        with open('ONet_Integration_Report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Integration report saved to: ONet_Integration_Report.md")
        
        return {
            'task_description_coverage_percent': task_desc_coverage_percent,
            'atomic_task_coverage_percent': atomic_task_coverage_percent,
            'synset_coverage_percent': synset_coverage_percent,
            'missing_synsets_count': len(missing_synsets),
            'missing_task_descriptions_count': missing_task_descriptions_count,
            'missing_atomic_tasks_count': missing_atomic_tasks_count
        }
    
    def print_statistics(self):
        """Print processing statistics."""
        total_atomic_tasks = 0
        total_verb_objects = 0
        
        def count_atomic_tasks(node):
            nonlocal total_atomic_tasks, total_verb_objects
            if isinstance(node, dict):
                if "(Atomic Tasks)" in node:
                    atomic_tasks = node["(Atomic Tasks)"]
                    if isinstance(atomic_tasks, dict):
                        for verb_object, tasks in atomic_tasks.items():
                            total_verb_objects += 1
                            if isinstance(tasks, list):
                                total_atomic_tasks += len(tasks)
                
                for value in node.values():
                    count_atomic_tasks(value)
        
        count_atomic_tasks(self.hierarchy)
        
        print(f"\n📈 INTEGRATION STATISTICS:")
        print(f"- Total atomic tasks integrated: {total_atomic_tasks}")
        print(f"- Total verb-object pairs: {total_verb_objects}")
        print(f"- Unique tasks integrated: {len(self.integrated_tasks)}")
        print(f"- Synsets with O*Net data: {len(self.onet_data)}")


def main():
    """Main function to process O*Net tasks into hierarchy."""
    print("🚀 O*Net Hierarchy Integration Tool")
    print("=" * 50)
    
    # CONFIGURATION - Modify these parameters for your use case
    config = {
        'input_csv_file': '0926_onet_classifications.csv',      # Input CSV file with O*Net classifications
        'hierarchy_file': '0926_hierarchy.json',  # Input hierarchy JSON file
        'output_file': '0926_hierarchy_with_onet_v2.2'    # Output integrated hierarchy
    }
    
    # Validate input files exist
    import os
    if not os.path.exists(config['input_csv_file']):
        print(f"❌ Error: Input CSV file '{config['input_csv_file']}' not found!")
        print("   Available CSV files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"   - {file}")
        return 1
    
    if not os.path.exists(config['hierarchy_file']):
        print(f"❌ Error: Hierarchy file '{config['hierarchy_file']}' not found!")
        print("   Available JSON files:")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.json'):
                    print(f"   - {os.path.join(root, file)}")
        return 1
    
    try:
        # Initialize mapper with configurable input file
        mapper = ONetHierarchyMapper(input_csv_file=config['input_csv_file'])
        
        # Load data
        mapper.load_onet_data()  # Uses self.input_csv_file
        mapper.load_hierarchy(config['hierarchy_file'])
        
        # Process hierarchy
        mapper.process_hierarchy()
        
        # Save results
        mapper.save_hierarchy(config['output_file'])
        
        # Print statistics
        mapper.print_statistics()
        
        # Generate integration report
        mapper.generate_integration_report()
        
        print(f"\n🎉 Success! O*Net tasks have been integrated into the hierarchy.")
        print(f"📄 Output saved to: {config['output_file']}")
        print(f"📊 Report saved to: ONet_Integration_Report.md")
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
