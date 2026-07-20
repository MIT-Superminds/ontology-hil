#!/usr/bin/env python3
"""
Interactive WordNet Synset Explorer
===================================
A terminal-based tool for exploring WordNet synsets, definitions, synonyms, and hypernym paths.

Usage: python wordnet.py
Then enter synsets in the format: word.pos.xx (e.g., dog.n.01, run.v.02)
"""

import sys
try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Error: NLTK library not found.")
    print("Please install it using: pip install nltk")
    sys.exit(1)

import re
from typing import Optional, List


class WordNetExplorer:
    def __init__(self):
        """Initialize the WordNet explorer and download required data if needed."""
        self._ensure_wordnet_data()
        
    def _ensure_wordnet_data(self):
        """Download WordNet data if not already available."""
        try:
            # Test if WordNet data is available
            wn.synsets('test')
        except LookupError:
            print("Downloading WordNet data (one-time setup)...")
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)  # For multilingual support
                print("✓ WordNet data downloaded successfully!")
            except Exception as e:
                print(f"Error downloading WordNet data: {e}")
                print("Please check your internet connection and try again.")
                sys.exit(1)
    
    def parse_synset_name(self, synset_name: str) -> Optional[object]:
        """
        Parse synset name and return synset object if valid.
        
        Args:
            synset_name: String in format "word.pos.xx" (e.g., "dog.n.01")
            
        Returns:
            Synset object if valid, None otherwise
        """
        # Validate format using regex
        pattern = r'^[a-zA-Z_-]+\.[nvars]\.\d+$'
        if not re.match(pattern, synset_name.strip()):
            return None
            
        try:
            synset = wn.synset(synset_name.strip())
            return synset
        except Exception:
            return None
    
    def get_hypernym_path(self, synset) -> List[str]:
        """
        Get the hypernym path from the synset to the root.
        
        Args:
            synset: WordNet synset object
            
        Returns:
            List of synset names forming the hypernym path
        """
        paths = synset.hypernym_paths()
        if not paths:
            return [synset.name()]
        
        # Get the longest path (most specific)
        longest_path = max(paths, key=len)
        return [s.name() for s in longest_path]
    
    def format_definition(self, synset) -> str:
        """Format the definition with examples if available."""
        definition = synset.definition()
        examples = synset.examples()
        
        result = f"Definition: {definition}"
        if examples:
            result += f"\nExamples: {'; '.join(examples)}"
        return result
    
    def get_synonyms(self, synset) -> List[str]:
        """Get all lemma names (synonyms) for the synset."""
        return [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
    
    def display_synset_info(self, synset):
        """Display comprehensive information about a synset."""
        print(f"\n{'='*60}")
        print(f"SYNSET: {synset.name()}")
        print(f"POS: {synset.pos()} ({self._pos_full_name(synset.pos())})")
        print(f"{'='*60}")
        
        # Definition and examples
        print(f"\n📖 {self.format_definition(synset)}")
        
        # Synonyms
        synonyms = self.get_synonyms(synset)
        print(f"\n🔗 Synonyms: {', '.join(synonyms)}")
        
        # Hypernyms (direct parent concepts)
        hypernyms = synset.hypernyms()
        if hypernyms:
            print(f"\n⬆️  Direct Hypernyms:")
            for hyp in hypernyms:
                print(f"   • {hyp.name()}: {hyp.definition()}")
        
        # Hyponyms (direct child concepts) - show first few
        hyponyms = synset.hyponyms()
        if hyponyms:
            print(f"\n⬇️  Direct Hyponyms ({len(hyponyms)} total):")
            for hyp in hyponyms[:5]:  # Show first 5
                print(f"   • {hyp.name()}: {hyp.definition()}")
            if len(hyponyms) > 5:
                print(f"   ... and {len(hyponyms) - 5} more")
        
        # Full hypernym path
        hypernym_path = self.get_hypernym_path(synset)
        if len(hypernym_path) > 1:
            # Compact path format
            path_string = " > ".join(hypernym_path)
            print(f"\n🔗 Hypernym Path: {path_string}")
            
            # Detailed hierarchical view
            print(f"\n🗂️  Hypernym Path (specific → general):")
            for i, path_synset in enumerate(hypernym_path):
                indent = "   " * i
                try:
                    path_obj = wn.synset(path_synset)
                    print(f"{indent}{'└─' if i > 0 else ''}📁 {path_synset}: {path_obj.definition()}")
                except:
                    print(f"{indent}{'└─' if i > 0 else ''}📁 {path_synset}")
        
        # Similar synsets (if any)
        similar = synset.similar_tos()
        if similar:
            print(f"\n🔄 Similar synsets:")
            for sim in similar[:3]:  # Show first 3
                print(f"   • {sim.name()}: {sim.definition()}")
        
        print(f"\n{'='*60}")
    
    def _pos_full_name(self, pos: str) -> str:
        """Convert POS tag to full name."""
        pos_names = {
            'n': 'Noun',
            'v': 'Verb', 
            'a': 'Adjective',
            'r': 'Adverb',
            's': 'Adjective Satellite'
        }
        return pos_names.get(pos, pos)
    
    def search_word_synsets(self, word: str):
        """Search and display all synsets for a given word."""
        synsets = wn.synsets(word)
        if not synsets:
            print(f"❌ No synsets found for '{word}'")
            return
        
        print(f"\n🔍 Found {len(synsets)} synset(s) for '{word}':")
        print("-" * 50)
        
        for i, synset in enumerate(synsets, 1):
            synonyms = ', '.join(self.get_synonyms(synset)[:3])  # First 3 synonyms
            if len(self.get_synonyms(synset)) > 3:
                synonyms += "..."
            
            print(f"{i}. {synset.name()} ({self._pos_full_name(synset.pos())})")
            print(f"   Definition: {synset.definition()}")
            print(f"   Synonyms: {synonyms}")
            print()
    
    def run(self):
        """Main interactive loop."""
        print("🔤 WordNet Synset Explorer")
        print("=" * 40)
        print("\nCommands:")
        print("• Enter synset name (e.g., 'dog.n.01', 'run.v.02')")
        print("• Enter word to see all synsets (e.g., 'dog', 'run')")
        print("• Type 'help' for more information")
        print("• Type 'quit' or 'exit' to leave")
        print("\n" + "=" * 40)
        
        while True:
            try:
                user_input = input("\n🔍 Enter synset or word: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Check if input looks like a synset name
                if '.' in user_input and len(user_input.split('.')) == 3:
                    synset = self.parse_synset_name(user_input)
                    if synset:
                        self.display_synset_info(synset)
                    else:
                        print(f"❌ Invalid synset format or synset not found: '{user_input}'")
                        print("   Expected format: word.pos.xx (e.g., dog.n.01)")
                        print("   pos can be: n (noun), v (verb), a (adjective), r (adverb)")
                else:
                    # Search for word synsets
                    self.search_word_synsets(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Please try again or type 'help' for assistance.")
    
    def _show_help(self):
        """Display help information."""
        print("\n" + "=" * 50)
        print("📚 HELP - WordNet Synset Explorer")
        print("=" * 50)
        print("\n🔍 SEARCH MODES:")
        print("1. Synset Search: Enter exact synset name")
        print("   Format: word.pos.number")
        print("   Example: dog.n.01, run.v.02, happy.a.01")
        print("\n2. Word Search: Enter any word")
        print("   Example: dog, run, happy")
        print("   Shows all available synsets for that word")
        
        print("\n📝 POS TAGS:")
        print("   n = noun, v = verb, a = adjective, r = adverb")
        
        print("\n📖 OUTPUT INCLUDES:")
        print("   • Definition and examples")
        print("   • Synonyms (alternative words)")
        print("   • Hypernyms (broader concepts)")
        print("   • Hyponyms (more specific concepts)")
        print("   • Full hypernym path (concept hierarchy)")
        
        print("\n⌨️  COMMANDS:")
        print("   help    - Show this help")
        print("   quit    - Exit the program")
        print("   exit    - Exit the program")
        print("   Ctrl+C  - Exit the program")
        print("=" * 50)


def main():
    """Main entry point."""
    try:
        explorer = WordNetExplorer()
        explorer.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
