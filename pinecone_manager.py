#!/usr/bin/env python3
"""
Pinecone Index Management Utility
- Clear all data from index
- Check index stats
- Reset index for new documents
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.services.embedding_service import EmbeddingService

class PineconeManager:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.pc = self.embedding_service.pc
        self.index = self.embedding_service.index
        self.index_name = settings.PINECONE_INDEX_NAME
    
    def get_index_stats(self):
        """Get current index statistics"""
        try:
            stats = self.index.describe_index_stats()
            print(f"📊 PINECONE INDEX STATS:")
            print(f"   • Index Name: {self.index_name}")
            print(f"   • Total Vectors: {stats.get('total_vector_count', 0)}")
            print(f"   • Index Dimension: {stats.get('dimension', 'Unknown')}")
            print(f"   • Index Fullness: {stats.get('index_fullness', 0):.2%}")
            
            # Show namespace info if available
            if 'namespaces' in stats:
                print(f"   • Namespaces: {list(stats['namespaces'].keys())}")
            
            return stats
        except Exception as e:
            print(f"❌ Error getting index stats: {str(e)}")
            return None
    
    def clear_all_data(self):
        """Clear all vectors from the index"""
        try:
            print("🧹 CLEARING ALL DATA FROM PINECONE INDEX...")
            
            # First, check current stats
            stats = self.get_index_stats()
            if not stats or stats.get('total_vector_count', 0) == 0:
                print("✅ Index is already empty!")
                return True
            
            print(f"⚠️  About to delete {stats['total_vector_count']} vectors")
            
            # Delete all vectors by deleting the entire namespace
            self.index.delete(delete_all=True)
            
            print("✅ All data cleared from Pinecone index!")
            
            # Verify deletion
            import time
            time.sleep(2)  # Wait for deletion to complete
            new_stats = self.get_index_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ Error clearing index: {str(e)}")
            return False
    
    def reset_index_for_new_document(self):
        """Complete reset for new document processing"""
        try:
            print("🔄 RESETTING INDEX FOR NEW DOCUMENT...")
            
            # Clear all existing data
            if self.clear_all_data():
                print("✅ Index reset complete! Ready for new document.")
                return True
            else:
                print("❌ Failed to reset index")
                return False
                
        except Exception as e:
            print(f"❌ Error resetting index: {str(e)}")
            return False
    
    def search_sample_data(self, query="test", k=5):
        """Search for sample data in the index"""
        try:
            print(f"🔍 SEARCHING FOR: '{query}'")
            
            # Create a dummy query vector (zeros) to see what's in the index
            dummy_vector = [0.0] * settings.VECTOR_DIMENSION
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=k,
                include_metadata=True
            )
            
            print(f"📋 Found {len(results.matches)} results:")
            
            for i, match in enumerate(results.matches):
                print(f"   {i+1}. ID: {match.id}")
                print(f"      Score: {match.score:.4f}")
                if match.metadata:
                    source = match.metadata.get('source', 'Unknown')
                    content = match.metadata.get('content', '')[:100]
                    print(f"      Source: {source}")
                    print(f"      Content: {content}...")
                print()
            
            return results.matches
            
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            return []

def main():
    """Main function with menu options"""
    manager = PineconeManager()
    
    print("🏗️  PINECONE INDEX MANAGER")
    print("=" * 50)
    
    while True:
        print("\n📋 OPTIONS:")
        print("1. 📊 Check Index Stats")
        print("2. 🔍 Search Sample Data")
        print("3. 🧹 Clear All Data")
        print("4. 🔄 Reset for New Document")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            manager.get_index_stats()
            
        elif choice == '2':
            manager.search_sample_data()
            
        elif choice == '3':
            confirm = input("⚠️  Are you sure you want to clear ALL data? (yes/no): ").strip().lower()
            if confirm == 'yes':
                manager.clear_all_data()
            else:
                print("❌ Operation cancelled")
                
        elif choice == '4':
            confirm = input("⚠️  Reset index for new document? This will clear all data! (yes/no): ").strip().lower()
            if confirm == 'yes':
                manager.reset_index_for_new_document()
            else:
                print("❌ Operation cancelled")
                
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main()
