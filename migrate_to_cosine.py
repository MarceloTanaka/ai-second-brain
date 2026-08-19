import chromadb

db_client = chromadb.PersistentClient("./chroma_storage")

# 1. Extract all the old data from the collection before deleting it
old_collection = db_client.get_collection(name="developer_brain")
old_data = old_collection.get()

print(f"Found {len(old_data['ids'])} entries to migrate.")

# 2. Delete the old collection with the default distance function
db_client.delete_collection(name="developer_brain")

# 3. Recreate the collection with cosine distance
new_collection = db_client.get_or_create_collection(
    name="developer_brain",
    configuration={
        "hnsw":{
            "space": "cosine"
        }
    }
)

# 4. Re-insert the extracted data into the new collection
if old_data["ids"]:
    new_collection.add(
        ids=old_data["ids"],
        documents=old_data["documents"],
        metadatas=old_data["metadatas"]
    )

print(f"Migration complete. {new_collection.count()} entries now in the collection.")