from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="ADD_YOUR_PINECONE_API_KEY_HERE")

pc.indexes.create(
    name="simplyrag",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print(pc.list_indexes())