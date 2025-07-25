from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

def create_vector(chunks):
    """
    Create vector embeddings for the given text chunks using a pre-trained model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    client = chromadb.Client()
    collection = client.create_collection(name="mahabharata")

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[str(i)]
        )
    
    return collection, model

def get_answer(question, collection, model, top_k=3):
    """
    Retrieve the most relevant chunk for a given question using a pre-trained model.
    """
    query_embedding = model.encode([question][0])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    context = "\n".join(results['documents'][0])

    prompt = f"Context:\n{context}\n\nQ: {question}\nA:"

    generator = pipeline("text-generation", model="gpt2")
    response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)

    return response[0]['generated_text']