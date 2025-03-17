import os
import chromadb
import locale
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer

# Ensure English locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Load Sentence Transformer Model
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# File paths for PRD+HLD and ISTQB Material
PRD_HLD_FILE = "PRDforstudentsAPIs.pdf"
ISTQB_FILE = "Foundations_of_Software_Testing.pdf"

def load_and_chunk_pdf(pdf_file, source_name, chunk_size=500, chunk_overlap=100):
    """Loads and chunks a PDF file into small text segments for ChromaDB indexing."""
    print(f"Loading and chunking: {pdf_file} ...")
    reader = PdfReader(pdf_file)
    all_text = "".join([page.extract_text() or "" for page in reader.pages])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(all_text)

    cleaned_chunks = [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    print(f"Total valid chunks from {pdf_file}: {len(cleaned_chunks)}")
    return [(source_name, chunk) for chunk in cleaned_chunks]

def setup_chromadb(prd_hld_chunks, istqb_chunks):
    """Store unique chunks in ChromaDB to avoid duplication."""
    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="knowledge_base")

    existing_ids = set(collection.get()["ids"])  # Fetch already stored IDs

    valid_chunks = []
    valid_embeddings = []
    valid_ids = []
    valid_metadata = []

    for i, (source, chunk) in enumerate(prd_hld_chunks + istqb_chunks):
        chunk_id = f"{source}_chunk_{i}"
        if chunk_id in existing_ids:
            print(f"Skipping duplicate chunk: {chunk_id}")
            continue  # Skip storing existing chunks

        try:
            if chunk.strip():
                embedding = text_embedding_model.encode(chunk).tolist()
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)
                valid_ids.append(chunk_id)
                valid_metadata.append({"source": source, "chunk_id": str(i)})  # Ensure metadata is a string
        except Exception as e:
            print(f"Error processing chunk {i} from {source}: {e}")

    if valid_chunks:
        collection.add(
            ids=valid_ids,
            embeddings=valid_embeddings,
            metadatas=valid_metadata,
            documents=valid_chunks
        )
        print(f"Stored {len(valid_chunks)} new chunks in ChromaDB.")

    return collection

def query_chromadb(collection, query, top_k=7, similarity_threshold=0.7):
    """Retrieve relevant chunks from ChromaDB."""
    query_embedding = text_embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_chunks = results["documents"][0] if results["documents"] else []
    retrieved_scores = results["distances"][0] if results["distances"] else []

    # Apply similarity threshold
    filtered_chunks = [chunk for chunk, score in zip(retrieved_chunks, retrieved_scores) if score >= similarity_threshold]

    print(f"Filtered {len(filtered_chunks)} relevant chunks (Threshold: {similarity_threshold})")
    return filtered_chunks[:5]  # Limit results to top 5

def generate_response(query, context_chunks):
    """Generate ISTQB-compliant test cases using a pre-trained AI model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-0.5B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    context = "\n".join(context_chunks)

    messages = [
        {"role": "system",
         "content": "You are an ISTQB-certified software test engineer. Generate ISTQB standard **manual test cases** "
                    "for the **Student API** based on the PRD, HLD, and project story. "
                    "Use White-box and Black-box testing techniques where applicable."
                    "\nEach test case **MUST** include:"
                    "\n- **Test Case ID** (Unique, PRD-XXX format)"
                    "\n- **Test Scenario**"
                    "\n- **Pre-conditions**"
                    "\n- **Test Steps** (Action, Input Data, Expected Result)"
                    "\n- **Priority** (High/Medium/Low)"
                    "\n\nThe output **MUST BE IN ENGLISH ONLY** and formatted as a structured table."},
        {"role": "user",
         "content": f"Generate ISTQB standard **manual Test Cases** for the **Student API** based on PRD, HLD, and the Project Story."
                    f"\n\nContext:\n{context}\n\nAnswer:"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)

    generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask,
                                   max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.batch_decode(
        [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
        skip_special_tokens=True
    )[0]

    return response

def main():
    """Main function to process PDFs and generate test cases."""
    # Step 1: Load and chunk both PDFs
    prd_hld_chunks = load_and_chunk_pdf(PRD_HLD_FILE, "PRD_HLD")
    istqb_chunks = load_and_chunk_pdf(ISTQB_FILE, "ISTQB")

    # Step 2: Set up ChromaDB
    collection = setup_chromadb(prd_hld_chunks, istqb_chunks)

    # Step 3: User Input Query
    user_input = """Write ISTQB standard manual test cases related to PRD (Product Requirements Document),
     HLD (High-Level Design), and user stories for Student API.
Each test case should include:
- **Test Case ID**  
- **Test Scenario**  
- **Pre-conditions**  
- **Test Steps** (Action, Input Data, Expected Result)  
- **Priority** (High/Medium/Low)  
Use **white-box and black-box testing techniques**. Ensure the output is formatted as a **structured table**."""

    # Step 4: Query ChromaDB for relevant chunks
    relevant_chunks = query_chromadb(collection, user_input)
    print(f"Relevant Chunks: {relevant_chunks}")

    # Step 5: Generate response
    response = generate_response(user_input, relevant_chunks)
    print(f"Generated Response:\n{response}")

if __name__ == "__main__":
    main()
