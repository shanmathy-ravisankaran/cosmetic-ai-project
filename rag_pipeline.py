import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DB_PATH = "vector_db"
COLLECTION = "cosmetic_chemicals"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DEFAULT_K = 5
_AUDIT_EXACT_MATCH_TEST_DONE = False


def _safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _row_to_document(row: pd.Series, row_index: int) -> Document:
    product = _safe_str(row.get("ProductName", ""))
    brand = _safe_str(row.get("BrandName", ""))
    company = _safe_str(row.get("CompanyName", ""))
    chemical = _safe_str(row.get("ChemicalName", ""))
    primary_category = _safe_str(row.get("PrimaryCategory", ""))
    sub_category = _safe_str(row.get("SubCategory", ""))

    page_content = (
        f"Product Name: {product}\n"
        f"Brand: {brand}\n"
        f"Company: {company}\n"
        f"Chemical: {chemical}\n"
        f"Primary Category: {primary_category}\n"
        f"Sub Category: {sub_category}"
    )

    metadata = {
        "row_index": row_index,
        "product": product,
        "brand": brand,
        "company": company,
        "chemical": chemical,
        "primary_category": primary_category,
        "sub_category": sub_category,
    }

    return Document(page_content=page_content, metadata=metadata)


def _embedding_client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _llm_client() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def build_vector_db(
    csv_path: str = "data/chemicals.csv",
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION,
) -> str:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file).fillna("")
    csv_abs_path = str(csv_file.resolve())
    total_rows = len(df)
    unique_chemicals = (
        df.get("ChemicalName", pd.Series([], dtype=str))
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
    salicylic_rows = (
        df.get("ChemicalName", pd.Series([], dtype=str))
        .astype(str)
        .str.contains("salicylic acid", case=False, na=False)
        .sum()
    )
    print(f"[RAG AUDIT] CSV absolute path: {csv_abs_path}")
    print(f"[RAG AUDIT] CSV total rows: {total_rows}")
    print(f"[RAG AUDIT] CSV unique chemicals: {int(unique_chemicals)}")
    print(f"[RAG AUDIT] Rows containing 'Salicylic Acid' (case-insensitive): {int(salicylic_rows)}")
    total_csv_rows = len(df)
    print(f"[RAG VALIDATION] CSV file contains {total_csv_rows} rows")
    
    documents = [_row_to_document(row, i) for i, (_, row) in enumerate(df.iterrows())]
    total_documents = len(documents)
    print(f"[RAG VALIDATION] Created {total_documents} documents from CSV rows")
    
    if total_csv_rows != total_documents:
        print(f"[RAG VALIDATION] WARNING: Row count mismatch! CSV rows: {total_csv_rows}, Documents: {total_documents}")
    else:
        print(f"[RAG VALIDATION] ✓ All CSV rows successfully converted to documents")

    db_dir = Path(db_path)
    db_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RAG AUDIT] DB_PATH full path: {db_dir.resolve()}")
    subfolders = [p for p in db_dir.iterdir() if p.is_dir()]
    if not subfolders:
        print("[RAG AUDIT] DB_PATH subfolders: none")
    else:
        print("[RAG AUDIT] DB_PATH subfolders and file counts:")
        for sub in subfolders:
            file_count = sum(1 for p in sub.rglob("*") if p.is_file())
            print(f"[RAG AUDIT] - {sub.name}: {file_count} files")

    pre_count = None
    try:
        existing_store = Chroma(
            persist_directory=str(db_dir),
            collection_name=collection_name,
            embedding_function=_embedding_client(),
        )
        pre_count = existing_store._collection.count()
    except Exception:
        pre_count = None

    # Delete existing collection if it exists to avoid duplicates
    try:
        existing_store = Chroma(
            persist_directory=str(db_dir),
            collection_name=collection_name,
            embedding_function=_embedding_client(),
        )
        existing_count = existing_store._collection.count()
        if existing_count > 0:
            print(f"[RAG FIX] Deleting existing collection '{collection_name}' with {existing_count} documents...")
            # Delete the collection using Chroma's delete_collection method
            try:
                existing_store.delete_collection()
                print(f"[RAG FIX] ✓ Deleted old collection")
            except Exception as del_e:
                # Fallback: try using the underlying client
                try:
                    client = existing_store._client
                    client.delete_collection(name=collection_name)
                    print(f"[RAG FIX] ✓ Deleted old collection (via client)")
                except Exception as del_e2:
                    print(f"[RAG FIX] Could not delete collection (will overwrite): {del_e2}")
    except Exception as e:
        print(f"[RAG FIX] No existing collection to delete (this is OK for first build): {e}")

    # Rebuild collection from current CSV snapshot.
    print(f"[RAG FIX] Creating new collection '{collection_name}' with {len(documents)} documents...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=_embedding_client(),
        persist_directory=str(db_dir),
        collection_name=collection_name,
    )

    # Keep compatibility with older/newer Chroma wrappers.
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    post_count = None
    try:
        post_count = vector_store._collection.count()
    except Exception as e:
        print(f"[RAG FIX] Error getting post-count: {e}")
        post_count = None

    print(f"[RAG AUDIT] Total documents created: {len(documents)}")
    print(f"[RAG AUDIT] Collection name: {collection_name}")
    print(f"[RAG AUDIT] Persist directory: {db_dir.resolve()}")
    print(f"[RAG AUDIT] Documents stored in Chroma after indexing: {post_count if post_count is not None else 'unavailable'}")
    
    # Critical validation: verify document count matches
    if post_count is not None:
        if post_count != len(documents):
            print(f"[RAG FIX] ⚠ WARNING: Document count mismatch! Expected {len(documents)}, got {post_count}")
        else:
            print(f"[RAG FIX] ✓ Verified: All {post_count} documents successfully indexed")
    else:
        print(f"[RAG FIX] ⚠ WARNING: Could not verify document count")
    
    if pre_count is None or post_count is None:
        print("[RAG AUDIT] Duplicate indexing check: unavailable")
    else:
        duplicate_indexing = pre_count > 0 and post_count > len(documents)
        print(f"[RAG AUDIT] Duplicate indexing detected: {duplicate_indexing} (pre_count={pre_count}, post_count={post_count})")

    print(f"[RAG VALIDATION] Indexed {total_documents} documents into Chroma collection '{collection_name}'")
    
    # Validation: Check for "Salicylic Acid" in indexed documents
    salicylic_acid_count = sum(1 for doc in documents if doc.metadata.get("chemical", "").lower() == "salicylic acid")
    print(f"[RAG VALIDATION] Documents with chemical == 'Salicylic Acid': {salicylic_acid_count}")
    if salicylic_acid_count > 0:
        print(f"[RAG VALIDATION] ✓ Confirmed: Vector DB contains {salicylic_acid_count} document(s) with chemical 'Salicylic Acid'")
    else:
        print(f"[RAG VALIDATION] ⚠ WARNING: No documents found with chemical 'Salicylic Acid'")

    return f"Indexed {len(documents)} records into Chroma collection '{collection_name}'."


def _get_vector_store(
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION,
) -> Chroma:
    return Chroma(
        persist_directory=db_path,
        collection_name=collection_name,
        embedding_function=_embedding_client(),
    )


def ensure_vector_db(
    csv_path: str = "data/chemicals.csv",
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION,
) -> None:
    db_dir = Path(db_path)
    
    # Check if directory exists and has files
    if not db_dir.exists() or not any(db_dir.iterdir()):
        print(f"[RAG FIX] Vector DB directory empty or missing, building...")
        build_vector_db(csv_path=csv_path, db_path=db_path, collection_name=collection_name)
        return
    
    # Verify the specific collection exists and has documents
    try:
        test_store = Chroma(
            persist_directory=str(db_dir),
            collection_name=collection_name,
            embedding_function=_embedding_client(),
        )
        doc_count = test_store._collection.count()
        print(f"[RAG FIX] Collection '{collection_name}' exists with {doc_count} documents")
        
        if doc_count == 0:
            print(f"[RAG FIX] Collection exists but is empty, rebuilding...")
            build_vector_db(csv_path=csv_path, db_path=db_path, collection_name=collection_name)
        else:
            print(f"[RAG FIX] ✓ Collection '{collection_name}' is ready with {doc_count} documents")
    except Exception as e:
        print(f"[RAG FIX] Error checking collection '{collection_name}': {e}")
        print(f"[RAG FIX] Rebuilding vector DB...")
        build_vector_db(csv_path=csv_path, db_path=db_path, collection_name=collection_name)


HYBRID_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, hybrid cosmetic assistant.\n\n"
            "Rules:\n"
            "- NEVER mention internal retrieval/RAG mechanics (do not say 'retrieved context', 'database does not contain', etc.).\n"
            "- Do NOT hallucinate product-specific claims that are not supported by the provided database facts.\n"
            "- If database facts are provided, use them as factual grounding, but weave them naturally into the answer.\n"
            "- When database facts include specific products or brands, mention those concrete product names early in the answer before broader explanation.\n"
            "- For safety/toxicity/usage questions, provide an educational explanation of at least 3–6 sentences.\n"
            "- Keep the tone clear, structured, and practical, and respond as natural paragraphs (no section headings like 'From our database' or 'Additional explanation').\n",
        ),
        (
            "human",
            "User question:\n{question}\n\n"
            "Database facts (may be empty):\n{db_facts}\n\n"
            "Write a single, coherent answer in natural paragraphs:\n"
            "- If database facts are provided, first mention specific product names, brands, or companies that are relevant, then continue with a concise explanation.\n"
            "- If no useful database facts are present, give a general but accurate explanation of the chemical or product without referring to any database limitations.\n"
            "- Avoid inventing product-specific safety claims that are not supported by the database facts.\n"
            "- Never mention that information is missing from a database; always provide the most helpful explanation you can.\n",
        ),
    ]
)


def _is_safety_question(question: str) -> bool:
    """Detect if question is about safety, harmful effects, toxicity, side effects, or usage."""
    q = (question or "").lower()
    safety_keywords = [
        "safe",
        "harmful",
        "toxic",
        "dangerous",
        "side effects",
        "side effect",
        "can we use",
        "can i use",
        "is it safe",
        "is this safe",
        "how to use",
        "usage",
        "warning",
        "risk",
        "risks",
        "toxicity",
    ]
    return any(k in q for k in safety_keywords)


def _norm_text(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _query_matches_retrieved_metadata(question: str, docs: List[Document]) -> bool:
    """Heuristic: treat as 'in dataset' if any query keyword appears in product/brand/company/chemical fields."""
    qn = _norm_text(question)
    if not qn:
        return False

    # Split question into keywords and keep reasonably informative ones
    q_words = [w for w in qn.split() if len(w) >= 3]
    if not q_words:
        return False

    fields = ("product", "brand", "company", "chemical")
    for doc in docs or []:
        meta = doc.metadata or {}
        for f in fields:
            v = _norm_text(str(meta.get(f, "")))
            if not v:
                continue
            for w in q_words:
                # Partial/keyword match: any query word occurring inside the metadata text
                if w in v:
                    return True
    return False


def _format_context(docs: List[Document]) -> str:
    if not docs:
        return "No relevant context retrieved."

    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[{i}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def ask_question(
    question: str,
    current_chemical: str = None,
    max_items: int = DEFAULT_K,
    csv_path: str = "data/chemicals.csv",
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION,
) -> Dict[str, Any]:
    del current_chemical  # Legacy argument retained for app compatibility.
    global _AUDIT_EXACT_MATCH_TEST_DONE

    user_question = (question or "").strip()
    if not user_question:
        return {
            "mode": "error",
            "answer": "Please enter a question.",
            "items": [],
            "sources": [],
        }

    # Verify OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[RAG FIX] ⚠ ERROR: OPENAI_API_KEY is missing!")
        return {
            "mode": "error",
            "answer": "OPENAI_API_KEY is missing. Add it to your environment or .env file.",
            "items": [],
            "sources": [],
        }
    else:
        print(f"[RAG FIX] ✓ OpenAI API Key found (length: {len(api_key)} characters)")

    try:
        ensure_vector_db(csv_path=csv_path, db_path=db_path, collection_name=collection_name)

        k = max(3, min(max_items or DEFAULT_K, 10))
        print(f"[RAG AUDIT] Embedding model: {EMBEDDING_MODEL}")
        print(f"[RAG AUDIT] Chat model: {CHAT_MODEL}")
        vector_store = _get_vector_store(db_path=db_path, collection_name=collection_name)
        
        # Verify collection has documents before querying
        try:
            collection_count = vector_store._collection.count()
            print(f"[RAG FIX] Collection '{collection_name}' document count: {collection_count}")
            if collection_count == 0:
                print(f"[RAG FIX] ⚠ ERROR: Collection is empty! Rebuilding...")
                build_vector_db(csv_path=csv_path, db_path=db_path, collection_name=collection_name)
                vector_store = _get_vector_store(db_path=db_path, collection_name=collection_name)
                collection_count = vector_store._collection.count()
                print(f"[RAG FIX] After rebuild, collection count: {collection_count}")
        except Exception as e:
            print(f"[RAG FIX] ⚠ ERROR: Could not verify collection count: {e}")
            print(f"[RAG FIX] Attempting to rebuild vector DB...")
            build_vector_db(csv_path=csv_path, db_path=db_path, collection_name=collection_name)
            vector_store = _get_vector_store(db_path=db_path, collection_name=collection_name)

        if not _AUDIT_EXACT_MATCH_TEST_DONE:
            exact_docs = vector_store.similarity_search("Salicylic Acid", k=5)
            print("[RAG AUDIT] One-time exact-match test query: 'Salicylic Acid' (k=5)")
            print(f"[RAG AUDIT] Exact-match returned docs: {len(exact_docs)}")
            for i, d in enumerate(exact_docs, start=1):
                meta = d.metadata or {}
                print(
                    f"[RAG AUDIT] Exact-match doc {i}: "
                    f"row_index={meta.get('row_index')}, "
                    f"product={meta.get('product', '')}, "
                    f"chemical={meta.get('chemical', '')}"
                )
            _AUDIT_EXACT_MATCH_TEST_DONE = True

        docs = vector_store.similarity_search(user_question, k=k)
        print(f"[RAG AUDIT] similarity_search requested k: {k}")
        print(f"[RAG AUDIT] similarity_search returned docs: {len(docs)}")
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            print(
                f"[RAG AUDIT] Retrieved doc {i}: "
                f"row_index={meta.get('row_index')}, "
                f"product={meta.get('product', '')}, "
                f"chemical={meta.get('chemical', '')}"
            )
        
        # Deduplicate documents based on product name (case-insensitive)
        seen_products = set()
        unique_docs = []
        for doc in docs:
            product = (doc.metadata.get("product", "") or "").strip().lower()
            if product and product not in seen_products:
                seen_products.add(product)
                unique_docs.append(doc)
        
        print(f"[RAG FIX] Deduplicated: {len(docs)} docs -> {len(unique_docs)} unique products")
        docs = unique_docs
        
        try:
            scored_docs = vector_store.similarity_search_with_relevance_scores(user_question, k=k)
            print("[RAG AUDIT] Similarity scores:")
            for i, (doc, score) in enumerate(scored_docs, start=1):
                meta = doc.metadata or {}
                print(
                    f"[RAG AUDIT] Score doc {i}: "
                    f"row_index={meta.get('row_index')}, "
                    f"chemical={meta.get('chemical', '')}, "
                    f"score={score}"
                )
        except Exception:
            print("[RAG AUDIT] Similarity scores: unavailable")
        
        print(f"[RAG VALIDATION] similarity_search retrieved {len(docs)} document(s) (requested k={k})")
        print(f"[RAG VALIDATION] Chemical field of each retrieved document:")
        for idx, doc in enumerate(docs, start=1):
            chemical = doc.metadata.get("chemical", "N/A")
            print(f"  [{idx}] Chemical: {chemical}")
        print("=" * 60)

        # Decide whether we have a real dataset match (avoid irrelevant sources for unknown brands/terms)
        is_safety_q = _is_safety_question(user_question)
        has_dataset_match = _query_matches_retrieved_metadata(user_question, docs)

        # Build "db facts" only when we believe the query is actually about something in the dataset.
        # Otherwise omit the database section entirely (do not mention missing data).
        db_facts = _format_context(docs) if has_dataset_match else ""

        if is_safety_q:
            mode_type = "chemical_knowledge"
            print(f"[RAG MODE] Chemical Knowledge Mode (hybrid). dataset_match={has_dataset_match}")
        else:
            mode_type = "hybrid_assistant"
            print(f"[RAG MODE] Hybrid Assistant Mode. dataset_match={has_dataset_match}")

        prompt = HYBRID_ASSISTANT_PROMPT.format_messages(
            question=user_question,
            db_facts=db_facts or "",
        )
        
        # Verify LLM connection
        try:
            llm_response = _llm_client().invoke(prompt)
            answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            print(f"[RAG FIX] ✓ LLM connected and responded successfully")
            print(f"[RAG FIX] Answer mode: {mode_type}")
            print(f"[RAG FIX] Answer length: {len(answer)} characters")
            
            # Clean any stray HTML tags from LLM response
            import re
            # Remove any standalone closing HTML tags that might have leaked in
            answer = re.sub(r'</div>', '', answer)
            answer = re.sub(r'<div[^>]*>', '', answer)
            answer = answer.strip()
        except Exception as llm_error:
            print(f"[RAG FIX] ⚠ LLM Error: {llm_error}")
            answer = f"Error generating answer: {str(llm_error)}. Please check your OPENAI_API_KEY."

        # Build sources from deduplicated docs (already unique by product).
        # If we don't have a dataset match, do NOT return irrelevant sources.
        sources = []
        seen_source_products = set()
        if has_dataset_match:
            for doc in docs:
                meta = doc.metadata or {}
                product = (meta.get("product", "") or "").strip().lower()

                # Additional deduplication check for sources
                if product and product not in seen_source_products:
                    seen_source_products.add(product)
                    sources.append(
                        {
                            "row_index": meta.get("row_index"),
                            "product": meta.get("product", ""),
                            "brand": meta.get("brand", ""),
                            "company": meta.get("company", ""),
                            "chemical": meta.get("chemical", ""),
                            "primary_category": meta.get("primary_category", ""),
                            "sub_category": meta.get("sub_category", ""),
                        }
                    )
        
        print(f"[RAG FIX] Final unique sources: {len(sources)}")

        return {
            "mode": mode_type,
            "answer": str(answer).strip(),
            "items": [],
            "sources": sources,
        }

    except Exception as exc:
        return {
            "mode": "error",
            "answer": f"RAG pipeline error: {exc}",
            "items": [],
            "sources": [],
        }
