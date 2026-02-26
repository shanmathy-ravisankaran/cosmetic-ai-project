import os
import re
from dotenv import load_dotenv
import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DB_PATH = "vector_db"
COLLECTION = "chemicals"


# ---------------- Vector DB (optional) ----------------
def build_vector_db(csv_path: str = "data/chemicals.csv"):
    df = pd.read_csv(csv_path)

    docs = []
    for _, row in df.iterrows():
        text = (
            f"Product Name: {row.get('ProductName', '')}\n"
            f"Brand: {row.get('BrandName', '')}\n"
            f"Company: {row.get('CompanyName', '')}\n"
            f"Chemical: {row.get('ChemicalName', '')}\n"
            f"Primary Category: {row.get('PrimaryCategory', '')}\n"
            f"Sub Category: {row.get('SubCategory', '')}\n"
        )
        docs.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.create_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=DB_PATH,
        collection_name=COLLECTION,
    )

    return "Vector DB built!"


# ---------------- Helpers ----------------
def _norm(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_question(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return f" {q} "


# ---------------- Safety Intelligence ----------------
HAZARDOUS_CHEMICALS = {
    "benzene": "known human carcinogen linked to blood disorders and leukemia",
    "formaldehyde": "classified carcinogen with prolonged exposure risks",
}

REGULATED_COMMON_INGREDIENTS = {
    "retinol": "commonly used in skincare for anti-aging and acne treatment",
    "salicylic acid": "widely used for acne and exfoliation",
    "niacinamide": "commonly used vitamin B3 derivative in skincare",
}


def _generate_safety_answer(chemical_name: str) -> str:
    chem = chemical_name.lower()

    if chem in HAZARDOUS_CHEMICALS:
        explanation = (
            f"{chemical_name} is considered harmful.\n\n"
            f"It is a {HAZARDOUS_CHEMICALS[chem]}. "
            "It is not intentionally added to cosmetic products."
        )

    elif chem in REGULATED_COMMON_INGREDIENTS:
        explanation = (
            f"{chemical_name} is {REGULATED_COMMON_INGREDIENTS[chem]}.\n\n"
            "It is generally considered safe when used at regulated cosmetic concentrations. "
            "Some individuals may experience irritation depending on skin sensitivity."
        )

    else:
        explanation = (
            f"Whether {chemical_name} is harmful depends on concentration and exposure.\n\n"
            "Most cosmetic ingredients are considered safe within regulated limits. "
            "Risk depends on dosage and individual sensitivity."
        )

    disclaimer = (
        "\n\nI cannot provide medical advice regarding "
        f"{chemical_name}.\n"
        "For personal health decisions — especially during pregnancy or breastfeeding — "
        "please consult a qualified healthcare professional."
    )

    return explanation + disclaimer


# ---------------- Main QA ----------------
def ask_question(question: str, current_chemical: str = None,
                 max_items: int = 12, csv_path: str = "data/chemicals.csv"):

    df = pd.read_csv(csv_path)

    q_raw = (question or "").strip()
    q_clean = _clean_question(q_raw)

    safety_keywords = [
        "harmful", "toxic", "dangerous", "safe",
        "carcinogen", "cancer", "risk", "use"
    ]

    is_safety_question = any(k in q_raw.lower() for k in safety_keywords)

    chemical_names = df.get("ChemicalName", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist()
    chemical_norm_map = {_norm(c): c for c in chemical_names if _norm(c)}

    found_chemical = None

    # Exact match
    for cn_norm, cn_original in chemical_norm_map.items():
        if cn_norm and f" {cn_norm} " in q_clean:
            found_chemical = cn_original
            break

    # Partial match
    if not found_chemical:
        tokens = [t for t in q_clean.split() if len(t) >= 4]
        for t in tokens:
            for cn_norm, cn_original in chemical_norm_map.items():
                if t in cn_norm:
                    found_chemical = cn_original
                    break
            if found_chemical:
                break

    # If no chemical mentioned but we have memory → use it
    if not found_chemical and current_chemical:
        found_chemical = current_chemical

    # ---------------- SAFETY MODE ----------------
    if is_safety_question and found_chemical:
        return {
            "mode": "safety_mode",
            "chemical": found_chemical,
            "answer": _generate_safety_answer(found_chemical),
            "items": []
        }

    # ---------------- Chemical → Products ----------------
    if found_chemical:

        hits = df[df["ChemicalName"].astype(str).str.contains(found_chemical, case=False, na=False)]

        items = []
        seen = set()

        for _, r in hits.iterrows():
            prod = str(r.get("ProductName", "")).strip()
            if not prod or prod.lower() in seen:
                continue
            seen.add(prod.lower())

            items.append({
                "product": prod,
                "brand": str(r.get("BrandName", "")).strip(),
                "company": str(r.get("CompanyName", "")).strip(),
                "primary_category": str(r.get("PrimaryCategory", "")).strip(),
                "sub_category": str(r.get("SubCategory", "")).strip(),
            })

            if len(items) >= max_items:
                break

        return {
            "mode": "chemical_to_products",
            "chemical": found_chemical,
            "answer": f"{found_chemical} appears in {len(items)} products.",
            "items": items
        }

    return {
        "mode": "unknown",
        "answer": "I couldn’t identify the chemical. Try typing the exact name.",
        "items": []
    }