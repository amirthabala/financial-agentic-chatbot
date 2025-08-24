from bs4 import BeautifulSoup
import re
import os
from langchain.schema import Document


def extract_text_from_html(path):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text


def normalize_section_name(name):
    return re.sub(r"\s+", " ", name.replace("\xa0", " ")).strip().upper().rstrip(".")


def split_and_create_documents(content, file_name):
    pattern = re.compile(r"(\bITEM\s+\d+(?:[A-Z]|\.)?\b)", re.IGNORECASE)
    sections_to_extract = ["ITEM 1A", "ITEM 7", "ITEM 7A", "ITEM 8", "ITEM 15"]
    sections_data = {}

    split_content = pattern.split(content)

    for i in range(1, len(split_content), 2):
        section_name = normalize_section_name(split_content[i])
        section_content = split_content[i + 1].strip()

        if section_name and section_name in sections_to_extract and section_content:
            # if section_name and section_content:
            sections_data[section_name] = (
                sections_data[section_name] + section_content
                if sections_data.get(section_name)
                else section_content
            )

    return [
        Document(
            page_content=content,
            metadata={"section": name, "source": file_name},
        )
        for [name, content] in sections_data.items()
    ]


def load_documents(folder_path):
    files = os.listdir(folder_path)
    docs = []

    # Iterate through files in the folder
    for file in files:
        file_path = os.path.join(folder_path, file)

        print(f"loading contents of file {file_path}....")
        file_content = extract_text_from_html(file_path)
        file_docs = split_and_create_documents(file_content, file)
        print(f"loaded --> {len(file_docs)} docs!")

        docs.extend(file_docs)

    return docs


def split_documents_by_pageno(path, file_name):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")  # use lxml for better parsing
        for tag in soup(["script", "style"]):
            tag.decompose()

        file_year = file_name.split("-")[1][:4]
        documents = []

        # Split by <hr> directly
        parts = str(soup).split("<hr")
        for page_number, part in enumerate(parts, start=1):
            page_soup = BeautifulSoup(part, "lxml")
            text = page_soup.get_text().strip()
            if not text:
                continue
            document = Document(
                page_content=text,
                metadata={
                    "page_number": page_number,
                    "source": file_name,
                    "year": file_year,
                },
            )
            documents.append(document)

        return documents


def load_documents_(folder_path):
    files = os.listdir(folder_path)
    docs = []

    # Iterate through files in the folder
    for file in files:
        file_path = os.path.join(folder_path, file)

        print(f"loading contents of file {file_path}....")
        file_docs = split_documents_by_pageno(file_path, file)
        doc = file_docs[3]
        print(
            f"Page {doc.metadata['page_number']} year {doc.metadata['year']} source {doc.metadata['source']} Content: {doc.page_content[:200]}..."
        )
        print(f"loaded --> {len(file_docs)} docs!")

        docs.extend(file_docs)

    return docs
