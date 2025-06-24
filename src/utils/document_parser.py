from unstructured.partition.pdf import partition_pdf


def parse_pdf(file_path: str, output_path: str = "./content"):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payloads=True,
        chunking_strategy="by_title",
        max_characters=10_000,
        combined_text_under_n_chars=2_000,
        new_after_n_chars=6_000,
    )

    return chunks

def separate_chunks(chunks):
    tables, texts = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    
    return texts, tables

def extract_images(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64