import fitz  # PyMuPDF
import os
import re
import hashlib
from pathlib import Path
from PIL import Image
import concurrent.futures
import io
from typing import List, Dict, Tuple, Optional, Any
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, "output_hybrid_images")
TXT_OUTPUT_DIR = os.path.join(BASE_DIR, "output_hybrid_text_files")
PIXMAP_DPI = 150

# Keep the get_image_explanation placeholder as is
def get_image_explanation(image_as_jpg: bytes, prompt_in_detail: str) -> Optional[str]:
    """Your existing image explanation function"""
    # ... (keep your existing implementation)
    pass

def extract_urls_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract URLs from text using regex patterns.
    Returns list of tuples (link_text, url)
    """
    # Match both explicit URLs and text with hyperlinks
    url_patterns = [
        r'(?P<text>[^\s]+)\s*\((?P<url>https?://[^\s\)]+)\)',  # Format: text(http://...)
        r'(?P<url>https?://[^\s]+)',  # Plain URLs
    ]
    
    urls = []
    for pattern in url_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            match_dict = match.groupdict()
            if 'text' in match_dict and match_dict['text']:
                urls.append((match_dict['text'], match_dict['url']))
            else:
                # If no explicit text, use the URL itself as text
                urls.append((match_dict['url'], match_dict['url']))
    
    return urls

def sanitize_filename_for_paths(filename: str) -> str:
    """Creates a safe string usable for directory and file names."""
    name_part = Path(filename).stem
    sanitized = re.sub(r'\W+', '_', name_part)
    hasher = hashlib.sha1(filename.encode())
    short_hash = hasher.hexdigest()[:8]
    final_name = f"{sanitized[:60]}_{short_hash}".strip('_')
    if not final_name:
        final_name = f"pdf_{short_hash}"
    return final_name

def extract_and_save_image_simple(
    doc: fitz.Document,
    xref: int,
    save_dir: Path,
    page_num: int,
    img_idx_on_page: int
) -> Optional[str]:
    """Extracts a single image by xref, saves it, returns absolute path."""
    try:
        img_data = doc.extract_image(xref)
        if not img_data:
            return None

        image_bytes = img_data["image"]
        ext = img_data["ext"]

        # Ensure save_dir exists
        os.makedirs(save_dir, exist_ok=True)

        image_filename = f"page_{page_num}_img_{img_idx_on_page}.{ext}"
        save_path = save_dir / image_filename

        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.mode in ['RGBA', 'P', 'LA']:
            img_pil = img_pil.convert('RGB')
        img_pil.save(save_path)

        # Always return absolute path
        return os.path.abspath(save_path)

    except Exception as e:
        logger.error(f"Error saving image xref {xref} on page {page_num}: {e}")
        return None

def process_single_pdf_hybrid(pdf_path: str) -> Optional[str]:
    """
    Processes a single PDF using the hybrid approach.
    Uses Fitz text for simple pages, multimodal func for complex pages.
    """
    pdf_path = os.path.abspath(pdf_path)
    filename = Path(pdf_path).name
    sanitized_name = sanitize_filename_for_paths(filename)
    logger.info(f"Processing '{filename}' (Sanitized: {sanitized_name})...")
    
    pdf_image_dir = Path(IMAGE_SAVE_DIR) / sanitized_name

    try:
        os.makedirs(pdf_image_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)  # Get actual page count
        final_text_parts: List[str] = []
        page_separator = f"\n\n---\n\n## Page {{page_num}} of {total_pages}\n\n"

        for page_num in range(total_pages):  # Use 0-based indexing internally
            actual_page_num = page_num + 1  # Convert to 1-based for display
            logger.info(f"Processing Page {actual_page_num}/{total_pages} for '{filename}'...")
            page_processed_text: Optional[str] = None

            try:
                page = doc[page_num]  # Get page using 0-based index
                # Check complexity
                image_list = page.get_images(full=False)
                is_complex = bool(image_list) or bool(page.find_tables())

                if not is_complex:
                    # Process Simple Page
                    logger.info(f"Page {actual_page_num}: Simple text extraction.")
                    # Fixed: Use correct fitz flags
                    page_processed_text = page.get_text(
                        "text",
                        sort=True,
                        flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_IMAGES
                    )
                    
                    # Extract and mark URLs in the text
                    urls = extract_urls_from_text(page_processed_text)
                    for link_text, url in urls:
                        marker = f"[URL_MARKER|href={url}|END_URL_MARKER]"
                        page_processed_text = page_processed_text.replace(
                            link_text,
                            f"{link_text} {marker}"
                        )

                    if not page_processed_text:
                        page_processed_text = "[EMPTY PAGE]\n"

                else:
                    # Process Complex Page
                    logger.info(f"Page {actual_page_num}: Complex page detected (Images/Tables).")
                    image_paths_on_page: List[str] = []
                    
                    # Extract & Save Images
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        absolute_image_path = extract_and_save_image_simple(
                            doc, xref, pdf_image_dir, actual_page_num, img_idx
                        )
                        if absolute_image_path:
                            image_paths_on_page.append(absolute_image_path)
                            logger.info(f"Extracted image: {absolute_image_path}")

                    # Render Page
                    pix = page.get_pixmap(dpi=PIXMAP_DPI)
                    page_image_bytes = pix.tobytes("png")

                    # Construct Detailed Prompt (using absolute paths)
                    detailed_prompt = f"""Analyze the provided image of page {actual_page_num} from document '{filename}'.
Instructions:
1. Perform OCR to extract all text in the correct reading order.
2. Identify any distinct images visible within the page image.
3. Identify any tables visible within the page image.
4. Generate a concise textual explanation for each identified image.
5. The following image file paths correspond to the images on this page: {image_paths_on_page}
6. Reconstruct the full text content of the page, maintaining the reading order.
7. Use absolute paths in image markers: [IMAGE_MARKER|path=<ABSOLUTE_PATH>|explanation=<EXPLANATION>|END_IMAGE_MARKER]
8. For tables: [TABLE_MARKER]<extracted table content>[END_TABLE_MARKER]
9. For URLs: [URL_MARKER|href=<url>|END_URL_MARKER]
10. Return ONLY the reconstructed text with markers embedded.
"""

                    # Call Multimodal Function
                    page_processed_text = get_image_explanation(
                        image_as_jpg=page_image_bytes,
                        prompt_in_detail=detailed_prompt
                    )

                    if page_processed_text is None:
                        logger.warning(f"Multimodal function failed for page {actual_page_num}. Using fallback.")
                        page_processed_text = page.get_text(
                            "text",
                            sort=True,
                            flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
                        )
                        if not page_processed_text:
                            page_processed_text = "[COMPLEX PAGE PROCESSING FAILED]\n"

            except Exception as page_proc_err:
                logger.error(f"Error processing page {actual_page_num}: {page_proc_err}")
                page_processed_text = f"[ERROR PROCESSING PAGE {actual_page_num}: {str(page_proc_err)}]\n"

            # Append processed text with correct page numbering
            final_text_parts.append(page_separator.format(page_num=actual_page_num))
            final_text_parts.append(page_processed_text.strip() if page_processed_text else "")

        doc.close()
        full_modified_text = "".join(final_text_parts)
        logger.info(f"Finished processing '{filename}'. Text length: {len(full_modified_text)}")
        return full_modified_text

    except Exception as e:
        logger.error(f"Error processing PDF '{pdf_path}': {e}", exc_info=True)
        return None

def process_pdfs_concurrent(pdf_paths: List[str], max_workers: Optional[int] = None) -> Dict[str, Optional[str]]:
    """
    Process multiple PDFs concurrently with proper thread management.
    """
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    
    results: Dict[str, Optional[str]] = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(process_single_pdf_hybrid, pdf_path): pdf_path 
            for pdf_path in pdf_paths
        }
        
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                results[pdf_path] = future.result()
            except Exception as exc:
                logger.error(f"PDF processing failed for {pdf_path}: {exc}")
                results[pdf_path] = None
                
    return results

if __name__ == "__main__":
    # Create output directories
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

    # Find PDF files
    pdf_files = [str(p) for p in Path(".").glob("*.pdf") if p.is_file()]
    
    if not pdf_files:
        logger.error("No PDF files found in the current directory.")
        exit()

    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    # Process PDFs
    results = process_pdfs_concurrent(pdf_files)
    
    # Save results
    success_count = 0
    for pdf_path, text_content in results.items():
        if text_content is not None:
            sanitized_name = sanitize_filename_for_paths(Path(pdf_path).name)
            output_path = os.path.join(TXT_OUTPUT_DIR, f"{sanitized_name}_hybrid_marked_up.txt")
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                success_count += 1
            except Exception as write_err:
                logger.error(f"Failed to save output for {pdf_path}: {write_err}")

    logger.info(f"\nProcessing Summary:")
    logger.info(f"Total PDFs: {len(pdf_files)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {len(pdf_files) - success_count}")
    logger.info(f"Output directory: {os.path.abspath(TXT_OUTPUT_DIR)}")
    logger.info(f"Images directory: {os.path.abspath(IMAGE_SAVE_DIR)}")
