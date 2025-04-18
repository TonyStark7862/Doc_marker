import fitz  # PyMuPDF
import os
import re
import hashlib
from pathlib import Path
from PIL import Image
import concurrent.futures
import io
from typing import List, Dict, Tuple, Optional, Any

# --- Configuration ---
IMAGE_SAVE_DIR = "./output_hybrid_images"  # Organised by PDF
TXT_OUTPUT_DIR = "./output_hybrid_text_files" # Where to save .txt files
PIXMAP_DPI = 150 # Resolution for rendering complex pages

# --- User's Multimodal Function ---
# IMPORTANT: Replace this entire placeholder function with your actual import and implementation.
# from your_module import get_image_explanation
def get_image_explanation(image_as_jpg: bytes, prompt_in_detail: str) -> Optional[str]:
    """
    Placeholder for user's powerful multimodal LLM function.
    Takes page image bytes and a detailed prompt.
    Expected to return the reconstructed page text with markers embedded.
    """
    print(f"    Calling (Placeholder) get_image_explanation for a page image...")
    print(f"      Prompt Snippet: {prompt_in_detail[:150]}...")
    # Simulate work and potential output structure
    import time; time.sleep(0.5) # Simulate API call time

    # --- Simulation Logic (Replace with your actual call) ---
    # This simulation crudely tries to mimic returning *something* based on the prompt.
    # A real LLM would perform OCR, analysis, and reconstruction.
    simulated_output = f"[Simulated LLM Output for page image]\n"
    simulated_output += f"Text snippet from prompt: '{prompt_in_detail[:100]}...' \n"

    # Try to include markers based on paths found in the prompt simulation
    image_paths_in_prompt = re.findall(r"path=(.*?)\|", prompt_in_detail) # Crude extraction from prompt
    if image_paths_in_prompt:
         for i, path in enumerate(image_paths_in_prompt):
              simulated_output += f"Some text related to image {i+1}.\n"
              simulated_output += f"[IMAGE_MARKER|path={path}|explanation=Simulated explanation for image {i+1}|END_IMAGE_MARKER]\n"

    # Simulate finding a URL
    simulated_output += "Some text containing a link [URL_MARKER|href=http://example.com/simulated|END_URL_MARKER] maybe.\n"
    # Simulate finding a table
    simulated_output += "[TABLE_MARKER]Simulated | Table | Content\n---|---|---\nData | Cell | Here[END_TABLE_MARKER]\n"
    simulated_output += "End of simulated page content.\n"
    # --- End Simulation Logic ---

    # Clean the simulated output just in case
    # cleaned_output = re.sub(r'\n{3,}', '\n\n', simulated_output).strip()
    print(f"    (Placeholder) get_image_explanation finished.")
    return simulated_output
# --- End User Function Placeholder ---


# --- Helper Functions ---
def sanitize_filename_for_paths(filename: str) -> str:
    """Creates a safe string usable for directory and file names."""
    name_part = Path(filename).stem
    sanitized = re.sub(r'\W+', '_', name_part)
    hasher = hashlib.sha1(filename.encode())
    short_hash = hasher.hexdigest()[:8]
    final_name = f"{sanitized[:60]}_{short_hash}".strip('_')
    if not final_name: final_name = f"pdf_{short_hash}"
    return final_name

def extract_and_save_image_simple(
    doc: fitz.Document,
    xref: int,
    save_dir: Path,
    page_num: int,
    img_idx_on_page: int
) -> Optional[str]:
    """Extracts a single image by xref, saves it, returns relative path."""
    try:
        img_data = doc.extract_image(xref)
        if not img_data:
            return None

        image_bytes = img_data["image"]
        ext = img_data["ext"]

        image_filename = f"page_{page_num}_img_{img_idx_on_page}.{ext}"
        save_path = save_dir / image_filename

        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.mode in ['RGBA', 'P', 'LA']:
            img_pil = img_pil.convert('RGB')
        img_pil.save(save_path)

        # Calculate relative path from the base TXT output dir for portability
        try:
            txt_output_base = Path(TXT_OUTPUT_DIR)
            relative_path = os.path.relpath(save_path, txt_output_base)
            return Path(relative_path).as_posix() # Use forward slashes
        except ValueError: # Different drives, etc.
            return Path(save_path).as_posix() # Fallback to absolute POSIX path

    except Exception as e:
        print(f"    Error saving image xref {xref} on page {page_num}: {e}")
        return None

# --- Core PDF Processing Function (Worker) ---
def process_single_pdf_hybrid(pdf_path: str) -> Optional[str]:
    """
    Processes a single PDF using the hybrid approach.
    Uses Fitz text for simple pages, user's multimodal func for complex pages.
    """
    filename = Path(pdf_path).name
    sanitized_name = sanitize_filename_for_paths(filename)
    print(f"Processing '{filename}' (Sanitized: {sanitized_name})...")
    pdf_image_dir = Path(IMAGE_SAVE_DIR) / sanitized_name

    try:
        os.makedirs(pdf_image_dir, exist_ok=True) # Ensure image dir exists
        doc = fitz.open(pdf_path)
        final_text_parts: List[str] = []
        page_separator = f"\n\n---\n\n## Page {{page_num}}\n\n" # Template

        for page_num, page in enumerate(doc.pages(), start=1):
            print(f"  Processing Page {page_num}/{doc.page_count} for '{filename}'...")
            page_processed_text: Optional[str] = None

            try:
                # Check complexity
                image_list = page.get_images(full=False) # Just get xrefs/basic info
                table_finder = page.find_tables()
                is_complex = bool(image_list) or bool(table_finder.tables)

                if not is_complex:
                    # --- Process Simple Page ---
                    print(f"    Page {page_num}: Simple text extraction.")
                    page_processed_text = page.get_text("text", sort=True, flags=fitz.TEXTFLAGS_PRESERVE_WHITESPACE | fitz.TEXTFLAGS_OUTPUT_NATIVE_NEWLINES)
                    if not page_processed_text: # Handle empty pages
                         page_processed_text = "[EMPTY PAGE]\n"

                else:
                    # --- Process Complex Page ---
                    print(f"    Page {page_num}: Complex page detected (Images/Tables). Using multimodal function...")
                    image_paths_on_page: List[str] = []
                    # Step 3a: Pre-extract & Save Images
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        relative_image_path = extract_and_save_image_simple(
                            doc, xref, pdf_image_dir, page_num, img_idx
                        )
                        if relative_image_path:
                            image_paths_on_page.append(relative_image_path)
                            print(f"      Extracted image path: {relative_image_path}")
                        else:
                            print(f"      Warn: Failed to extract/save image xref {xref} on page {page_num}")


                    # Step 3b: Render Page
                    print(f"      Rendering page {page_num} to image...")
                    pix = page.get_pixmap(dpi=PIXMAP_DPI)
                    page_image_bytes = pix.tobytes("png") # Use PNG for lossless rendering
                    print(f"      Page image size: {len(page_image_bytes) / 1024:.1f} KB")

                    # Step 3c: Construct Detailed Prompt
                    # THIS IS CRITICAL - Adjust prompt based on your multimodal LLM's requirements
                    detailed_prompt = f"""Analyze the provided image of page {page_num} from document '{filename}'.
Instructions:
1. Perform OCR to extract all text in the correct reading order.
2. Identify any distinct images visible within the page image.
3. Identify any tables visible within the page image.
4. Generate a concise textual explanation for each identified image.
5. The following image file paths correspond to the images on this page, likely in top-to-bottom reading order: {image_paths_on_page}
6. Reconstruct the full text content of the page, maintaining the reading order.
7. When you reference an image in the reconstructed text (at the point where it appears), insert the following marker EXACTLY: [IMAGE_MARKER|path=<CORRECT_PATH_FROM_PROVIDED_LIST>|explanation=<YOUR_GENERATED_EXPLANATION>|END_IMAGE_MARKER]. Match the paths from the list ({image_paths_on_page}) to the images you identified in order.
8. When you reference a table, extract its content (e.g., as text) and wrap it like this: [TABLE_MARKER]<extracted table content>[END_TABLE_MARKER].
9. Identify any hyperlinks in the OCR'd text and mark them like this: [URL_MARKER|href=<extracted_url>|END_URL_MARKER] directly after the link text.
10. Return ONLY the fully reconstructed text content for this page with all specified markers embedded. Do not add any preamble or concluding remarks outside of the reconstructed text.
"""

                    # Step 3d: Call User's Multimodal Function
                    page_processed_text = get_image_explanation(
                        image_as_jpg=page_image_bytes, # Assuming func takes bytes
                        prompt_in_detail=detailed_prompt
                    )

                    if page_processed_text is None:
                        print(f"    Warn: Multimodal function returned None for page {page_num}. Falling back to simple text.")
                        page_processed_text = page.get_text("text", sort=True, flags=fitz.TEXTFLAGS_PRESERVE_WHITESPACE) # Fallback
                        if not page_processed_text: page_processed_text = "[COMPLEX PAGE PROCESSING FAILED]\n"

            except Exception as page_proc_err:
                 print(f"    Error processing page {page_num}: {page_proc_err}. Skipping page content.")
                 page_processed_text = f"[ERROR PROCESSING PAGE {page_num}: {page_proc_err}]\n"

            # Append processed text for the page
            final_text_parts.append(page_separator.format(page_num=page_num))
            final_text_parts.append(page_processed_text.strip() if page_processed_text else "")

        doc.close()
        full_modified_text = "".join(final_text_parts)
        print(f"  Finished processing '{filename}'. Final text length: {len(full_modified_text)}")
        return full_modified_text

    except Exception as e:
        print(f"Error processing PDF file '{pdf_path}': {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

# --- Main Execution Logic ---
if __name__ == "__main__":
    # pdf_files = ["./doc1.pdf", "./doc2.pdf"] # Or provide specific paths
    pdf_files = [str(p) for p in Path(".").glob("*.pdf") if p.is_file()] # PDFs in current dir

    if not pdf_files:
        print("No PDF files found in the current directory.")
        exit()

    print(f"Found {len(pdf_files)} PDF files to process.")
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

    success_count = 0
    failure_count = 0
    results: Dict[str, Optional[str]] = {}

    # Use ThreadPoolExecutor for concurrent PDF processing
    max_workers = min(4, os.cpu_count() or 1) # Adjust as needed
    print(f"Processing files concurrently using up to {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_single_pdf_hybrid, pdf_path): pdf_path for pdf_path in pdf_files}

        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result_text = future.result() # Get the processed text string (or None)
                results[pdf_path] = result_text # Store result regardless of success/failure
                if result_text is not None: # Consider None as failure for count
                    # print(f"Successfully processed: {pdf_path}") # Can be verbose
                    success_count += 1
                else:
                    print(f"Failed to process: {pdf_path}")
                    failure_count += 1
            except Exception as exc:
                print(f"'{pdf_path}' generated an exception during future handling: {exc}")
                results[pdf_path] = None # Mark as failed
                failure_count += 1

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {success_count}")
    print(f"Failed processing:    {failure_count}")

    # Save successful results to text files
    saved_count = 0
    for pdf_path, text_content in results.items():
        if text_content is not None: # Only save if processing didn't return None
            sanitized_name = sanitize_filename_for_paths(Path(pdf_path).name)
            output_filename = Path(TXT_OUTPUT_DIR) / f"{sanitized_name}_hybrid_marked_up.txt"
            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(text_content)
                saved_count += 1
            except Exception as write_err:
                print(f"Error writing output file for '{pdf_path}': {write_err}")

    print(f"Saved {saved_count} marked-up text files to: {os.path.abspath(TXT_OUTPUT_DIR)}")
    print(f"Images saved in subdirectories under: {os.path.abspath(IMAGE_SAVE_DIR)}")
