import fitz  # PyMuPDF
import os
import re
import hashlib
from pathlib import Path
from PIL import Image
import concurrent.futures
import io
import traceback
from typing import List, Dict, Tuple, Optional, Any

# --- Configuration ---
IMAGE_SAVE_DIR = "./output_hybrid_images_abs"  # Organised by PDF, using absolute paths
TXT_OUTPUT_DIR = "./output_hybrid_text_files_abs" # Where to save .txt files
PIXMAP_DPI = 150 # Resolution for rendering complex pages

# --- User's Multimodal Function ---
# IMPORTANT: Replace this entire placeholder function with your actual import and implementation.
# Make sure your function can handle potential errors gracefully and return None or raise Exception.
# from your_module import image_response as get_image_explanation # Example rename on import
def get_image_explanation(image_as_jpg: bytes, prompt_in_detail: str) -> Optional[str]:
    """
    Placeholder for user's powerful multimodal LLM function.
    Takes page image bytes and a detailed prompt.
    Expected to return the reconstructed page text with markers embedded.
    """
    print(f"    Calling (Placeholder) get_image_explanation for a page image...")
    print(f"      Prompt Snippet: {prompt_in_detail[:200]}...") # Show a bit more prompt
    # Simulate work and potential output structure
    import time; time.sleep(0.5) # Simulate API call time

    # --- Simulation Logic (Replace with your actual call) ---
    simulated_output = f"[Simulated LLM Output for page image]\n"
    simulated_output += f"Text snippet from prompt: '{prompt_in_detail[:100]}...' \n"
    image_paths_in_prompt = re.findall(r"path=(.*?)\|", prompt_in_detail)
    if image_paths_in_prompt:
         for i, path in enumerate(image_paths_in_prompt):
              # Ensure path in marker matches input (use absolute POSIX path)
              abs_path_posix = Path(path).resolve().as_posix()
              simulated_output += f"Some simulated text related to image {i+1}.\n"
              simulated_output += f"[IMAGE_MARKER|path={abs_path_posix}|explanation=Simulated explanation for image {i+1}|END_IMAGE_MARKER]\n"
    simulated_output += "Some text containing a link [URL_MARKER|href=http://example.com/simulated|END_URL_MARKER] maybe.\n"
    simulated_output += "[TABLE_MARKER]Simulated | Table | Content\n---|---|---\nData | Cell | Here[END_TABLE_MARKER]\n"
    simulated_output += "End of simulated page content.\n"
    # --- End Simulation Logic ---

    print(f"    (Placeholder) get_image_explanation finished.")
    return simulated_output
# --- End User Function Placeholder ---


# --- Helper Functions ---
def sanitize_filename_for_paths(filename: str) -> str:
    """Creates a safe string usable for directory and file names."""
    name_part = Path(filename).stem
    sanitized = re.sub(r'\W+', '_', name_part)
    # Use SHA1 for shorter hash, ensure consistency
    hasher = hashlib.sha1(filename.encode('utf-8', errors='replace'))
    short_hash = hasher.hexdigest()[:8]
    final_name = f"{sanitized[:60]}_{short_hash}".strip('_')
    if not final_name: final_name = f"pdf_{short_hash}"
    # Ensure it doesn't start with problematic chars if possible
    if final_name.startswith(('.', '-', '_')): final_name = 'f' + final_name[1:]
    if not final_name : final_name = f"pdf_{short_hash}" # Final fallback
    return final_name

def extract_and_save_image_simple(
    doc: fitz.Document,
    xref: int,
    save_dir: Path,
    page_num: int,
    img_idx_on_page: int
) -> Optional[str]:
    """Extracts a single image by xref, saves it, returns ABSOLUTE POSIX path."""
    try:
        img_data = doc.extract_image(xref)
        if not img_data:
            print(f"    Warn: No image data found for xref {xref} on page {page_num}")
            return None

        image_bytes = img_data["image"]
        ext = img_data["ext"]

        image_filename = f"page_{page_num}_img_{img_idx_on_page}.{ext}"
        save_path = save_dir / image_filename

        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.mode in ['RGBA', 'P', 'LA']:
            img_pil = img_pil.convert('RGB') # Convert to RGB for broader compatibility
        img_pil.save(save_path)

        # Return absolute path, resolved, using forward slashes
        abs_posix_path = Path(save_path).resolve().as_posix()
        return abs_posix_path

    except Exception as e:
        print(f"    Error saving image xref {xref} on page {page_num}: {e}")
        return None

# --- Core PDF Processing Function (Worker) ---
def process_single_pdf_hybrid(pdf_path: str) -> Optional[str]:
    """
    Processes a single PDF using the hybrid approach with fixes.
    Uses Fitz text + URL markers for simple pages, user's multimodal func for complex pages.
    Uses absolute paths in markers. Ensures full iteration. Robust error handling.
    """
    filename = Path(pdf_path).name
    sanitized_name = sanitize_filename_for_paths(filename)
    print(f"Processing '{filename}' (Sanitized: {sanitized_name})...")
    pdf_image_dir = Path(IMAGE_SAVE_DIR) / sanitized_name

    # Ensure directories exist *before* processing starts for this PDF
    try:
        os.makedirs(pdf_image_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory '{pdf_image_dir}': {e}. Aborting for this file.")
        return None # Cannot proceed without image directory

    final_text_parts: List[str] = []
    doc = None # Initialize doc to None for finally block

    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count # Get total pages accurately
        print(f"  '{filename}' has {total_pages} pages.")

        # Process all pages using the loop correctly
        for page_num in range(total_pages): # Use 0-based index for Fitz page loading
            page = doc.load_page(page_num) # Load page by 0-based index
            display_page_num = page_num + 1 # Use 1-based index for logging/markers
            print(f"  Processing Page {display_page_num}/{total_pages} for '{filename}'...")
            page_processed_text: Optional[str] = None
            page_separator = f"\n\n---\n\n## Page {display_page_num}\n\n" # Use 1-based index here

            try:
                # Check complexity
                image_list = page.get_images(full=False)
                table_finder = page.find_tables(vertical_strategy="lines", horizontal_strategy="lines") # Example strategies
                is_complex = bool(image_list) or bool(table_finder.tables)

                if not is_complex:
                    # --- Process Simple Page ---
                    print(f"    Page {display_page_num}: Simple text extraction.")
                    # Fix Fitz Flags: Use TEXT_PRESERVE_WHITESPACE for better layout preservation potentially
                    page_text = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_WHITESPACE)
                    if not page_text: page_text = "[EMPTY PAGE]"

                    # Add URL extraction for simple pages
                    page_urls = []
                    links = page.get_links()
                    for link in links:
                        if link.get('kind') == fitz.LINK_URI:
                            url = link.get('uri', '')
                            if url:
                                # Simple approach: Append URL markers at the end
                                url_marker = f"[URL_MARKER|href={url}|END_URL_MARKER]"
                                page_urls.append(url_marker)

                    # Combine text and URL markers
                    page_processed_text = page_text.strip()
                    if page_urls:
                         page_processed_text += "\n\n[URLS_ON_PAGE]\n" + "\n".join(page_urls) + "\n[END_URLS_ON_PAGE]"

                else:
                    # --- Process Complex Page ---
                    print(f"    Page {display_page_num}: Complex page detected. Using multimodal function...")
                    image_absolute_paths_on_page: List[str] = []
                    # Step 3a: Pre-extract & Save Images -> Get Absolute Paths
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        # Pass display_page_num (1-based) for clearer filenames
                        absolute_image_path = extract_and_save_image_simple(
                            doc, xref, pdf_image_dir, display_page_num, img_idx
                        )
                        if absolute_image_path:
                            image_absolute_paths_on_page.append(absolute_image_path)
                            print(f"      Extracted image abs path: {absolute_image_path}")
                        else:
                            print(f"      Warn: Failed to extract/save image xref {xref} on page {display_page_num}")

                    # Step 3b: Render Page
                    print(f"      Rendering page {display_page_num} to image...")
                    pix = page.get_pixmap(dpi=PIXMAP_DPI)
                    page_image_bytes = pix.tobytes("png") # Use PNG for lossless rendering
                    print(f"      Page image size: {len(page_image_bytes) / 1024:.1f} KB")

                    # Step 3c: Construct Detailed Prompt (Using Absolute Paths)
                    prompt_image_paths = "', '".join(image_absolute_paths_on_page)
                    detailed_prompt = f"""Analyze the provided image of page {display_page_num} from document '{filename}'.
Instructions:
1. Perform OCR to extract all text in the correct reading order.
2. Identify any distinct images visible within the page image.
3. Identify any tables visible within the page image.
4. Generate a concise textual explanation for each identified image.
5. The following image file paths correspond to the images on this page, likely in top-to-bottom reading order: ['{prompt_image_paths}']
6. Reconstruct the full text content of the page, maintaining the reading order.
7. When you reference an image in the reconstructed text (at the point where it appears), insert the following marker EXACTLY using the correct absolute path: [IMAGE_MARKER|path=<ABSOLUTE_PATH_FROM_PROVIDED_LIST>|explanation=<YOUR_GENERATED_EXPLANATION>|END_IMAGE_MARKER]. Match the paths from the list ({image_absolute_paths_on_page}) to the images you identified in order.
8. When you reference a table, extract its content (e.g., as text, preserving rows/columns if possible) and wrap it like this: [TABLE_MARKER]<extracted table content>[END_TABLE_MARKER].
9. Identify any hyperlinks in the OCR'd text and mark them like this: [URL_MARKER|href=<extracted_url>|END_URL_MARKER] directly after the link text.
10. Return ONLY the fully reconstructed text content for this page with all specified markers embedded. Do not add any preamble or concluding remarks outside of the reconstructed text.
"""

                    # Step 3d: Call User's Multimodal Function
                    page_processed_text = get_image_explanation(
                        image_as_jpg=page_image_bytes, # Func name implies JPG, but we send PNG bytes - ensure your func handles it
                        prompt_in_detail=detailed_prompt
                    )

                    if page_processed_text is None:
                        print(f"    Warn: Multimodal function returned None for page {display_page_num}. Falling back to simple text + markers.")
                        # Fallback: simple text + append markers for images found
                        page_text = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_WHITESPACE)
                        if not page_text: page_text = "[COMPLEX PAGE PROCESSING FAILED - NO TEXT]"
                        img_markers = []
                        for img_path in image_absolute_paths_on_page:
                             # Could optionally call get_image_explanation just for description here
                             explanation = "Fallback explanation"
                             img_markers.append(f"[IMAGE_MARKER|path={img_path}|explanation={explanation}|END_IMAGE_MARKER]")
                        page_processed_text = page_text.strip() + "\n\n[FALLBACK_IMAGE_MARKERS]\n" + "\n".join(img_markers) + "\n[END_FALLBACK_IMAGE_MARKERS]"


            except Exception as page_proc_err:
                 print(f"    !! Severe Error processing page {display_page_num}: {page_proc_err} !!")
                 traceback.print_exc() # Print full traceback for page errors
                 page_processed_text = f"[ERROR PROCESSING PAGE {display_page_num}: {page_proc_err}]\n"

            # Append processed text for the page
            final_text_parts.append(page_separator.format(page_num=display_page_num))
            # Ensure page_processed_text is a string before stripping
            final_text_parts.append(str(page_processed_text).strip() if page_processed_text is not None else "")


        full_modified_text = "".join(final_text_parts)
        print(f"  Finished processing '{filename}'. Final text length: {len(full_modified_text)}")
        return full_modified_text

    except Exception as e:
        print(f"Critical Error processing PDF file '{pdf_path}': {e}")
        traceback.print_exc() # Print full traceback for file errors
        return None
    finally:
        if doc:
            doc.close() # Ensure document is closed even if errors occur


# --- Main Execution Logic ---
if __name__ == "__main__":
    pdf_files = [str(p) for p in Path(".").glob("*.pdf") if p.is_file()]

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
                # Get result, might raise exception if worker failed unexpectedly
                result_text = future.result()
                results[pdf_path] = result_text
                if result_text is not None:
                    success_count += 1
                else:
                    # Worker function handled its error and returned None
                    print(f"Processing returned None (likely internal error) for: {pdf_path}")
                    failure_count += 1
            except Exception as exc:
                # Exception occurred *within the worker function* and wasn't caught / was re-raised
                # Or an error occurred during future handling itself
                print(f"'{pdf_path}' generated an unhandled exception: {exc}")
                traceback.print_exc()
                results[pdf_path] = None # Mark as failed
                failure_count += 1

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {success_count}")
    print(f"Failed processing:    {failure_count}")

    # Save successful results to text files
    saved_count = 0
    for pdf_path, text_content in results.items():
        # Check for non-empty string content before saving
        if isinstance(text_content, str) and text_content.strip():
            sanitized_name = sanitize_filename_for_paths(Path(pdf_path).name)
            output_filename = Path(TXT_OUTPUT_DIR) / f"{sanitized_name}_hybrid_marked_up.txt"
            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(text_content)
                saved_count += 1
            except Exception as write_err:
                print(f"Error writing output file for '{pdf_path}': {write_err}")
        elif text_content is None:
             print(f"Skipping save for '{pdf_path}' as processing failed.")
        # else: # Optional: Handle cases where text_content is empty string?
             # print(f"Skipping save for '{pdf_path}' as processed text was empty.")


    print(f"Saved {saved_count} marked-up text files to: {os.path.abspath(TXT_OUTPUT_DIR)}")
    print(f"Images saved in subdirectories under: {os.path.abspath(IMAGE_SAVE_DIR)}")
