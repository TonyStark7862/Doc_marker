import fitz  # PyMuPDF
import os
import re
import hashlib
from pathlib import Path
from PIL import Image
import concurrent.futures
import io
import time
import traceback
from typing import List, Dict, Tuple, Optional, Any

# --- Configuration ---
IMAGE_SAVE_DIR = os.path.abspath("./output_hybrid_images")  # Organised by PDF
TXT_OUTPUT_DIR = os.path.abspath("./output_hybrid_text_files") # Where to save .txt files
PIXMAP_DPI = 150 # Resolution for rendering complex pages

# --- User's Multimodal Function ---
# Keeping as is per request
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
    """Extracts a single image by xref, saves it, returns absolute path."""
    try:
        img_data = doc.extract_image(xref)
        if not img_data:
            return None

        image_bytes = img_data["image"]
        ext = img_data["ext"]

        image_filename = f"page_{page_num}_img_{img_idx_on_page}.{ext}"
        save_path = save_dir / image_filename
        
        # Ensure save_path is absolute
        save_path = os.path.abspath(save_path)

        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.mode in ['RGBA', 'P', 'LA']:
            img_pil = img_pil.convert('RGB')
        img_pil.save(save_path)

        # Return absolute path for consistency
        return save_path

    except Exception as e:
        print(f"    Error saving image xref {xref} on page {page_num}: {e}")
        return None

def extract_urls_from_text(text: str) -> str:
    """
    Extract URLs from text and add URL markers.
    This function helps extract URLs that might be missed in the image explanation function.
    """
    # Basic URL regex pattern - can be expanded for better matching
    url_pattern = r'https?://[^\s)>\]\"\']+|www\.[^\s)>\]\"\']+\.[^\s)>\]\"\']+' 
    
    def replace_with_marker(match):
        url = match.group(0)
        return f"[URL_MARKER|href={url}|END_URL_MARKER]"
    
    return re.sub(url_pattern, replace_with_marker, text)

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
    pdf_image_dir = os.path.abspath(pdf_image_dir)  # Convert to absolute path

    try:
        os.makedirs(pdf_image_dir, exist_ok=True) # Ensure image dir exists
        doc = fitz.open(pdf_path)
        final_text_parts: List[str] = []
        page_separator = f"\n\n---\n\n## Page {{page_num}}\n\n" # Template

        # Properly track and process all pages
        total_pages = len(doc)
        
        for page_idx in range(total_pages):  # Use index to ensure we process every page
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-based page numbering for output
            
            print(f"  Processing Page {page_num}/{total_pages} for '{filename}'...")
            page_processed_text: Optional[str] = None

            try:
                # Check complexity
                image_list = page.get_images(full=False) # Just get xrefs/basic info
                table_finder = page.find_tables()
                is_complex = bool(image_list) or bool(table_finder.tables)

                if not is_complex:
                    # --- Process Simple Page ---
                    print(f"    Page {page_num}: Simple text extraction.")
                    # Fix: Use proper text extraction flags
                    page_processed_text = page.get_text("text", sort=True)
                    
                    # Extract any URLs that might be in the text
                    if page_processed_text:
                        page_processed_text = extract_urls_from_text(page_processed_text)
                    
                    if not page_processed_text: # Handle empty pages
                         page_processed_text = "[EMPTY PAGE]\n"

                else:
                    # --- Process Complex Page ---
                    print(f"    Page {page_num}: Complex page detected (Images/Tables). Using multimodal function...")
                    image_paths_on_page: List[str] = []
                    # Step 3a: Pre-extract & Save Images
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        absolute_image_path = extract_and_save_image_simple(
                            doc, xref, Path(pdf_image_dir), page_num, img_idx
                        )
                        if absolute_image_path:
                            image_paths_on_page.append(absolute_image_path)
                            print(f"      Extracted image path: {absolute_image_path}")
                        else:
                            print(f"      Warn: Failed to extract/save image xref {xref} on page {page_num}")

                    # Step 3b: Render Page
                    print(f"      Rendering page {page_num} to image...")
                    pix = page.get_pixmap(dpi=PIXMAP_DPI)
                    page_image_bytes = pix.tobytes("png") # Use PNG for lossless rendering
                    print(f"      Page image size: {len(page_image_bytes) / 1024:.1f} KB")

                    # Step 3c: Construct Detailed Prompt
                    # Use absolute paths in the prompt
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

                    # Step 3d: Call User's Multimodal Function with robust fallback
                    try:
                        page_processed_text = get_image_explanation(
                            image_as_jpg=page_image_bytes, # Assuming func takes bytes
                            prompt_in_detail=detailed_prompt
                        )
                    except Exception as multimodal_error:
                        print(f"    Error in multimodal function for page {page_num}: {multimodal_error}")
                        traceback.print_exc()
                        page_processed_text = None  # Force fallback
                        
                    # Fallback logic - if multimodal function fails or returns None
                    if page_processed_text is None:
                        print(f"    Warn: Multimodal function failed for page {page_num}. Falling back to simple text extraction.")
                        try:
                            # First fallback - try normal Fitz text extraction
                            page_processed_text = page.get_text("text", sort=True)
                            if page_processed_text:
                                page_processed_text = extract_urls_from_text(page_processed_text)
                                print(f"    Successfully extracted text using fallback method for page {page_num}")
                            else:
                                # Second fallback - if text extraction returns empty
                                print(f"    Warn: Fallback text extraction returned empty for page {page_num}")
                                page_processed_text = "[COMPLEX PAGE PROCESSING FAILED - NO TEXT EXTRACTED]\n"
                        except Exception as fallback_error:
                            print(f"    Error in fallback text extraction for page {page_num}: {fallback_error}")
                            traceback.print_exc()
                            page_processed_text = f"[COMPLEX PAGE PROCESSING FAILED WITH ERROR: {str(fallback_error)}]\n"

            except Exception as page_proc_err:
                 print(f"    Error processing page {page_num}: {page_proc_err}. Skipping page content.")
                 traceback.print_exc()  # Print full traceback for better debugging
                 page_processed_text = f"[ERROR PROCESSING PAGE {page_num}: {str(page_proc_err)}]\n"

            # Append processed text for the page
            final_text_parts.append(page_separator.format(page_num=page_num))
            final_text_parts.append(page_processed_text.strip() if page_processed_text else "")

        doc.close()
        full_modified_text = "".join(final_text_parts)
        print(f"  Finished processing '{filename}'. Final text length: {len(full_modified_text)}")
        return full_modified_text

    except Exception as e:
        print(f"Error processing PDF file '{pdf_path}': {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None

# --- Thread Management for Multi-User Support ---
class PDFProcessorManager:
    """Manages PDF processing threads and resources to ensure stability with multiple PDFs/users."""
    
    def __init__(self, max_concurrent_pdfs=4):
        self.max_workers = min(max_concurrent_pdfs, os.cpu_count() or 2) 
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_jobs = {}
        self.results = {}
        self.lock = concurrent.futures.thread.threading.Lock()
    
    def submit_pdf(self, pdf_path):
        """Submit a PDF for processing and return a job ID."""
        with self.lock:
            future = self.executor.submit(process_single_pdf_hybrid, pdf_path)
            job_id = f"job_{len(self.active_jobs) + 1}_{int(time.time())}"
            self.active_jobs[job_id] = (future, pdf_path)
            # Add callback to handle future completion
            future.add_done_callback(lambda f, jid=job_id: self._handle_completion(jid, f))
            return job_id
    
    def _handle_completion(self, job_id, future):
        """Internal callback to handle job completion."""
        with self.lock:
            if job_id in self.active_jobs:
                _, pdf_path = self.active_jobs[job_id]
                try:
                    result = future.result()
                    self.results[job_id] = (pdf_path, result)
                    print(f"Job {job_id} for '{pdf_path}' completed successfully")
                    # Save result to file
                    if result is not None:
                        self._save_result_to_file(pdf_path, result)
                except Exception as e:
                    print(f"Job {job_id} for '{pdf_path}' failed: {e}")
                    traceback.print_exc()
                    self.results[job_id] = (pdf_path, None)
                
                # Remove from active jobs
                del self.active_jobs[job_id]
    
    def get_job_status(self, job_id):
        """Get the status of a job."""
        with self.lock:
            if job_id in self.active_jobs:
                future, pdf_path = self.active_jobs[job_id]
                status = "running"
                if future.done():
                    status = "done" if not future.exception() else "failed"
                return {"status": status, "pdf_path": pdf_path}
            elif job_id in self.results:
                pdf_path, result = self.results[job_id]
                status = "completed" if result is not None else "failed"
                return {"status": status, "pdf_path": pdf_path}
            return {"status": "unknown"}
    
    def get_all_jobs(self):
        """Get status of all jobs."""
        with self.lock:
            all_jobs = {}
            # Active jobs
            for job_id, (future, pdf_path) in self.active_jobs.items():
                status = "running"
                if future.done():
                    status = "done" if not future.exception() else "failed"
                all_jobs[job_id] = {"status": status, "pdf_path": pdf_path}
            
            # Completed/failed jobs
            for job_id, (pdf_path, result) in self.results.items():
                status = "completed" if result is not None else "failed"
                all_jobs[job_id] = {"status": status, "pdf_path": pdf_path}
            
            return all_jobs
    
    def _save_result_to_file(self, pdf_path, text_content):
        """Save the result to a text file."""
        if text_content is not None:
            sanitized_name = sanitize_filename_for_paths(Path(pdf_path).name)
            output_filename = os.path.join(TXT_OUTPUT_DIR, f"{sanitized_name}_hybrid_marked_up.txt")
            output_filename = os.path.abspath(output_filename)
            try:
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(text_content)
                print(f"Saved output file: {output_filename}")
            except Exception as write_err:
                print(f"Error writing output file for '{pdf_path}': {write_err}")
                traceback.print_exc()
    
    def shutdown(self, wait=True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)

# --- Main Execution Logic ---
if __name__ == "__main__":
    # Create output directories with absolute paths
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)
    print(f"Image save directory: {IMAGE_SAVE_DIR}")
    print(f"Text output directory: {TXT_OUTPUT_DIR}")
    
    # pdf_files = ["./doc1.pdf", "./doc2.pdf"] # Or provide specific paths
    pdf_files = [os.path.abspath(str(p)) for p in Path(".").glob("*.pdf") if p.is_file()] # PDFs in current dir with absolute paths
    
    if not pdf_files:
        print("No PDF files found in the current directory.")
        exit()

    print(f"Found {len(pdf_files)} PDF files to process.")
    
    # Create processor manager
    processor = PDFProcessorManager(max_concurrent_pdfs=4)
    
    # Submit all PDFs for processing
    job_ids = [processor.submit_pdf(pdf_path) for pdf_path in pdf_files]
    print(f"Submitted {len(job_ids)} jobs for processing.")
    
    # Wait for all jobs to complete
    try:
        while any(processor.get_job_status(job_id)["status"] == "running" for job_id in job_ids):
            time.sleep(1)  # Check every second
            
        # Print final status
        print("\n--- Processing Complete ---")
        all_jobs = processor.get_all_jobs()
        success_count = sum(1 for job_id in job_ids if all_jobs[job_id]["status"] == "completed")
        failure_count = sum(1 for job_id in job_ids if all_jobs[job_id]["status"] == "failed")
        
        print(f"Successfully processed: {success_count}")
        print(f"Failed processing:      {failure_count}")
        print(f"Saved marked-up text files to: {TXT_OUTPUT_DIR}")
        print(f"Images saved in subdirectories under: {IMAGE_SAVE_DIR}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Shutting down gracefully...")
    finally:
        processor.shutdown()
