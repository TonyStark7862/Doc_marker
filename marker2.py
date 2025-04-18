import fitz  # PyMuPDF
import os
import re
import hashlib
import logging
from pathlib import Path
from PIL import Image
import concurrent.futures
import io
import time
import threading
from typing import List, Dict, Tuple, Optional, Any

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, "output_hybrid_images")
TXT_OUTPUT_DIR = os.path.join(BASE_DIR, "output_hybrid_text_files")
PIXMAP_DPI = 150  # Resolution for rendering complex pages
MAX_WORKERS_PDF = 4  # Max concurrent PDFs
MAX_WORKERS_PAGE = 2  # Max concurrent pages per PDF
MIN_IMAGE_SIZE = 100  # Min size for images to extract
MAX_API_CALLS_PER_SEC = 5  # Rate limiting for API calls

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pdf_parser")

# --- User's Multimodal Function ---
# This function should be replaced with the actual implementation
def get_image_explanation(image_as_jpg: bytes, prompt_in_detail: str) -> Optional[str]:
    """
    Function to process page images with a multimodal LLM.
    
    Args:
        image_as_jpg: The page rendered as a JPEG image in bytes
        prompt_in_detail: Detailed prompt with instructions
        
    Returns:
        Processed text with markers embedded
    """
    # Here you would normally call your LLM API
    # For now this is a simple pass-through that will be replaced
    return f"Processed content for image of {len(image_as_jpg)} bytes with prompt: {prompt_in_detail[:100]}..."


# --- Rate Limiter for API Calls ---
class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls_per_sec: int = 5):
        self.max_calls = max_calls_per_sec
        self.interval = 1.0 / max_calls_per_sec
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call
            
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
                
            self.last_call = time.time()


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


def extract_and_save_image(
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

        # Filter out small images
        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.width < MIN_IMAGE_SIZE or img_pil.height < MIN_IMAGE_SIZE:
            return None

        image_filename = f"page_{page_num}_img_{img_idx_on_page}.{ext}"
        save_path = os.path.join(save_dir, image_filename)

        if img_pil.mode in ['RGBA', 'P', 'LA']:
            img_pil = img_pil.convert('RGB')
        img_pil.save(save_path)

        # Return absolute path for reliability
        return os.path.abspath(save_path)

    except Exception as e:
        logger.error(f"    Error saving image xref {xref} on page {page_num}: {e}")
        return None


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = re.compile(r'(https?://\S+)')
    return list(set(url_pattern.findall(text)))


# --- Core PDF Processing ---
class ProcessedResult:
    """Class to track PDF processing results."""
    
    def __init__(self, pdf_path: str, task_id: str):
        self.pdf_path = pdf_path
        self.task_id = task_id
        self.success = False
        self.text_content = None
        self.error_message = None
        self.page_count = 0
        self.processed_pages = 0
        self.output_path = None
        self.start_time = time.time()
        self.text_parts = {}  # page_num -> text
        self.lock = threading.RLock()
        self.callback = None
    
    def set_callback(self, callback):
        """Set callback function to call when processing completes."""
        self.callback = callback
    
    def update_page(self, page_num: int, text: str):
        """Update a processed page."""
        with self.lock:
            self.text_parts[page_num] = text
            self.processed_pages += 1
    
    def is_complete(self):
        """Check if all pages are processed."""
        with self.lock:
            return self.processed_pages >= self.page_count
    
    def finalize(self):
        """Combine all text parts into final result."""
        with self.lock:
            # Combine pages in order
            parts = []
            for page_num in range(self.page_count):
                text = self.text_parts.get(page_num, f"[MISSING PAGE {page_num+1}]")
                parts.append(f"\n\n---\n\n## Page {page_num+1}\n\n")
                parts.append(text)
            
            self.text_content = "".join(parts)
            self.success = True
            
            # Save to file
            sanitized_name = sanitize_filename_for_paths(os.path.basename(self.pdf_path))
            output_filename = f"{sanitized_name}_hybrid_marked_up.txt"
            self.output_path = os.path.join(TXT_OUTPUT_DIR, output_filename)
            
            os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(self.text_content)
            
            # Call callback if set
            if self.callback:
                self.callback(self)


def process_page(
    doc: fitz.Document,
    page_idx: int,
    pdf_path: str,
    pdf_image_dir: str,
    result: ProcessedResult,
    api_limiter: RateLimiter
):
    """Process a single page of a PDF."""
    page_num = page_idx + 1  # 1-indexed for display
    logger.info(f"  Processing Page {page_num}/{result.page_count} for '{os.path.basename(pdf_path)}'...")
    
    try:
        # Access the page
        page = doc[page_idx]
        
        # Check complexity
        images = page.get_images(full=False)
        has_tables = False
        try:
            table_finder = page.find_tables()
            has_tables = bool(table_finder.tables)
        except AttributeError:
            # Table finder not available in this version
            pass
            
        is_complex = bool(images) or has_tables
        
        # Extract text and URLs
        text = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_WHITESPACE)
        urls = extract_urls(text)
        
        # Simple page processing
        if not is_complex:
            logger.info(f"    Page {page_num}: Simple text extraction.")
            
            # Add URL markers
            for url in urls:
                text = text.replace(url, f"{url} [URL_MARKER|href={url}|END_URL_MARKER]")
            
            processed_text = text or "[EMPTY PAGE]\n"
            
        # Complex page processing
        else:
            logger.info(f"    Page {page_num}: Complex page with images/tables.")
            
            # Extract images
            image_paths = []
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                img_path = extract_and_save_image(
                    doc, xref, pdf_image_dir, page_num, img_idx
                )
                if img_path:
                    image_paths.append(img_path)
                    logger.debug(f"      Extracted image: {img_path}")
            
            # Render page
            pix = page.get_pixmap(dpi=PIXMAP_DPI)
            page_image = pix.tobytes("jpeg", quality=90)
            
            # Apply rate limiting
            api_limiter.wait_if_needed()
            
            # Construct prompt similar to original
            image_paths_str = "\n".join([f"- {path}" for path in image_paths])
            urls_str = "\n".join([f"- {url}" for url in urls])
            
            detailed_prompt = f"""Analyze the provided image of page {page_num} from document '{os.path.basename(pdf_path)}'.
Instructions:
1. Perform OCR to extract all text in the correct reading order.
2. Identify any distinct images visible within the page image.
3. Identify any tables visible within the page image.
4. Generate a concise textual explanation for each identified image.
5. The following image file paths correspond to the images on this page, likely in top-to-bottom reading order:
{image_paths_str}
6. The following URLs were detected in the text:
{urls_str if urls else "None detected"}
7. Reconstruct the full text content of the page, maintaining the reading order.
8. When you reference an image in the reconstructed text (at the point where it appears), insert the following marker EXACTLY: [IMAGE_MARKER|path=<CORRECT_PATH_FROM_PROVIDED_LIST>|explanation=<YOUR_GENERATED_EXPLANATION>|END_IMAGE_MARKER]. Match the paths from the list to the images you identified in order.
9. When you reference a table, extract its content (e.g., as text) and wrap it like this: [TABLE_MARKER]<extracted table content>[END_TABLE_MARKER].
10. Mark any hyperlinks in the OCR'd text like this: [URL_MARKER|href=<extracted_url>|END_URL_MARKER] directly after the link text.
11. Return ONLY the fully reconstructed text content for this page with all specified markers embedded.
"""
            # Call multimodal function
            processed_text = get_image_explanation(
                image_as_jpg=page_image,
                prompt_in_detail=detailed_prompt
            )
            
            # Fallback if needed
            if not processed_text:
                logger.warning(f"    Multimodal function returned None for page {page_num}. Falling back to simple text.")
                processed_text = text or "[COMPLEX PAGE PROCESSING FAILED]\n"
        
        # Update result with processed text
        result.update_page(page_idx, processed_text)
        
        # Check if all pages are done
        if result.is_complete():
            result.finalize()
            
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        # Update with error
        result.update_page(page_idx, f"[ERROR PROCESSING PAGE {page_num}: {e}]\n")
        
        # Check if all pages are done (even with errors)
        if result.is_complete():
            result.finalize()


def process_pdf(pdf_path: str, task_id: str = None, callback = None):
    """Process a single PDF file."""
    start_time = time.time()
    filename = os.path.basename(pdf_path)
    sanitized_name = sanitize_filename_for_paths(filename)
    
    if not task_id:
        task_id = sanitized_name
    
    logger.info(f"Processing PDF: {filename} (Task ID: {task_id})")
    
    # Create result object
    result = ProcessedResult(pdf_path, task_id)
    if callback:
        result.set_callback(callback)
    
    # Create API rate limiter
    api_limiter = RateLimiter(MAX_API_CALLS_PER_SEC)
    
    try:
        # Create output directories
        pdf_image_dir = os.path.join(IMAGE_SAVE_DIR, sanitized_name)
        os.makedirs(pdf_image_dir, exist_ok=True)
        os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)
        
        # Open document
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        result.page_count = page_count
        
        # Process pages with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_PAGE) as executor:
            futures = []
            
            # Submit all pages for processing
            for page_idx in range(page_count):
                future = executor.submit(
                    process_page,
                    doc=doc,
                    page_idx=page_idx,
                    pdf_path=pdf_path,
                    pdf_image_dir=pdf_image_dir,
                    result=result,
                    api_limiter=api_limiter
                )
                futures.append(future)
            
            # Wait for all pages to complete
            concurrent.futures.wait(futures)
        
        # Close document
        doc.close()
        
        # Double-check finalization
        if not result.success and result.is_complete():
            result.finalize()
        
        logger.info(f"PDF processing completed for {filename} in {time.time() - start_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF file '{pdf_path}': {e}")
        result.error_message = str(e)
        result.success = False
        
        # Call callback even on error
        if callback:
            callback(result)
            
        return result


# --- Multi-User API ---
class PDFProcessingAPI:
    """API for multiple users to process PDFs."""
    
    def __init__(self):
        self.results = {}  # task_id -> result
        self.user_tasks = {}  # user_id -> [task_ids]
        self.lock = threading.RLock()
    
    def submit_pdf(self, pdf_path: str, user_id: str = "default", 
                  priority: int = 0, callback = None) -> str:
        """Submit a PDF for processing."""
        # Create task ID
        task_id = f"{user_id}_{int(time.time())}_{os.path.basename(pdf_path)}"
        
        # Add to user tasks
        with self.lock:
            if user_id not in self.user_tasks:
                self.user_tasks[user_id] = []
            self.user_tasks[user_id].append(task_id)
        
        # Define wrapper callback to update results
        def _wrapped_callback(result):
            with self.lock:
                self.results[task_id] = result
            if callback:
                callback(result)
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=self._process_pdf_thread,
            args=(pdf_path, task_id, _wrapped_callback)
        )
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _process_pdf_thread(self, pdf_path, task_id, callback):
        """Thread function to process a PDF."""
        result = process_pdf(pdf_path, task_id, callback)
        with self.lock:
            self.results[task_id] = result
    
    def get_task_status(self, task_id: str) -> dict:
        """Get the status of a task."""
        with self.lock:
            result = self.results.get(task_id)
            
        if not result:
            return {"status": "not_found"}
        
        return {
            "status": "complete" if result.success else "in_progress",
            "pdf_path": result.pdf_path,
            "success": result.success,
            "error": result.error_message,
            "page_count": result.page_count,
            "processed_pages": result.processed_pages,
            "output_path": result.output_path,
            "progress": result.processed_pages / result.page_count if result.page_count else 0
        }
    
    def get_user_tasks(self, user_id: str) -> List[str]:
        """Get all tasks for a user."""
        with self.lock:
            return self.user_tasks.get(user_id, []).copy()
    
    def process_multiple_pdfs(self, pdf_paths: List[str], user_id: str = "default", 
                           callback = None) -> List[str]:
        """Process multiple PDFs at once."""
        task_ids = []
        for pdf_path in pdf_paths:
            task_id = self.submit_pdf(pdf_path, user_id, callback=callback)
            task_ids.append(task_id)
        return task_ids
    
    def wait_for_completion(self, task_ids: List[str], timeout: float = None) -> bool:
        """Wait for tasks to complete."""
        start_time = time.time()
        while True:
            all_done = True
            for task_id in task_ids:
                status = self.get_task_status(task_id)
                if status["status"] != "complete":
                    all_done = False
                    break
            
            if all_done:
                return True
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                return False
            
            time.sleep(0.5)


# --- Main Execution ---
def main():
    """Main function to run the PDF parser."""
    # Create API
    api = PDFProcessingAPI()
    
    # Find PDF files in current directory
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    # Define callback
    def on_pdf_complete(result):
        status = "Success" if result.success else "Failed"
        print(f"{status}: {os.path.basename(result.pdf_path)} - {result.processed_pages}/{result.page_count} pages")
    
    # Process all PDFs
    task_ids = api.process_multiple_pdfs(pdf_files, callback=on_pdf_complete)
    
    # Wait for completion
    print("Waiting for processing to complete...")
    api.wait_for_completion(task_ids)
    
    print("All PDFs processed.")


if __name__ == "__main__":
    main()
