import fitz  # PyMuPDF
import os
import re
import hashlib
import logging
import time
import threading
import queue
import concurrent.futures
from pathlib import Path
from PIL import Image
import io
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Callable, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pdf_parser")

# Global settings
class Config:
    # Output directories (absolute paths)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    IMAGE_DIR = os.path.join(BASE_DIR, "output_images")
    TEXT_DIR = os.path.join(BASE_DIR, "output_text")
    
    # Processing settings
    DPI = 150  # Resolution for rendering pages
    MAX_PDF_WORKERS = 4  # Max concurrent PDFs
    MAX_PAGE_WORKERS = 4  # Max concurrent pages per PDF
    MIN_IMAGE_SIZE = 100  # Minimum size (px) to extract images
    PAGE_TIMEOUT = 120  # Seconds per page
    
    # Image and URL detection
    URL_PATTERN = re.compile(r'(https?://\S+)')
    IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    
    # API rate limiting 
    MAX_API_CALLS_PER_SECOND = 5


# Result data structure
@dataclass
class ProcessingResult:
    """Tracks the result of processing a PDF."""
    task_id: str
    pdf_path: str
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    error: Optional[str] = None
    page_count: int = 0
    processed_pages: int = 0
    output_path: Optional[str] = None
    image_dir: Optional[str] = None


# Helper for API rate limiting
class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, calls_per_second: int):
        self.calls_per_second = max(1, calls_per_second)
        self.interval = 1.0 / self.calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait(self):
        """Wait if necessary to maintain rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.interval:
                # Need to wait
                sleep_time = self.interval - time_since_last_call
                time.sleep(sleep_time)
            
            # Update last call time
            self.last_call_time = time.time()


# Multimodal LLM function
def analyze_page_image(image_bytes: bytes, prompt: str, rate_limiter: RateLimiter) -> str:
    """
    Function that calls a multimodal LLM to analyze page images.
    Replace this with your actual implementation.
    """
    # Apply rate limiting
    rate_limiter.wait()
    
    # This is a placeholder - replace with your actual LLM call
    # In a real implementation, this would call your multimodal API
    return f"Analyzed content for image ({len(image_bytes) / 1024:.1f} KB) with prompt: {prompt[:100]}..."


# File and path handling functions
def create_safe_filename(original_path: str) -> str:
    """Create a safe, unique filename based on the original."""
    base_name = os.path.basename(original_path)
    name_part = os.path.splitext(base_name)[0]
    
    # Remove non-alphanumeric characters
    safe_name = re.sub(r'[^\w\-_]', '_', name_part)
    
    # Add hash for uniqueness
    file_hash = hashlib.md5(original_path.encode()).hexdigest()[:8]
    
    return f"{safe_name[:50]}_{file_hash}"


def setup_directories(pdf_path: str) -> Tuple[str, str]:
    """
    Set up directories for a specific PDF.
    Returns (image_dir, pdf_id)
    """
    pdf_id = create_safe_filename(pdf_path)
    
    # Create base directories
    os.makedirs(Config.IMAGE_DIR, exist_ok=True)
    os.makedirs(Config.TEXT_DIR, exist_ok=True)
    
    # Create PDF-specific image directory
    pdf_image_dir = os.path.join(Config.IMAGE_DIR, pdf_id)
    os.makedirs(pdf_image_dir, exist_ok=True)
    
    return pdf_image_dir, pdf_id


def extract_image(
    doc: fitz.Document, 
    xref: int, 
    page_num: int, 
    img_idx: int, 
    output_dir: str
) -> Optional[str]:
    """
    Extract an image from a PDF and save it.
    Returns the absolute path to the saved image.
    """
    try:
        # Extract the image
        img_data = doc.extract_image(xref)
        if not img_data:
            return None
        
        # Create a PIL Image for processing
        img = Image.open(io.BytesIO(img_data["image"]))
        
        # Skip small images (likely icons or bullets)
        if img.width < Config.MIN_IMAGE_SIZE or img.height < Config.MIN_IMAGE_SIZE:
            return None
        
        # Convert transparency if needed
        if img.mode in ['RGBA', 'P', 'LA']:
            img = img.convert('RGB')
        
        # Create filename and path
        img_filename = f"page_{page_num+1:03d}_img_{img_idx:03d}.{img_data['ext']}"
        img_path = os.path.join(output_dir, img_filename)
        
        # Save the image
        img.save(img_path)
        
        # Return absolute path
        return os.path.abspath(img_path)
        
    except Exception as e:
        logger.error(f"Failed to extract image (page {page_num+1}, xref {xref}): {e}")
        return None


# PDF Processing Core
class PDFProcessor:
    """Core processing engine for PDFs."""
    
    def __init__(self):
        self.results = {}  # task_id -> ProcessingResult
        self.results_lock = threading.Lock()
        self.page_queue = queue.PriorityQueue()
        self.running = True
        self.workers = []
        self.rate_limiter = RateLimiter(Config.MAX_API_CALLS_PER_SECOND)
    
    def start_workers(self, num_workers: int):
        """Start worker threads for page processing."""
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"PageWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Main loop for worker threads."""
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    _, task = self.page_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the page
                try:
                    self._process_page(task)
                except Exception as e:
                    logger.error(f"Error processing page {task['page_num']+1}: {e}")
                finally:
                    self.page_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _process_page(self, task):
        """Process a single page of a PDF."""
        doc = task['doc']
        page_idx = task['page_num']
        page_num = page_idx + 1  # 1-indexed for display
        task_id = task['task_id']
        image_dir = task['image_dir']
        
        logger.info(f"Processing page {page_num}/{task['total_pages']} of {os.path.basename(task['pdf_path'])}")
        
        try:
            # Get the page
            page = doc[page_idx]
            
            # Check if page has images or tables
            images = page.get_images(full=False)
            has_tables = False
            try:
                tables = page.find_tables()
                has_tables = bool(tables.tables)
            except (AttributeError, Exception):
                # Table detection not available or failed
                pass
                
            is_complex = bool(images) or has_tables
            
            # Get basic text and URLs regardless of complexity
            text = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_WHITESPACE)
            urls = list(set(Config.URL_PATTERN.findall(text)))
            
            # Process simple page
            if not is_complex:
                logger.info(f"Page {page_num}: Simple text extraction")
                
                # Add URL markers to text
                for url in urls:
                    text = text.replace(url, f"{url} [URL_MARKER|href={url}|END_URL_MARKER]")
                
                page_text = text or "[EMPTY PAGE]"
            
            # Process complex page with images/tables
            else:
                logger.info(f"Page {page_num}: Complex page with images/tables")
                
                # Extract images
                image_paths = []
                for img_idx, img_info in enumerate(images):
                    xref = img_info[0]
                    img_path = extract_image(doc, xref, page_idx, img_idx, image_dir)
                    if img_path:
                        image_paths.append(img_path)
                
                # Render the page
                pix = page.get_pixmap(dpi=Config.DPI)
                page_bytes = pix.tobytes("jpeg", quality=90)
                
                # Create detailed prompt
                image_list = "\n".join([f"- {path}" for path in image_paths])
                url_list = "\n".join([f"- {url}" for url in urls])
                
                prompt = f"""Analyze the provided image of page {page_num} from document '{os.path.basename(task['pdf_path'])}'.
Instructions:
1. Perform OCR to extract all text in the correct reading order.
2. Identify any distinct images visible within the page image.
3. Identify any tables visible within the page image.
4. Generate a concise textual explanation for each identified image.
5. The following image file paths correspond to the images on this page, likely in top-to-bottom reading order:
{image_list if image_paths else "No extracted images."}
6. Reconstruct the full text content of the page, maintaining the reading order.
7. When you reference an image in the reconstructed text (at the point where it appears), insert the following marker EXACTLY: [IMAGE_MARKER|path=<CORRECT_PATH_FROM_PROVIDED_LIST>|explanation=<YOUR_GENERATED_EXPLANATION>|END_IMAGE_MARKER]. Match the paths from the list ({image_paths}) to the images you identified in order.
8. When you reference a table, extract its content (e.g., as text) and wrap it like this: [TABLE_MARKER]<extracted table content>[END_TABLE_MARKER].
9. Mark any hyperlinks in the OCR'd text like this: [URL_MARKER|href=<extracted_url>|END_URL_MARKER] directly after the link text. The following URLs were detected: {url_list if urls else "None detected"}
10. Return ONLY the fully reconstructed text content for this page with all specified markers embedded. Do not add any preamble or concluding remarks outside of the reconstructed text.
"""
                # Call multimodal function
                page_text = analyze_page_image(page_bytes, prompt, self.rate_limiter)
                
                # Fallback if empty result
                if not page_text:
                    logger.warning(f"Empty result from multimodal function for page {page_num}")
                    page_text = text or "[COMPLEX PAGE EXTRACTION FAILED]"
            
            # Update the result
            with self.results_lock:
                result = self.results.get(task_id)
                if result:
                    # Ensure text_parts exists
                    if not hasattr(result, 'text_parts'):
                        result.text_parts = {}
                    
                    # Store page text
                    result.text_parts[page_idx] = page_text
                    result.processed_pages += 1
                    
                    # Check if all pages are done
                    if result.processed_pages >= result.page_count:
                        self._finalize_result(result)
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            
            # Update the result with error
            with self.results_lock:
                result = self.results.get(task_id)
                if result:
                    # Ensure text_parts exists
                    if not hasattr(result, 'text_parts'):
                        result.text_parts = {}
                    
                    # Store error text
                    result.text_parts[page_idx] = f"[ERROR PROCESSING PAGE {page_num}: {e}]"
                    result.processed_pages += 1
                    
                    # Check if all pages are done despite errors
                    if result.processed_pages >= result.page_count:
                        self._finalize_result(result)
    
    def _finalize_result(self, result):
        """Finalize a processing result."""
        try:
            # Combine all page texts in order
            all_text = []
            for page_idx in range(result.page_count):
                page_num = page_idx + 1
                page_text = result.text_parts.get(page_idx, f"[MISSING PAGE {page_num}]")
                all_text.append(f"\n\n---\n\n## Page {page_num}\n\n")
                all_text.append(page_text)
            
            # Set final text content
            result.text_content = "".join(all_text)
            
            # Create output filename
            pdf_id = create_safe_filename(result.pdf_path)
            output_filename = f"{pdf_id}_marked_up.txt"
            output_path = os.path.join(Config.TEXT_DIR, output_filename)
            
            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.text_content)
            
            # Update result
            result.success = True
            result.output_path = output_path
            result.completed_at = time.time()
            
            logger.info(f"Completed processing {os.path.basename(result.pdf_path)}: {result.processed_pages} pages")
            
            # Call callback if exists
            if hasattr(result, 'callback') and result.callback:
                result.callback(result)
                
        except Exception as e:
            logger.error(f"Error finalizing result: {e}")
            result.success = False
            result.error = str(e)
            result.completed_at = time.time()
            
            # Call callback even on error
            if hasattr(result, 'callback') and result.callback:
                result.callback(result)
    
    def process_pdf(self, pdf_path: str, task_id: str = None, 
                   priority: int = 0, callback: Callable = None) -> str:
        """
        Process a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            task_id: Optional custom task ID
            priority: Processing priority (higher = processed sooner)
            callback: Function to call when processing completes
            
        Returns:
            Task ID for tracking
        """
        # Validate PDF exists
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create task ID if not provided
        if not task_id:
            task_id = f"task_{int(time.time())}_{os.path.basename(pdf_path)}"
        
        # Set up directories
        image_dir, pdf_id = setup_directories(pdf_path)
        
        # Create result object
        result = ProcessingResult(
            task_id=task_id,
            pdf_path=pdf_path,
            started_at=time.time(),
            image_dir=image_dir
        )
        
        # Add callback
        result.callback = callback
        result.text_parts = {}
        
        # Store initial result
        with self.results_lock:
            self.results[task_id] = result
        
        # Make sure workers are started
        if not self.workers:
            self.start_workers(Config.MAX_PAGE_WORKERS)
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            # Update result with page count
            with self.results_lock:
                result = self.results[task_id]
                result.page_count = page_count
            
            # Submit each page for processing
            for page_idx in range(page_count):
                task = {
                    'doc': doc,
                    'page_num': page_idx,
                    'total_pages': page_count,
                    'pdf_path': pdf_path,
                    'task_id': task_id,
                    'image_dir': image_dir
                }
                
                # Lower priority number = higher actual priority
                self.page_queue.put((100 - priority, task))
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error starting PDF processing for {pdf_path}: {e}")
            
            # Update result with error
            with self.results_lock:
                result = self.results[task_id]
                result.success = False
                result.error = str(e)
                result.completed_at = time.time()
                
                # Call callback on error
                if callback:
                    callback(result)
            
            return task_id
    
    def get_result(self, task_id: str) -> Optional[ProcessingResult]:
        """Get the current status/result of a processing task."""
        with self.results_lock:
            return self.results.get(task_id)
    
    def shutdown(self):
        """Shutdown the processor."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)


# Multi-user API
class PDFParserAPI:
    """API for multi-user PDF processing."""
    
    def __init__(self):
        self.processor = PDFProcessor()
        self.user_tasks = {}  # user_id -> [task_ids]
        self.user_lock = threading.Lock()
    
    def submit_pdf(self, pdf_path: str, user_id: str = "default", 
                  priority: int = 0, callback: Callable = None) -> str:
        """
        Submit a PDF for processing.
        
        Args:
            pdf_path: Path to the PDF file
            user_id: User identifier for multi-user support
            priority: Processing priority (higher = processed sooner)
            callback: Function to call when processing completes
            
        Returns:
            Task ID for tracking
        """
        # Create task ID with user prefix
        task_id = f"{user_id}_{int(time.time())}_{os.path.basename(pdf_path)}"
        
        # Track for this user
        with self.user_lock:
            if user_id not in self.user_tasks:
                self.user_tasks[user_id] = []
            self.user_tasks[user_id].append(task_id)
        
        # Process the PDF
        return self.processor.process_pdf(
            pdf_path=pdf_path,
            task_id=task_id,
            priority=priority,
            callback=callback
        )
    
    def process_multiple_pdfs(self, pdf_paths: List[str], user_id: str = "default",
                             priority: int = 0, callback: Callable = None) -> List[str]:
        """
        Process multiple PDFs for a user.
        
        Args:
            pdf_paths: List of PDF file paths
            user_id: User identifier
            priority: Processing priority
            callback: Function to call when each PDF completes
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for pdf_path in pdf_paths:
            task_id = self.submit_pdf(
                pdf_path=pdf_path,
                user_id=user_id,
                priority=priority,
                callback=callback
            )
            task_ids.append(task_id)
        return task_ids
    
    def get_task_status(self, task_id: str) -> Dict:
        """
        Get the status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Status dictionary
        """
        result = self.processor.get_result(task_id)
        if not result:
            return {"status": "not_found", "task_id": task_id}
        
        # Calculate progress
        progress = 0
        if result.page_count > 0:
            progress = result.processed_pages / result.page_count
        
        # Calculate processing time
        processing_time = 0
        if result.completed_at > 0:
            processing_time = result.completed_at - result.started_at
        elif result.started_at > 0:
            processing_time = time.time() - result.started_at
        
        return {
            "status": "complete" if result.success else "in_progress",
            "task_id": result.task_id,
            "pdf_path": result.pdf_path,
            "success": result.success,
            "error": result.error,
            "page_count": result.page_count,
            "processed_pages": result.processed_pages,
            "progress": progress,
            "processing_time": processing_time,
            "output_path": result.output_path
        }
    
    def get_user_tasks(self, user_id: str) -> List[str]:
        """Get all tasks for a specific user."""
        with self.user_lock:
            return self.user_tasks.get(user_id, []).copy()
    
    def wait_for_completion(self, task_ids: List[str], timeout: float = None) -> bool:
        """
        Wait for tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum seconds to wait (None = wait forever)
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        if not task_ids:
            return True
            
        start_time = time.time()
        while True:
            # Check all tasks
            all_complete = True
            for task_id in task_ids:
                status = self.get_task_status(task_id)
                if status["status"] != "complete":
                    all_complete = False
                    break
            
            # Return if all tasks are done
            if all_complete:
                return True
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    return False
            
            # Sleep briefly
            time.sleep(0.5)
    
    def shutdown(self):
        """Shutdown the API."""
        self.processor.shutdown()


# Example usage
def main():
    """Example usage of the PDF parser."""
    # Create API
    api = PDFParserAPI()
    
    # Find PDF files
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    # Define callback
    def on_complete(result):
        status = "Success" if result.success else "Failed"
        print(f"{status}: {os.path.basename(result.pdf_path)} - {result.processed_pages}/{result.page_count} pages")
    
    # Submit PDFs for processing
    task_ids = api.process_multiple_pdfs(pdf_files, callback=on_complete)
    
    # Wait for completion
    print("Waiting for processing to complete...")
    api.wait_for_completion(task_ids)
    
    print("All PDFs processed successfully!")
    api.shutdown()


if __name__ == "__main__":
    main()
