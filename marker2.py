import fitz  # PyMuPDF
import os
import re
import hashlib
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingJob:
    job_id: str
    pdf_paths: List[Path]
    user_id: str
    output_base_dir: Path
    status: str = "pending"
    
class PDFProcessor:
    def __init__(self, base_output_dir: Path):
        self.base_output_dir = base_output_dir
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = queue.Queue()
        self.job_lock = threading.Lock()
        self.processed_files: Set[Path] = set()
        self.file_lock = threading.Lock()
        
        # Create worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        # Ensure base directories exist
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_user_dirs(self, user_id: str) -> Tuple[Path, Path]:
        """Create and return user-specific output directories."""
        user_base = self.base_output_dir / user_id
        image_dir = user_base / "images"
        text_dir = user_base / "text"
        
        image_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        
        return image_dir, text_dir

    def submit_job(self, pdf_paths: List[Path], user_id: str) -> str:
        """Submit a new processing job."""
        # Generate unique job ID
        job_id = hashlib.sha256(f"{user_id}_{time.time()}".encode()).hexdigest()[:12]
        
        # Create user directories
        image_dir, text_dir = self._get_user_dirs(user_id)
        
        # Create job
        job = ProcessingJob(
            job_id=job_id,
            pdf_paths=pdf_paths,
            user_id=user_id,
            output_base_dir=self.base_output_dir
        )
        
        with self.job_lock:
            self.active_jobs[job_id] = job
            self.job_queue.put(job)
            
        logger.info(f"Submitted job {job_id} for user {user_id} with {len(pdf_paths)} PDFs")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get the status of a specific job."""
        with self.job_lock:
            job = self.active_jobs.get(job_id)
            return job.status if job else None

    def _process_queue(self):
        """Main worker thread processing loop."""
        while True:
            try:
                job = self.job_queue.get()
                logger.info(f"Processing job {job.job_id}")
                
                try:
                    self._process_job(job)
                except Exception as e:
                    logger.error(f"Error processing job {job.job_id}: {e}")
                    with self.job_lock:
                        job.status = "failed"
                
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                time.sleep(1)  # Prevent tight loop on error

    def _process_job(self, job: ProcessingJob):
        """Process a single job with multiple PDFs."""
        image_dir, text_dir = self._get_user_dirs(job.user_id)
        
        with self.job_lock:
            job.status = "processing"
        
        # Process PDFs in parallel
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            future_to_pdf = {
                executor.submit(
                    self._process_single_pdf,
                    pdf_path,
                    image_dir,
                    text_dir
                ): pdf_path for pdf_path in job.pdf_paths
            }
            
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
        
        with self.job_lock:
            job.status = "completed"

    def _process_single_pdf(self, pdf_path: Path, image_dir: Path, text_dir: Path):
        """Process a single PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Check if already processed
        with self.file_lock:
            if pdf_path in self.processed_files:
                logger.info(f"Skipping already processed file: {pdf_path}")
                return
            self.processed_files.add(pdf_path)
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Create PDF-specific directories
        pdf_name = self._sanitize_filename(pdf_path.name)
        pdf_image_dir = image_dir / pdf_name
        pdf_image_dir.mkdir(exist_ok=True)
        
        doc = fitz.open(str(pdf_path))
        text_parts = []
        total_pages = len(doc)
        
        try:
            # Process all pages
            for page_num in range(total_pages):
                actual_page_num = page_num + 1  # Convert to 1-based page numbers
                logger.info(f"Processing page {actual_page_num}/{total_pages} of {pdf_path.name}")
                
                try:
                    page = doc[page_num]
                    page_text = self._process_page(
                        page=page,
                        page_num=actual_page_num,
                        pdf_path=pdf_path,
                        image_dir=pdf_image_dir
                    )
                    
                    text_parts.append(f"\n\n--- Page {actual_page_num} ---\n\n")
                    text_parts.append(page_text)
                    
                except Exception as e:
                    logger.error(f"Error processing page {actual_page_num}: {e}")
                    text_parts.append(f"\n\n--- Page {actual_page_num} ---\n\n")
                    text_parts.append(f"[Error processing page: {str(e)}]\n")
                
        finally:
            doc.close()
            
        # Save complete text output
        output_text = "".join(text_parts)
        output_path = text_dir / f"{pdf_name}_processed.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
            
        logger.info(f"Completed processing {pdf_path}")

    def _process_page(self, page: fitz.Page, page_num: int, pdf_path: Path, image_dir: Path) -> str:
        """Process a single page, handling both simple and complex cases."""
        # Check page complexity
        images = page.get_images(full=True)
        tables = page.find_tables()
        has_complex_content = bool(images) or bool(tables)
        
        if not has_complex_content:
            # Simple text extraction
            return self._extract_simple_page(page)
        
        # Complex page processing
        return self._process_complex_page(
            page=page,
            page_num=page_num,
            pdf_path=pdf_path,
            image_dir=image_dir,
            images=images,
            tables=tables
        )

    def _extract_simple_page(self, page: fitz.Page) -> str:
        """Extract text from a simple page."""
        text = page.get_text(sort=True)
        if not text.strip():
            return "[EMPTY PAGE]\n"
        return text

    def _process_complex_page(
        self,
        page: fitz.Page,
        page_num: int,
        pdf_path: Path,
        image_dir: Path,
        images: List,
        tables: List
    ) -> str:
        """Process a complex page with images and/or tables."""
        try:
            # Extract and save images
            image_paths = []
            for img_idx, img in enumerate(images):
                try:
                    xref = img[0]
                    image_path = self._extract_and_save_image(
                        page.parent,  # doc
                        xref,
                        image_dir,
                        page_num,
                        img_idx
                    )
                    if image_path:
                        image_paths.append(str(image_path.absolute()))
                except Exception as e:
                    logger.error(f"Error extracting image {img_idx} from page {page_num}: {e}")

            # Try complex extraction first
            try:
                text = page.get_text(sort=True)
                
                # Process tables
                for table in tables:
                    table_text = self._extract_table_content(table)
                    if table_text:
                        text = self._insert_table_marker(text, table_text)

                # Add image markers
                for idx, image_path in enumerate(image_paths):
                    marker = f"[IMAGE_MARKER|path={image_path}|explanation=Image {idx + 1} on page {page_num}|END_IMAGE_MARKER]\n"
                    text = self._insert_image_marker(text, marker)

            except Exception as e:
                logger.warning(f"Complex extraction failed for page {page_num}, using simple fallback: {e}")
                # Simple fallback: just get basic text
                text = page.get_text() or "[Failed to extract text]\n"
                
            return text

        except Exception as e:
            logger.error(f"Page processing failed completely for page {page_num}: {e}")
            return "[Page processing failed]\n"

    def _extract_and_save_image(
        self,
        doc: fitz.Document,
        xref: int,
        image_dir: Path,
        page_num: int,
        img_idx: int
    ) -> Optional[Path]:
        """Extract and save a single image."""
        try:
            img_data = doc.extract_image(xref)
            if not img_data:
                return None

            image_bytes = img_data["image"]
            ext = img_data["ext"]
            
            image_path = image_dir / f"page_{page_num}_img_{img_idx}.{ext}"
            
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ['RGBA', 'P', 'LA']:
                img = img.convert('RGB')
            
            img.save(image_path)
            return image_path

        except Exception as e:
            logger.error(f"Error saving image {xref} from page {page_num}: {e}")
            return None

    def _extract_table_content(self, table) -> str:
        """Extract content from a table."""
        try:
            cells = table.cells
            if not cells:
                return ""
                
            table_text = []
            for row in cells:
                row_text = " | ".join(cell.text.strip() for cell in row)
                table_text.append(row_text)
                
            return "\n".join(table_text)
        except Exception as e:
            logger.error(f"Error extracting table content: {e}")
            return ""

    def _insert_table_marker(self, text: str, table_content: str) -> str:
        """Insert table marker at appropriate position."""
        return f"{text}\n[TABLE_MARKER]\n{table_content}\n[END_TABLE_MARKER]\n"

    def _insert_image_marker(self, text: str, marker: str) -> str:
        """Insert image marker at appropriate position."""
        return f"{text}\n{marker}"

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Create a safe string for directory and file names."""
        name = Path(filename).stem
        sanitized = re.sub(r'\W+', '_', name)
        hash_part = hashlib.sha1(filename.encode()).hexdigest()[:8]
        final_name = f"{sanitized[:60]}_{hash_part}".strip('_')
        return final_name if final_name else f"pdf_{hash_part}"

# Usage Example
if __name__ == "__main__":
    base_dir = Path("/absolute/path/to/output")
    processor = PDFProcessor(base_dir)
    
    # Example usage for multiple users/PDFs
    pdf_paths = [Path("/path/to/pdf1.pdf"), Path("/path/to/pdf2.pdf")]
    user_id = "user123"
    
    # Submit job
    job_id = processor.submit_job(pdf_paths, user_id)
    
    # Check status (in real usage, you'd probably want to poll this)
    while True:
        status = processor.get_job_status(job_id)
        if status in ["completed", "failed"]:
            break
        time.sleep(1)
    
    print(f"Job {job_id} finished with status: {status}")
