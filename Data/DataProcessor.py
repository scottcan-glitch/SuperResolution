import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and normalize image data for super-resolution tasks."""
    
    MIN_WIDTH = 1920
    MIN_HEIGHT = 1080
    TARGET_WIDTH = 1920
    TARGET_HEIGHT = 1080
    PATCH_SIZE = 96
    
    def __init__(self, raw_dir: str, output_dir: str):
        """
        Initialize the DataProcessor.
        
        Args:
            raw_dir: Path to raw image directory
            output_dir: Path to save processed images
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_image_files(self, category: str, max_images: Optional[int] = None) -> List[Path]:
        """
        Get image files from a specific category, sorted numerically.
        
        Args:
            category: Category name (e.g., 'bathroom')
            max_images: Maximum number of images to return (None for all)
            
        Returns:
            List of image file paths
        """
        category_dir = self.raw_dir / category
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            return []
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(category_dir.glob(ext))
        
        # Sort numerically by extracting number from filename
        image_files.sort(key=lambda x: self._extract_number(x.stem))
        
        if max_images is None:
            return image_files
        
        return image_files[:max_images]
    
    @staticmethod
    def _extract_number(filename: str) -> int:
        """Extract numeric part from filename for sorting."""
        # Extract all digits from the filename
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0
    
    def validate_resolution(self, image_path: Path) -> bool:
        """
        Check if image meets minimum resolution requirements.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image meets requirements, False otherwise
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return False
            
            height, width = img.shape[:2]
            if width < DataProcessor.MIN_WIDTH or height < DataProcessor.MIN_HEIGHT:
                logger.info(f"Image too small ({width}x{height}): {image_path.name}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating {image_path}: {e}")
            return False
    
    def crop_to_target(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to target resolution (1920x1080) from center.
        
        Args:
            image: Input image array
            
        Returns:
            Cropped image array
        """
        height, width = image.shape[:2]
        
        # Calculate crop coordinates for center crop
        left = (width - DataProcessor.TARGET_WIDTH) // 2
        top = (height - DataProcessor.TARGET_HEIGHT) // 2
        
        cropped = image[
            top:top + DataProcessor.TARGET_HEIGHT,
            left:left + DataProcessor.TARGET_WIDTH
        ]
        
        return cropped
    
    def process_and_filter_category(
        self,
        category: str,
        max_images: Optional[int] = None,
        output_subdir: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Process images from a category, validate and crop them.
        
        Args:
            category: Category name (e.g., 'bathroom')
            max_images: Maximum number of images to process (None for all)
            output_subdir: Subdirectory for output (default: category name)
            
        Returns:
            Tuple of (processed_count, skipped_count)
        """
        if output_subdir is None:
            output_subdir = category
        
        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = self.get_image_files(category, max_images)
        processed = 0
        skipped = 0
        
        for image_path in image_files:
            if not self.validate_resolution(image_path):
                skipped += 1
                continue
            
            try:
                # Read and crop image
                img = cv2.imread(str(image_path))
                if img is None:
                    skipped += 1
                    continue
                
                # Convert BGR to RGB for consistency
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Crop to target resolution
                cropped = self.crop_to_target(img)
                
                # Save processed image
                output_filename = image_path.stem + '.png'
                output_path_file = output_path / output_filename
                
                # Convert back to BGR for OpenCV save
                cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path_file), cropped_bgr)
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed} images from {category}")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                skipped += 1
        
        logger.info(f"Category '{category}' complete: {processed} processed, {skipped} skipped")
        return processed, skipped
    
    def extract_patches(
        self,
        image: np.ndarray,
        patch_size: int = PATCH_SIZE,
        overlap: float = 0.0
    ) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
        """
        Extract patches from an image with adjustable overlap.
        
        Args:
            image: Input image array (H x W x C)
            patch_size: Size of patches (patch_size x patch_size)
            overlap: Overlap ratio between 0 and 1 (e.g., 0.5 for 50% overlap)
            
        Yields:
            Tuple of (patch, (y, x)) coordinates
        """
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")
        
        height, width = image.shape[:2]
        stride = int(patch_size * (1 - overlap))
        
        # Ensure stride is at least 1
        stride = max(1, stride)
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image[y:y + patch_size, x:x + patch_size]
                yield patch, (y, x)
    
    def calculate_color_std(self, patch: np.ndarray) -> float:
        """
        Calculate the standard deviation of color values in a patch.
        Computes std across all channels and pixels.
        
        Args:
            patch: Input patch array (H x W x C)
            
        Returns:
            Standard deviation value
        """
        if patch.size == 0:
            return 0.0
        
        return float(np.std(patch))
    
    def filter_patches_by_std(
        self,
        patches_with_coords: List[Tuple[np.ndarray, Tuple[int, int]]],
        min_std: float = 10.0
    ) -> List[Tuple[np.ndarray, Tuple[int, int], float]]:
        """
        Filter patches based on color standard deviation.
        Removes patches with low std (white skies, uniform colors).
        
        Args:
            patches_with_coords: List of (patch, (y, x)) tuples
            min_std: Minimum std threshold to keep patch
            
        Returns:
            List of (patch, (y, x), std_value) tuples for patches that pass filter
        """
        filtered = []
        
        for patch, coords in patches_with_coords:
            std = self.calculate_color_std(patch)
            
            if std >= min_std:
                filtered.append((patch, coords, std))
        
        return filtered
    
    def extract_and_filter_patches(
        self,
        image_path: Path,
        patch_size: int = PATCH_SIZE,
        overlap: float = 0.0,
        min_std: float = 10.0,
        downscale_factor: Optional[float] = None
    ) -> List[Tuple[np.ndarray, Tuple[int, int], float]]:
        """
        Extract patches from an image and filter by color std.
        
        Args:
            image_path: Path to image file
            patch_size: Size of patches
            overlap: Overlap ratio (0 to 1)
            min_std: Minimum color std threshold
            downscale_factor: Optional scale factor to resize image before patches
            
        Returns:
            List of filtered (patch, (y, x), std) tuples
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return []
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if downscale_factor is not None:
            if downscale_factor <= 0 or downscale_factor >= 1:
                raise ValueError("downscale_factor must be between 0 and 1")
            new_width = max(1, int(img.shape[1] * downscale_factor))
            new_height = max(1, int(img.shape[0] * downscale_factor))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Extract all patches
        patches_with_coords = list(self.extract_patches(img, patch_size, overlap))
        
        # Filter by std
        filtered_patches = self.filter_patches_by_std(patches_with_coords, min_std)
        
        return filtered_patches
    
    def save_patches(
        self,
        patches_with_data: List[Tuple[np.ndarray, Tuple[int, int], float]],
        output_dir: Path,
        image_name: str
    ) -> int:
        """
        Save filtered patches to disk.
        
        Args:
            patches_with_data: List of (patch, (y, x), std) tuples
            output_dir: Directory to save patches
            image_name: Base name of source image
            
        Returns:
            Number of patches saved
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (patch, (y, x), std) in enumerate(patches_with_data):
            # Convert RGB back to BGR for OpenCV save
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            
            filename = f"{image_name}_y{y}_x{x}_std{std:.1f}.png"
            output_path = output_dir / filename
            
            cv2.imwrite(str(output_path), patch_bgr)
        
        return len(patches_with_data)
    
    def process_category_to_patches(
        self,
        category: str,
        target_images_with_patches: int = 100,
        patch_size: int = PATCH_SIZE,
        overlap: float = 0.0,
        min_std: float = 10.0,
        downscale_factor: Optional[float] = None,
        output_subdir: Optional[str] = None
    ) -> dict:
        """
        Process a category: filter images and extract/save patches.
        
        Args:
            category: Category name (e.g., 'bathroom')
            target_images_with_patches: Number of images that must yield patches
            patch_size: Size of patches
            overlap: Overlap ratio (0 to 1)
            min_std: Minimum color std threshold
            downscale_factor: Optional scale factor to resize image before patches
            output_subdir: Subdirectory for output
            
        Returns:
            Dict with processing statistics
        """
        if output_subdir is None:
            output_subdir = category
        
        output_path = self.output_dir / output_subdir / "patches"
        
        image_files = self.get_image_files(category)
        
        stats = {
            'category': category,
            'images_found': len(image_files),
            'images_considered': 0,
            'images_processed': 0,
            'images_skipped': 0,
            'total_patches_extracted': 0,
            'total_patches_saved': 0,
        }
        
        for image_path in image_files:
            if stats['images_processed'] >= target_images_with_patches:
                break
            
            stats['images_considered'] += 1
            if not self.validate_resolution(image_path):
                stats['images_skipped'] += 1
                continue
            
            try:
                # Extract and filter patches
                filtered_patches = self.extract_and_filter_patches(
                    image_path,
                    patch_size=patch_size,
                    overlap=overlap,
                    min_std=min_std,
                    downscale_factor=downscale_factor
                )
                
                stats['total_patches_extracted'] += (
                    len(filtered_patches) if filtered_patches else 0
                )
                
                # Save patches
                if filtered_patches:
                    num_saved = self.save_patches(
                        filtered_patches,
                        output_path,
                        image_path.stem
                    )
                    stats['total_patches_saved'] += num_saved
                    stats['images_processed'] += 1
                
                if stats['images_processed'] % 10 == 0:
                    logger.info(
                        f"Processed {stats['images_processed']} images, "
                        f"saved {stats['total_patches_saved']} patches"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                stats['images_skipped'] += 1
        
        logger.info(f"Category '{category}' complete:")
        logger.info(f"  Images processed: {stats['images_processed']}")
        logger.info(f"  Patches saved: {stats['total_patches_saved']}")
        
        return stats


if __name__ == "__main__":
    # Choose to output filtered patches, or just filtered high-def images
    patch_mode = False

    raw_dir = Path(__file__).parent / "Raw"
    output_dir = Path(__file__).parent / ("HighDefPatches" if patch_mode else "HighDefImages")
    processor = DataProcessor(str(raw_dir), str(output_dir))

    # Process categories in Data/Raw directory
    subfolder_names = [p.name for p in raw_dir.iterdir() if p.is_dir()]

    if patch_mode:
        for subfolder in subfolder_names:
            logger.info(f"Starting processing of {subfolder} category...")
            stats = processor.process_category_to_patches(
                category=subfolder,
                target_images_with_patches=10,
                patch_size=96,
                overlap=0.25,  # 25% overlap between patches
                min_std=10.0,
            )

        logger.info(f"Processing complete: {stats}")
    else:
        for subfolder in subfolder_names:
            logger.info(f"Filtering {subfolder} category...")
            output_category_dir = output_dir / subfolder
            output_category_dir.mkdir(parents=True, exist_ok=True)

            image_files = processor.get_image_files(subfolder)
            copied = 0
            skipped = 0

            for image_path in image_files:
                if not processor.validate_resolution(image_path):
                    skipped += 1
                    continue

                destination = output_category_dir / image_path.name
                shutil.copy2(image_path, destination)
                copied += 1

            logger.info(
                f"{subfolder}: copied {copied} images, skipped {skipped} (too small)"
            )

        logger.info("HighDefImages copy complete")