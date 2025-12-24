"""
Demonstration script for Digital Image Processing Suite
Shows practical applications of all implemented algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom implementations
from image_processing_suite import (
    SpatialFilters, EdgeDetection, 
    MorphologicalOperations, FrequencyDomain
 )


def load_image(path: str) -> np.ndarray:
    """Load image and convert to grayscale if needed"""
    try:
        from PIL import Image
        img = Image.open(path)
        return np.array(img)
    except ImportError:
        print("PIL not available, using matplotlib")
        img = plt.imread(path)
        return (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)


def create_synthetic_test_images():
    """Create synthetic test images for demonstration"""
    
    # 1. Simple shapes
    shapes = np.zeros((200, 200))
    shapes[50:100, 50:100] = 255  # Square
    for i in range(200):
        for j in range(200):
            if (i - 150)**2 + (j - 150)**2 <= 30**2:
                shapes[i, j] = 255  # Circle
    
    # 2. Noisy image
    noisy = shapes.copy()
    noise = np.random.normal(0, 25, shapes.shape)
    noisy = np.clip(noisy + noise, 0, 255)
    
    # 3. Gradient image
    gradient = np.zeros((200, 200))
    for i in range(200):
        gradient[i, :] = (i / 200) * 255
    
    return shapes, noisy, gradient


def demo_spatial_filters():
    """Demonstrate spatial domain filtering"""
    print("\n" + "="*60)
    print("SPATIAL DOMAIN FILTERING DEMONSTRATION")
    print("="*60)
    
    shapes, noisy, _ = create_synthetic_test_images()
    
    # Create Gaussian kernel manually
    print("\n1. Gaussian Blur (σ=1.0, kernel=5x5)")
    kernel = SpatialFilters.create_gaussian_kernel(5, 1.0)
    print(f"   Kernel sum: {np.sum(kernel):.6f} (should be ~1.0)")
    print(f"   Kernel center value: {kernel[2,2]:.6f}")
    
    blurred = SpatialFilters.gaussian_blur(noisy, kernel_size=5, sigma=1.0)
    noise_reduction = np.std(noisy) - np.std(blurred)
    print(f"   Noise reduction: {noise_reduction:.2f} (std deviation decreased)")
    
    # Laplacian edge detection
    print("\n2. Laplacian Edge Detection")
    edges_lap = SpatialFilters.laplacian_filter(shapes)
    edge_pixels = np.sum(np.abs(edges_lap) > 10)
    print(f"   Edge pixels detected: {edge_pixels}")
    print(f"   Edge percentage: {(edge_pixels/edges_lap.size)*100:.2f}%")
    
    # Median filter for salt-and-pepper noise
    print("\n3. Median Filter (3x3)")
    salt_pepper = shapes.copy()
    salt_pepper[np.random.random(shapes.shape) < 0.02] = 255
    salt_pepper[np.random.random(shapes.shape) < 0.02] = 0
    
    median_filtered = SpatialFilters.median_filter(salt_pepper, kernel_size=3)
    print(f"   Original noise pixels: {np.sum((salt_pepper == 0) | (salt_pepper == 255))}")
    print(f"   After filtering: {np.sum((median_filtered == 0) | (median_filtered == 255))}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original (Noisy)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('Gaussian Blur')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(edges_lap), cmap='gray')
    axes[0, 2].set_title('Laplacian Edges')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(salt_pepper, cmap='gray')
    axes[1, 0].set_title('Salt & Pepper Noise')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(median_filtered, cmap='gray')
    axes[1, 1].set_title('Median Filtered')
    axes[1, 1].axis('off')
    
    # Show kernel
    axes[1, 2].imshow(kernel, cmap='hot')
    axes[1, 2].set_title('Gaussian Kernel')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('spatial_filters_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'spatial_filters_demo.png'")


def demo_edge_detection():
    """Demonstrate edge detection algorithms"""
    print("\n" + "="*60)
    print("EDGE DETECTION ALGORITHMS DEMONSTRATION")
    print("="*60)
    
    shapes, _, _ = create_synthetic_test_images()
    
    # Sobel operator
    print("\n1. Sobel Edge Detection")
    magnitude, grad_x, grad_y = EdgeDetection.sobel_operator(shapes)
    print(f"   Gradient magnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    print(f"   X-gradient range: [{grad_x.min():.2f}, {grad_x.max():.2f}]")
    print(f"   Y-gradient range: [{grad_y.min():.2f}, {grad_y.max():.2f}]")
    
    # Canny edge detection
    print("\n2. Canny Edge Detection")
    print("   Step 1: Gaussian smoothing...")
    print("   Step 2: Gradient calculation...")
    print("   Step 3: Non-maximum suppression...")
    print("   Step 4: Hysteresis thresholding...")
    
    canny_edges = EdgeDetection.canny_edge_detection(
        shapes, low_threshold=50, high_threshold=150, sigma=1.0
    )
    edge_pixels = np.sum(canny_edges > 0)
    print(f"   Final edge pixels: {edge_pixels}")
    print(f"   Edge density: {(edge_pixels/canny_edges.size)*100:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(shapes, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(magnitude, cmap='gray')
    axes[0, 1].set_title('Sobel Magnitude')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(grad_x, cmap='gray')
    axes[0, 2].set_title('Sobel X-Gradient')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(grad_y, cmap='gray')
    axes[1, 0].set_title('Sobel Y-Gradient')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(canny_edges, cmap='gray')
    axes[1, 1].set_title('Canny Edges (Complete)')
    axes[1, 1].axis('off')
    
    # Compare thresholds
    threshold_simple = magnitude > 100
    axes[1, 2].imshow(threshold_simple, cmap='gray')
    axes[1, 2].set_title('Simple Threshold (comparison)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'edge_detection_demo.png'")


def demo_morphological_operations():
    """Demonstrate morphological transformations"""
    print("\n" + "="*60)
    print("MORPHOLOGICAL OPERATIONS DEMONSTRATION")
    print("="*60)
    
    # Create binary image with noise
    binary = np.zeros((200, 200), dtype=np.uint8)
    binary[50:150, 50:150] = 255
    
    # Add small noise
    noise_points = np.random.random((200, 200)) < 0.05
    binary[noise_points] = 255 - binary[noise_points]
    
    # Test different structuring elements
    print("\n1. Structuring Elements")
    square_se = MorphologicalOperations.create_structuring_element('square', 5)
    cross_se = MorphologicalOperations.create_structuring_element('cross', 5)
    circle_se = MorphologicalOperations.create_structuring_element('circle', 5)
    
    print(f"   Square SE: {np.sum(square_se)} active pixels")
    print(f"   Cross SE: {np.sum(cross_se)} active pixels")
    print(f"   Circle SE: {np.sum(circle_se)} active pixels")
    
    # Apply operations
    print("\n2. Basic Operations")
    dilated = MorphologicalOperations.dilate(binary, square_se)
    eroded = MorphologicalOperations.erode(binary, square_se)
    opened = MorphologicalOperations.opening(binary, square_se)
    closed = MorphologicalOperations.closing(binary, square_se)
    gradient = MorphologicalOperations.morphological_gradient(binary, square_se)
    
    print(f"   Dilation increased pixels by: {np.sum(dilated) - np.sum(binary)}")
    print(f"   Erosion decreased pixels by: {np.sum(binary) - np.sum(eroded)}")
    print(f"   Opening removed noise: {np.sum(binary) - np.sum(opened)} pixels")
    print(f"   Closing filled gaps: {np.sum(closed) - np.sum(binary)} pixels")
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('Original (with noise)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(dilated, cmap='gray')
    axes[0, 1].set_title('Dilation (expand)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(eroded, cmap='gray')
    axes[0, 2].set_title('Erosion (shrink)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(gradient, cmap='gray')
    axes[0, 3].set_title('Morphological Gradient')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(opened, cmap='gray')
    axes[1, 0].set_title('Opening (denoise)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(closed, cmap='gray')
    axes[1, 1].set_title('Closing (fill gaps)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(square_se, cmap='gray')
    axes[1, 2].set_title('Square SE')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(circle_se, cmap='gray')
    axes[1, 3].set_title('Circle SE')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('morphological_ops_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'morphological_ops_demo.png'")


def demo_frequency_domain():
    """Demonstrate frequency domain processing"""
    print("\n" + "="*60)
    print("FREQUENCY DOMAIN PROCESSING DEMONSTRATION")
    print("="*60)
    
    shapes, noisy, _ = create_synthetic_test_images()
    
    # Compute FFT
    print("\n1. Fast Fourier Transform")
    spectrum = FrequencyDomain.fft_2d(shapes)
    magnitude_spectrum = np.abs(spectrum)
    phase_spectrum = np.angle(spectrum)
    
    print(f"   Spectrum shape: {spectrum.shape}")
    print(f"   DC component: {magnitude_spectrum[0, 0]:.2f}")
    print(f"   Max frequency magnitude: {magnitude_spectrum.max():.2f}")
    
    # Frequency domain filtering
    print("\n2. Frequency Domain Filtering")
    
    def ideal_lowpass(spec, cutoff):
        return FrequencyDomain.ideal_lowpass_filter(spec, cutoff)
    
    def gaussian_lowpass(spec, d0):
        return FrequencyDomain.gaussian_lowpass_filter(spec, d0)
    
    # Apply filters
    ideal_filtered = FrequencyDomain.frequency_domain_filter(noisy, ideal_lowpass, 30)
    gaussian_filtered = FrequencyDomain.frequency_domain_filter(noisy, gaussian_lowpass, 30)
    
    print(f"   Ideal lowpass filter applied (cutoff=30)")
    print(f"   Gaussian lowpass filter applied (D0=30)")
    
    # Compare DFT vs FFT performance
    print("\n3. Algorithm Comparison")
    small_signal = shapes[:32, :32]
    
    import time
    
    # DFT timing
    start = time.time()
    dft_result = FrequencyDomain.dft_2d(small_signal)
    dft_time = time.time() - start
    
    # FFT timing
    start = time.time()
    fft_result = FrequencyDomain.fft_2d(small_signal)
    fft_time = time.time() - start
    
    print(f"   DFT time (32x32): {dft_time:.4f}s")
    print(f"   FFT time (32x32): {fft_time:.4f}s")
    print(f"   Speedup: {dft_time/fft_time:.2f}x faster")
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original (Noisy)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    axes[0, 1].set_title('Magnitude Spectrum (log)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(ideal_filtered, cmap='gray')
    axes[0, 2].set_title('Ideal Lowpass Filtered')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(gaussian_filtered, cmap='gray')
    axes[0, 3].set_title('Gaussian Lowpass Filtered')
    axes[0, 3].axis('off')
    
    # Show phase and filter masks
    axes[1, 0].imshow(phase_spectrum, cmap='hsv')
    axes[1, 0].set_title('Phase Spectrum')
    axes[1, 0].axis('off')
    
    # Create filter visualization
    H_ideal = np.zeros((100, 100))
    H_gaussian = np.zeros((100, 100))
    center = 50
    for i in range(100):
        for j in range(100):
            d = np.sqrt((i-center)**2 + (j-center)**2)
            H_ideal[i,j] = 1 if d <= 20 else 0
            H_gaussian[i,j] = np.exp(-(d**2)/(2*20**2))
    
    axes[1, 1].imshow(H_ideal, cmap='gray')
    axes[1, 1].set_title('Ideal Lowpass Filter')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(H_gaussian, cmap='gray')
    axes[1, 2].set_title('Gaussian Lowpass Filter')
    axes[1, 2].axis('off')
    
    # Comparison
    axes[1, 3].plot([dft_time, fft_time], ['DFT', 'FFT'], 'o-', linewidth=2, markersize=10)
    axes[1, 3].set_title('Speed Comparison')
    axes[1, 3].set_xlabel('Time (seconds)')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frequency_domain_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'frequency_domain_demo.png'")


def generate_comprehensive_report():
    """Generate a comprehensive analysis report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Test on different image sizes
    sizes = [32, 64, 128]
    
    print("\nAlgorithm Complexity Analysis:")
    print("-" * 60)
    
    for size in sizes:
        test_img = np.random.rand(size, size) * 255
        
        import time
        
        # Convolution
        kernel = SpatialFilters.create_gaussian_kernel(5, 1.0)
        start = time.time()
        _ = SpatialFilters.convolve2d(test_img, kernel)
        conv_time = time.time() - start
        
        # Edge detection
        start = time.time()
        _ = EdgeDetection.sobel_operator(test_img)
        edge_time = time.time() - start
        
        # FFT
        start = time.time()
        _ = FrequencyDomain.fft_2d(test_img)
        fft_time = time.time() - start
        
        print(f"\nImage size: {size}x{size}")
        print(f"  Convolution: {conv_time*1000:.2f}ms")
        print(f"  Edge Detection: {edge_time*1000:.2f}ms")
        print(f"  FFT: {fft_time*1000:.2f}ms")
    
    print("\n" + "="*60)
    print("Mathematical Implementations Summary:")
    print("-" * 60)
    print("✓ Gaussian kernel: 2D Gaussian formula")
    print("✓ Convolution: Discrete convolution operation")
    print("✓ Sobel operator: Gradient computation via kernels")
    print("✓ Canny: Multi-stage edge detection pipeline")
    print("✓ Morphology: Set-theoretic operations on shapes")
    print("✓ DFT: Direct implementation of Fourier transform")
    print("✓ FFT: Cooley-Tukey divide-and-conquer algorithm")
    print("=" * 60)


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("DIGITAL IMAGE PROCESSING SUITE - COMPLETE DEMONSTRATION")
    print("="*60)
    print("\nThis script demonstrates:")
    print("• Custom implementation of mathematical algorithms")
    print("• No reliance on OpenCV or high-level libraries")
    print("• Direct translation of equations to code")
    print("• Computational efficiency analysis")
    
    try:
        demo_spatial_filters()
        demo_edge_detection()
        demo_morphological_operations()
        demo_frequency_domain()
        generate_comprehensive_report()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  • spatial_filters_demo.png")
        print("  • edge_detection_demo.png")
        print("  • morphological_ops_demo.png")
        print("  • frequency_domain_demo.png")
        print("\nTotal lines of custom implementation: ~800-1000 LOC")
        print("Algorithms: 15+ from scratch")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()