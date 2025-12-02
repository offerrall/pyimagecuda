import pytest
import gc
import threading
import tempfile
import os
from pyimagecuda import (
    Image, ImageU8, Fill, Filter, Adjust, Transform, Resize, Blend, Effect,
    load, save, upload, download, copy, convert_float_to_u8, convert_u8_to_float
)
from pyimagecuda.utils import ensure_capacity


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_image():
    img = Image(64, 64)
    yield img
    img.free()

@pytest.fixture
def medium_image():
    img = Image(512, 512)
    yield img
    img.free()

@pytest.fixture
def large_image():
    img = Image(1920, 1080)
    yield img
    img.free()

@pytest.fixture
def reusable_buffer():
    buf = Image(2048, 2048)
    yield buf
    buf.free()

@pytest.fixture
def u8_buffer():
    buf = ImageU8(1024, 1024)
    yield buf
    buf.free()

@pytest.fixture
def image_pair():
    img1 = Image(512, 512)
    img2 = Image(256, 256)
    yield img1, img2
    img1.free()
    img2.free()


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class TestMemoryManagement:
    
    def test_basic_allocation_deallocation(self):
        img = Image(512, 512)
        assert img.width == 512
        assert img.height == 512
        img.free()
    
    def test_context_manager(self):
        with Image(512, 512) as img:
            assert img.width == 512
    
    def test_nested_context_managers(self):
        with Image(1024, 1024) as img1:
            with Image(512, 512) as img2:
                assert img1.width == 1024
                assert img2.width == 512
    
    def test_multiple_allocations(self):
        images = []
        for _ in range(100):
            img = Image(128, 128)
            images.append(img)
        
        for img in images:
            img.free()
    
    def test_large_allocation(self):
        try:
            img = Image(4096, 4096)
            assert img.width == 4096
            img.free()
        except Exception as e:
            assert "memory" in str(e).lower() or "allocation" in str(e).lower()
    
    def test_zero_size_allocation_fails(self):
        with pytest.raises(Exception):
            Image(0, 512)
        
        with pytest.raises(Exception):
            Image(512, 0)
    
    def test_negative_size_allocation_fails(self):
        with pytest.raises(Exception):
            Image(-512, 512)
        
        with pytest.raises(Exception):
            Image(512, -512)
    
    def test_u8_allocation(self):
        img = ImageU8(512, 512)
        assert img.width == 512
        assert img.height == 512
        img.free()
    
    def test_both_image_types_coexist(self):
        f32 = Image(512, 512)
        u8 = ImageU8(512, 512)
        
        assert f32.width == 512
        assert u8.width == 512
        
        f32.free()
        u8.free()


# ============================================================================
# BUFFER CAPACITY
# ============================================================================

class TestBufferCapacity:
    
    def test_get_max_capacity(self):
        img = Image(1024, 768)
        max_w, max_h = img.get_max_capacity()
        
        assert max_w == 1024
        assert max_h == 768
        img.free()
    
    def test_resize_within_capacity(self):
        img = Image(1024, 1024)
        
        img.width = 512
        img.height = 512
        
        assert img.width == 512
        assert img.height == 512
        
        max_w, max_h = img.get_max_capacity()
        assert max_w == 1024
        assert max_h == 1024
        
        img.free()
    
    def test_resize_exceeds_capacity_fails(self):
        img = Image(512, 512)
        
        with pytest.raises(ValueError, match="exceeds buffer capacity"):
            img.width = 1024
        
        with pytest.raises(ValueError, match="exceeds buffer capacity"):
            img.height = 1024
        
        img.free()
    
    def test_resize_to_zero_fails(self):
        img = Image(512, 512)
        
        with pytest.raises(ValueError, match="must be positive"):
            img.width = 0
        
        with pytest.raises(ValueError, match="must be positive"):
            img.height = 0
        
        img.free()
    
    def test_resize_to_negative_fails(self):
        img = Image(512, 512)
        
        with pytest.raises(ValueError, match="must be positive"):
            img.width = -100
        
        with pytest.raises(ValueError, match="must be positive"):
            img.height = -100
        
        img.free()
    
    def test_multiple_resizes(self):
        img = Image(1024, 1024)
        
        sizes = [(512, 512), (256, 768), (1024, 512), (100, 100), (1024, 1024)]
        
        for w, h in sizes:
            img.width = w
            img.height = h
            assert img.width == w
            assert img.height == h
        
        img.free()


# ============================================================================
# BUFFER REUSE
# ============================================================================

class TestBufferReuse:
    
    def test_buffer_reuse_same_size(self, reusable_buffer):
        reusable_buffer.width = 512
        reusable_buffer.height = 512
        Fill.color(reusable_buffer, (1.0, 0.0, 0.0, 1.0))
        
        Fill.color(reusable_buffer, (0.0, 1.0, 0.0, 1.0))
        
        assert reusable_buffer.width == 512
        assert reusable_buffer.height == 512
    
    def test_buffer_reuse_different_sizes(self, reusable_buffer):
        sizes = [(512, 512), (256, 768), (1024, 512), (100, 100)]
        
        for w, h in sizes:
            reusable_buffer.width = w
            reusable_buffer.height = h
            Fill.color(reusable_buffer, (1.0, 0.0, 0.0, 1.0))
            
            assert reusable_buffer.width == w
            assert reusable_buffer.height == h
    
    def test_buffer_reuse_with_operations(self, reusable_buffer):
        for i in range(10):
            size = 256 + i * 64
            reusable_buffer.width = size
            reusable_buffer.height = size
            
            Fill.color(reusable_buffer, (1.0, 0.0, 0.0, 1.0))
            Filter.invert(reusable_buffer)
            
            assert reusable_buffer.width == size
    
    def test_buffer_reuse_stress(self, reusable_buffer):
        for _ in range(1000):
            reusable_buffer.width = 512
            reusable_buffer.height = 512
            Fill.color(reusable_buffer, (1.0, 0.0, 0.0, 1.0))


# ============================================================================
# MEMORY LEAKS
# ============================================================================

class TestMemoryLeaks:
    
    def test_no_leak_on_repeated_allocations(self):
        for _ in range(1000):
            img = Image(128, 128)
            img.free()
    
    def test_no_leak_with_context_manager(self):
        for _ in range(1000):
            with Image(128, 128) as img:
                pass
    
    def test_no_leak_on_failed_operations(self):
        img = Image(512, 512)
        
        for _ in range(100):
            try:
                img.width = 9999
            except ValueError:
                pass
        
        img.free()
    
    def test_cleanup_after_exception(self):
        try:
            with Image(512, 512) as img:
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass


# ============================================================================
# OPERATIONS MEMORY
# ============================================================================

class TestOperationsMemory:
    
    def test_inplace_operations_preserve_buffer(self, medium_image):
        original_handle = medium_image._buffer._handle
        
        Fill.color(medium_image, (1.0, 0.0, 0.0, 1.0))
        assert medium_image._buffer._handle == original_handle
        
        Filter.invert(medium_image)
        assert medium_image._buffer._handle == original_handle
        
        Adjust.brightness(medium_image, 0.5)
        assert medium_image._buffer._handle == original_handle
    
    def test_operations_with_new_image(self, medium_image):
        blurred = Filter.gaussian_blur(medium_image, radius=5)
        
        assert blurred is not None
        assert blurred is not medium_image
        assert blurred.width == medium_image.width
        assert blurred.height == medium_image.height
        
        blurred.free()
    
    def test_operations_with_dst_buffer(self, medium_image, reusable_buffer):
        reusable_buffer.width = medium_image.width
        reusable_buffer.height = medium_image.height
        
        result = Filter.gaussian_blur(medium_image, radius=5, dst_buffer=reusable_buffer)
        
        assert result is None
        assert reusable_buffer.width == medium_image.width
    
    def test_buffer_reuse_across_operations(self, medium_image, reusable_buffer):
        Fill.color(medium_image, (1.0, 0.0, 0.0, 1.0))
        
        Filter.gaussian_blur(medium_image, radius=5, dst_buffer=reusable_buffer)
        Filter.sharpen(medium_image, strength=1.0, dst_buffer=reusable_buffer)
        Transform.flip(medium_image, direction='horizontal', dst_buffer=reusable_buffer)
        
        assert reusable_buffer.width == medium_image.width


# ============================================================================
# BUFFER VALIDATION
# ============================================================================

class TestBufferValidation:
    
    def test_insufficient_buffer_capacity_fails(self):
        small_buffer = Image(256, 256)
        
        with pytest.raises(ValueError, match="capacity too small"):
            ensure_capacity(small_buffer, 512, 512)
        
        small_buffer.free()
    
    def test_ensure_capacity_updates_dimensions(self):
        buffer = Image(1024, 1024)
        ensure_capacity(buffer, 512, 512)
        
        assert buffer.width == 512
        assert buffer.height == 512
        
        buffer.free()
    
    def test_operations_validate_buffer_capacity(self, small_image):
        tiny_buffer = Image(32, 32)
        
        with pytest.raises(ValueError):
            Filter.gaussian_blur(small_image, radius=5, dst_buffer=tiny_buffer)
        
        tiny_buffer.free()


# ============================================================================
# IO MEMORY
# ============================================================================

class TestIOMemory:
    
    def test_load_allocates_correctly(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            img = Image(256, 256)
            Fill.color(img, (1.0, 0.0, 0.0, 1.0))
            save(img, tmp_path)
            img.free()
            
            loaded = load(tmp_path)
            assert loaded.width == 256
            assert loaded.height == 256
            loaded.free()
        finally:
            os.unlink(tmp_path)
    
    def test_load_with_buffer_reuse(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            img = Image(256, 256)
            Fill.color(img, (1.0, 0.0, 0.0, 1.0))
            save(img, tmp_path)
            img.free()
            
            buffer = Image(1024, 1024)
            result = load(tmp_path, f32_buffer=buffer)
            
            assert result is None
            assert buffer.width == 256
            assert buffer.height == 256
            
            buffer.free()
        finally:
            os.unlink(tmp_path)
    
    def test_upload_download_roundtrip(self, medium_image):
        data = download(medium_image)
        
        img2 = Image(medium_image.width, medium_image.height)
        upload(img2, data)
        
        assert img2.width == medium_image.width
        assert img2.height == medium_image.height
        
        img2.free()
    
    def test_upload_wrong_size_fails(self, medium_image):
        wrong_data = bytes(100)
        
        with pytest.raises(ValueError, match="Expected .* bytes"):
            upload(medium_image, wrong_data)
    
    def test_copy_buffers(self):
        src = Image(512, 512)
        dst = Image(512, 512)
        
        Fill.color(src, (1.0, 0.0, 0.0, 1.0))
        
        copy(dst, src)
        
        assert dst.width == src.width
        assert dst.height == src.height
        
        src.free()
        dst.free()
    
    def test_copy_requires_sufficient_capacity(self):
        src = Image(512, 512)
        dst = Image(256, 256)
        
        with pytest.raises(ValueError, match="capacity too small"):
            copy(dst, src)
        
        src.free()
        dst.free()


# ============================================================================
# CONVERSION MEMORY
# ============================================================================

class TestConversionMemory:
    
    def test_conversion_roundtrip(self):
        f32 = Image(512, 512)
        u8 = ImageU8(512, 512)
        f32_2 = Image(512, 512)
        
        convert_float_to_u8(u8, f32)
        convert_u8_to_float(f32_2, u8)
        
        assert f32_2.width == 512
        assert f32_2.height == 512
        
        f32.free()
        u8.free()
        f32_2.free()
    
    def test_conversion_updates_dimensions(self):
        f32 = Image(256, 256)
        u8 = ImageU8(1024, 1024)
        
        convert_float_to_u8(u8, f32)
        
        assert u8.width == 256
        assert u8.height == 256
        
        f32.free()
        u8.free()


# ============================================================================
# ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    
    def test_gamma_zero_fails(self, medium_image):
        with pytest.raises(ValueError, match="must be positive"):
            Adjust.gamma(medium_image, 0.0)
    
    def test_gamma_negative_fails(self, medium_image):
        with pytest.raises(ValueError, match="must be positive"):
            Adjust.gamma(medium_image, -1.0)
    
    def test_invalid_blend_anchor(self, image_pair):
        base, overlay = image_pair
        Fill.color(base, (1.0, 0.0, 0.0, 1.0))
        Fill.color(overlay, (0.0, 1.0, 0.0, 1.0))
        

        Blend.normal(base, overlay, anchor='invalid')

        assert base.width == 512
    
    def test_invalid_gradient_direction(self, medium_image):
        with pytest.raises(ValueError, match="Invalid direction"):
            Fill.gradient(
                medium_image,
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                direction='invalid'
            )
    
    def test_negative_rounded_corners_fails(self, medium_image):
        with pytest.raises(ValueError, match="non-negative"):
            Effect.rounded_corners(medium_image, -10.0)
    
    def test_zero_checkerboard_size_fails(self, medium_image):
        with pytest.raises(ValueError, match="must be positive"):
            Fill.checkerboard(medium_image, size=0)
    
    def test_negative_stroke_width_fails(self, medium_image):
        Fill.color(medium_image, (1.0, 0.0, 0.0, 1.0))
        
        with pytest.raises(ValueError, match="Invalid stroke width"):
            Effect.stroke(medium_image, width=-5)
    
    def test_invalid_ngon_sides(self, medium_image):
        with pytest.raises(ValueError, match="at least 3 sides"):
            Fill.ngon(medium_image, sides=2)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    
    def test_single_pixel_image(self):
        img = Image(1, 1)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        img.free()
    
    def test_very_wide_image(self):
        img = Image(4096, 1)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        img.free()
    
    def test_very_tall_image(self):
        img = Image(1, 4096)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        img.free()
    
    def test_extreme_brightness(self, medium_image):
        Fill.color(medium_image, (0.5, 0.5, 0.5, 1.0))
        Adjust.brightness(medium_image, 100.0)
        Adjust.brightness(medium_image, -100.0)
    
    def test_extreme_contrast(self, medium_image):
        Fill.color(medium_image, (0.5, 0.5, 0.5, 1.0))
        Adjust.contrast(medium_image, 100.0)
        Adjust.contrast(medium_image, 0.001)
    
    def test_zero_saturation(self, medium_image):
        Fill.color(medium_image, (1.0, 0.5, 0.0, 1.0))
        Adjust.saturation(medium_image, 0.0)
    
    def test_crop_outside_bounds(self):
        img = Image(512, 512)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        
        cropped = Transform.crop(img, x=-100, y=-100, width=200, height=200)
        
        assert cropped.width == 200
        assert cropped.height == 200
        
        img.free()
        cropped.free()
    
    def test_crop_partial_overlap(self):
        img = Image(512, 512)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        
        cropped = Transform.crop(img, x=400, y=400, width=200, height=200)
        
        assert cropped.width == 200
        assert cropped.height == 200
        
        img.free()
        cropped.free()


# ============================================================================
# THREAD SAFETY
# ============================================================================

class TestThreadSafety:
    
    def test_concurrent_allocations(self):
        images = []
        errors = []
        
        def allocate():
            try:
                img = Image(256, 256)
                Fill.color(img, (1.0, 0.0, 0.0, 1.0))
                images.append(img)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=allocate) for _ in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(images) == 10
        
        for img in images:
            img.free()
    
    def test_concurrent_operations_separate_images(self):
        images = [Image(256, 256) for _ in range(10)]
        errors = []
        
        def process(img):
            try:
                Fill.color(img, (1.0, 0.0, 0.0, 1.0))
                Filter.invert(img)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=process, args=(img,)) for img in images]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        for img in images:
            img.free()


# ============================================================================
# REGRESSION BUGS
# ============================================================================

class TestRegressionBugs:
    
    def test_buffer_reuse_doesnt_leak(self):
        buffer = Image(1024, 1024)
        
        for i in range(1000):
            buffer.width = 512 + (i % 100)
            buffer.height = 512 + (i % 100)
            Fill.color(buffer, (1.0, 0.0, 0.0, 1.0))
        
        buffer.free()
    
    def test_drop_shadow_expand_doesnt_crash(self):
        img = Image(256, 256)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        Effect.rounded_corners(img, 20)
        
        shadowed = Effect.drop_shadow(img, blur=30, expand=True)
        
        assert shadowed is not None
        assert shadowed.width > img.width
        assert shadowed.height > img.height
        
        img.free()
        shadowed.free()
    
    def test_rotate_arbitrary_angle_stable(self):
        img = Image(512, 512)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        
        angles = [15, 37.5, 123.456, 287.9]
        
        for angle in angles:
            rotated = Transform.rotate(img, angle, expand=True)
            assert rotated is not None
            rotated.free()
        
        img.free()
    
    def test_multiple_effects_chain(self):
        img = Image(512, 512)
        Fill.gradient(img, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')
        
        Effect.rounded_corners(img, 50)
        
        stroked = Effect.stroke(img, 10, (1, 1, 1, 1))
        shadowed = Effect.drop_shadow(stroked, blur=30)
        rotated = Transform.rotate(shadowed, 45)
        
        assert rotated is not None
        
        img.free()
        stroked.free()
        shadowed.free()
        rotated.free()
    
    def test_stroke_inside_no_expand(self):
        img = Image(512, 512)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        Effect.rounded_corners(img, 50)
        
        stroked = Effect.stroke(img, 20, (1, 1, 1, 1), position='inside')
        
        assert stroked.width == img.width
        assert stroked.height == img.height
        
        img.free()
        stroked.free()
    
    def test_rotate_0_degrees(self):
        img = Image(512, 512)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        
        rotated = Transform.rotate(img, 0)
        
        assert rotated.width == img.width
        assert rotated.height == img.height
        
        img.free()
        rotated.free()
    
    def test_rotate_fixed_angles(self):
        img = Image(512, 512)
        Fill.color(img, (1.0, 0.0, 0.0, 1.0))
        
        for angle in [90, 180, 270]:
            rotated = Transform.rotate(img, angle)
            assert rotated is not None
            rotated.free()
        
        img.free()


# ============================================================================
# BATCH PROCESSING SIMULATION
# ============================================================================

class TestBatchProcessing:
    
    def test_batch_blur_with_buffer_reuse(self):
        src_buffer = Image(1920, 1080)
        dst_buffer = Image(1920, 1080)
        temp_buffer = Image(1920, 1080)
        
        for i in range(100):
            Fill.color(src_buffer, (float(i % 255) / 255, 0.5, 0.5, 1.0))
            Filter.gaussian_blur(src_buffer, radius=5, dst_buffer=dst_buffer, temp_buffer=temp_buffer)
        
        src_buffer.free()
        dst_buffer.free()
        temp_buffer.free()
    
    def test_batch_resize_with_buffer_reuse(self):
        src_buffer = Image(1920, 1080)
        dst_buffer = Image(1920, 1080)
        
        sizes = [(800, 600), (1280, 720), (640, 480), (1024, 768)]
        
        for w, h in sizes * 10:
            Fill.color(src_buffer, (1.0, 0.0, 0.0, 1.0))
            Resize.bilinear(src_buffer, width=w, height=h, dst_buffer=dst_buffer)
        
        src_buffer.free()
        dst_buffer.free()
    
    def test_batch_transform_with_buffer_reuse(self):
        src_buffer = Image(512, 512)
        dst_buffer = Image(1024, 1024)
        
        for _ in range(100):
            Fill.color(src_buffer, (1.0, 0.0, 0.0, 1.0))
            Transform.flip(src_buffer, direction='horizontal', dst_buffer=dst_buffer)
        
        src_buffer.free()
        dst_buffer.free()
    
    def test_batch_load_save_with_buffers(self):
        f32_buffer = Image(1024, 1024)
        u8_buffer = ImageU8(1024, 1024)
        
        files = []
        
        for i in range(10):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                files.append(tmp_path)
            
            f32_buffer.width = 256
            f32_buffer.height = 256
            Fill.color(f32_buffer, (float(i) / 10, 0.5, 0.5, 1.0))
            save(f32_buffer, tmp_path, u8_buffer=u8_buffer)
        
        for tmp_path in files:
            load(tmp_path, f32_buffer=f32_buffer, u8_buffer=u8_buffer)
            os.unlink(tmp_path)
        
        f32_buffer.free()
        u8_buffer.free()


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStress:
    
    def test_stress_many_small_images(self):
        for _ in range(10000):
            img = Image(32, 32)
            Fill.color(img, (1.0, 0.0, 0.0, 1.0))
            img.free()
    
    def test_stress_buffer_dimension_changes(self):
        buffer = Image(2048, 2048)
        
        for i in range(10000):
            w = 64 + (i % 1984)
            h = 64 + (i % 1984)
            buffer.width = w
            buffer.height = h
            Fill.color(buffer, (1.0, 0.0, 0.0, 1.0))
        
        buffer.free()
    
    def test_stress_nested_context_managers(self):
        for _ in range(1000):
            with Image(128, 128) as img1:
                with Image(128, 128) as img2:
                    with Image(128, 128) as img3:
                        Fill.color(img1, (1.0, 0.0, 0.0, 1.0))
                        Fill.color(img2, (0.0, 1.0, 0.0, 1.0))
                        Fill.color(img3, (0.0, 0.0, 1.0, 1.0))


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

class TestParameterValidation:
    
    def test_fill_gradient_invalid_direction(self, medium_image):
        with pytest.raises(ValueError):
            Fill.gradient(medium_image, (1, 0, 0, 1), (0, 1, 0, 1), direction='invalid')
    
    def test_transform_flip_invalid_direction(self, medium_image):
        with pytest.raises(ValueError):
            Transform.flip(medium_image, direction='invalid')
    
    def test_effect_stroke_invalid_position(self, medium_image):
        Fill.color(medium_image, (1.0, 0.0, 0.0, 1.0))
        with pytest.raises(ValueError):
            Effect.stroke(medium_image, width=10, position='invalid')
    
    def test_resize_no_dimensions(self, medium_image):
        with pytest.raises(ValueError):
            Resize.bilinear(medium_image)
    
    def test_transform_crop_negative_dimensions(self, medium_image):
        with pytest.raises(ValueError):
            Transform.crop(medium_image, x=0, y=0, width=-100, height=100)
    
    def test_transform_crop_zero_dimensions(self, medium_image):
        with pytest.raises(ValueError):
            Transform.crop(medium_image, x=0, y=0, width=0, height=100)


# ============================================================================
# OPERATIONS CORRECTNESS
# ============================================================================

class TestOperationsCorrectness:
    
    def test_flip_horizontal_dimensions(self, medium_image):
        flipped = Transform.flip(medium_image, direction='horizontal')
        assert flipped.width == medium_image.width
        assert flipped.height == medium_image.height
        flipped.free()
    
    def test_flip_vertical_dimensions(self, medium_image):
        flipped = Transform.flip(medium_image, direction='vertical')
        assert flipped.width == medium_image.width
        assert flipped.height == medium_image.height
        flipped.free()
    
    def test_rotate_90_dimensions(self, medium_image):
        rotated = Transform.rotate(medium_image, 90)
        assert rotated.width == medium_image.height
        assert rotated.height == medium_image.width
        rotated.free()
    
    def test_rotate_180_dimensions(self, medium_image):
        rotated = Transform.rotate(medium_image, 180)
        assert rotated.width == medium_image.width
        assert rotated.height == medium_image.height
        rotated.free()
    
    def test_crop_dimensions(self, medium_image):
        cropped = Transform.crop(medium_image, x=100, y=100, width=200, height=200)
        assert cropped.width == 200
        assert cropped.height == 200
        cropped.free()
    
    def test_resize_aspect_ratio_width_only(self):
        img = Image(1920, 1080)
        resized = Resize.bilinear(img, width=800)
        assert resized.width == 800
        assert resized.height == 450
        img.free()
        resized.free()
    
    def test_resize_aspect_ratio_height_only(self):
        img = Image(1920, 1080)
        resized = Resize.bilinear(img, height=540)
        assert resized.width == 960
        assert resized.height == 540
        img.free()
        resized.free()
    
    def test_blur_preserves_dimensions(self, medium_image):
        blurred = Filter.gaussian_blur(medium_image, radius=10)
        assert blurred.width == medium_image.width
        assert blurred.height == medium_image.height
        blurred.free()
    
    def test_sharpen_preserves_dimensions(self, medium_image):
        sharpened = Filter.sharpen(medium_image, strength=1.0)
        assert sharpened.width == medium_image.width
        assert sharpened.height == medium_image.height
        sharpened.free()