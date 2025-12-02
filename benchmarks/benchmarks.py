from gaussian_blur import *
from blend import *
from resize import *
from rotate import *
from flip import *
from crop import *
import os
from functools import partial

class BenchmarkRunner:
    def __init__(self, filename="BENCHMARK_REPORT.md"):
        self.filename = filename
        if os.path.exists(self.filename):
            os.remove(self.filename)
        self._write_md("# PyImageCUDA Performance Report")
        self._write_md("Generated automatically by /benchmarks/benchmarks.py\n")

    def _write_md(self, text=""):
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _save_table(self, title: str, results: list[dict]):
        if not results: return
        
        results.sort(key=lambda x: x['fps'], reverse=True)
        baseline = results[-1]
        
        self._write_md(f"### {title}")
        self._write_md(f"| Library | Avg (ms) | FPS | Speedup |")
        self._write_md(f"| :--- | :--- | :--- | :--- |")
        
        for r in results:
            speedup = r['fps'] / baseline['fps']
            lib_name = f"**{r['lib']}**"
            self._write_md(f"| {lib_name} | {r['avg_ms']:.2f} | {r['fps']:.1f} | {speedup:.1f}x |")
        self._write_md("")

    def run_suite(self, title: str, config: dict, pure_tests: list, e2e_tests: list):
        print(f"Running {title}...")
        
        self._write_md(f"## {title}")
        config_str = ", ".join([f"{k}: `{v}`" for k, v in config.items()])
        self._write_md(f"> **Config:** {config_str}\n")
        
        if pure_tests:
            pure_results = [func() for func in pure_tests]
            self._save_table("Pure Algorithm (Compute Bound)", pure_results)

        if e2e_tests:
            e2e_results = [func() for func in e2e_tests]
            self._save_table("End-to-End (Disk I/O + Encode)", e2e_results)
        
        self._write_md("---\n")

def run_all_benchmarks():
    runner = BenchmarkRunner()
    
    IMG = "photo.jpg"
    LOOPS = 50
    RADIUS = 20

    runner.run_suite(
        title="Gaussian Blur Benchmark (1080p)",
        config={"Image": IMG, "Radius": RADIUS, "Iterations": LOOPS},
        pure_tests=[
            partial(bench_pillow_blur, IMG, RADIUS, LOOPS),
            partial(bench_opencv_blur, IMG, RADIUS, LOOPS),
            partial(bench_pyimagecuda_reuse, IMG, RADIUS, LOOPS),
            partial(bench_pyimagecuda_no_reuse, IMG, RADIUS, LOOPS)
        ],
        e2e_tests=[
            partial(bench_pillow_blur_e2e, IMG, RADIUS, LOOPS),
            partial(bench_opencv_blur_e2e, IMG, RADIUS, LOOPS),
            partial(bench_pyimagecuda_blur_e2e, IMG, RADIUS, LOOPS),
            partial(bench_pyimagecuda_blur_e2e_buffered, IMG, RADIUS, LOOPS)
        ]
    )

    runner.run_suite(
        title="Blend Normal Benchmark (1080p)",
        config={"Image": IMG, "Iterations": LOOPS},
        pure_tests=[
            partial(bench_pillow_blend, IMG, LOOPS),
            partial(bench_opencv_blend, IMG, LOOPS),
            partial(bench_pyimagecuda_blend, IMG, LOOPS)
        ],
        e2e_tests=[
            partial(bench_pillow_blend_e2e, IMG, LOOPS),
            partial(bench_opencv_blend_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_blend_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_blend_e2e_buffered, IMG, LOOPS)
        ]
    )

    runner.run_suite(
        title="Resize Bilinear Benchmark (1080p -> 800x600)",
        config={"Image": IMG, "Target": "800x600", "Interpolation": "Bilinear", "Iterations": LOOPS},
        pure_tests=[
            partial(bench_pillow_resize_bilinear, IMG, LOOPS),
            partial(bench_opencv_resize_bilinear, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_bilinear_reuse, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_bilinear_no_reuse, IMG, LOOPS)
        ],
        e2e_tests=[
            partial(bench_pillow_resize_bilinear_e2e, IMG, LOOPS),
            partial(bench_opencv_resize_bilinear_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_bilinear_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_bilinear_e2e_buffered, IMG, LOOPS)
        ]
    )

    runner.run_suite(
        title="Resize Lanczos Benchmark (1080p -> 800x600)",
        config={"Image": IMG, "Target": "800x600", "Interpolation": "Lanczos/Bicubic", "Iterations": LOOPS},
        pure_tests=[
            partial(bench_pillow_resize_lanczos, IMG, LOOPS),
            partial(bench_opencv_resize_lanczos, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_lanczos_reuse, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_lanczos_no_reuse, IMG, LOOPS)
        ],
        e2e_tests=[
            partial(bench_pillow_resize_lanczos_e2e, IMG, LOOPS),
            partial(bench_opencv_resize_lanczos_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_lanczos_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_resize_lanczos_e2e_buffered, IMG, LOOPS)
        ]
    )

    runner.run_suite(
        title="Rotate 35° Benchmark (1080p)",
        config={"Image": IMG, "Angle": "35°", "Expand": True, "Iterations": LOOPS},
        pure_tests=[
            partial(bench_pillow_rotate, IMG, LOOPS),
            partial(bench_opencv_rotate, IMG, LOOPS),
            partial(bench_pyimagecuda_rotate_reuse, IMG, LOOPS),
            partial(bench_pyimagecuda_rotate_no_reuse, IMG, LOOPS)
        ],
        e2e_tests=[
            partial(bench_pillow_rotate_e2e, IMG, LOOPS),
            partial(bench_opencv_rotate_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_rotate_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_rotate_e2e_buffered, IMG, LOOPS)
        ]
    )

    runner.run_suite(
        title="Flip Horizontal Benchmark (1080p)",
        config={"Image": IMG, "Direction": "Horizontal", "Iterations": LOOPS},
        pure_tests=[
            partial(bench_pillow_flip, IMG, LOOPS),
            partial(bench_opencv_flip, IMG, LOOPS),
            partial(bench_pyimagecuda_flip_reuse, IMG, LOOPS),
            partial(bench_pyimagecuda_flip_no_reuse, IMG, LOOPS)
        ],
        e2e_tests=[
            partial(bench_pillow_flip_e2e, IMG, LOOPS),
            partial(bench_opencv_flip_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_flip_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_flip_e2e_buffered, IMG, LOOPS)
        ]
    )

    runner.run_suite(
        title="Crop Center Benchmark (1080p → 512×512)",
        config={"Image": IMG, "Source": "1920×1080", "Output": "512×512", "Iterations": LOOPS},
        pure_tests=[
            partial(bench_pillow_crop, IMG, LOOPS),
            partial(bench_opencv_crop, IMG, LOOPS),
            partial(bench_pyimagecuda_crop_reuse, IMG, LOOPS),
            partial(bench_pyimagecuda_crop_no_reuse, IMG, LOOPS)
        ],
        e2e_tests=[
            partial(bench_pillow_crop_e2e, IMG, LOOPS),
            partial(bench_opencv_crop_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_crop_e2e, IMG, LOOPS),
            partial(bench_pyimagecuda_crop_e2e_buffered, IMG, LOOPS)
        ]
    )

    print(f"\n✅ Done! Report saved to: {os.path.abspath(runner.filename)}")

if __name__ == "__main__":
    run_all_benchmarks()

    for file in os.listdir():
        if file.startswith("temp_"):
            try:
                os.remove(file)
            except OSError:
                pass