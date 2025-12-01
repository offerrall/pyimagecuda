# Text

The `Text` module provides GPU-accelerated text rendering with full typography control using Pango markup.

Text is rendered to a new image sized to fit the content. The resulting image can be composited onto backgrounds, styled with effects, use as masks, or saved directly.

---

## Basic Text Rendering

Render simple text with default styling.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create("Hello World", size=60)

save(text_img, 'text_basic.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_basic.png" alt="Basic text" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `text` (str): Text to render (supports newlines for multiple lines)
- `font` (str): Font family name (default: "Sans")
- `size` (float): Font size in points (default: 12.0)
- `color` (tuple[float, float, float, float]): Text color in RGBA (default: black)
- `bg_color` (tuple[float, float, float, float]): Background color in RGBA (default: transparent)
- `align` (Literal['left', 'centre', 'right']): Text alignment (default: 'left')
- `justify` (bool): Justify text (default: False)
- `spacing` (int): Line spacing in pixels (default: 0)
- `letter_spacing` (float): Letter spacing in pixels (default: 0.0)
- `dst_buffer` (Image | None): Optional output buffer (default: None)
- `u8_buffer` (ImageU8 | None): Optional temporary buffer (default: None)

**Returns:** New `Image` containing rendered text (or None if `dst_buffer` provided)

---

## Font Selection

Use any system font by name. Combine font family with weight/style for variations.

**Example - Bold Font:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create(
    "Bold Title",
    font="Arial Bold",
    size=48,
    color=(0.2, 0.2, 0.2, 1.0)
)

save(text_img, 'text_bold.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_bold.png" alt="Bold font" style="width: 100%;">
  </div>
</div>

**Available font modifiers:**
- Weight: `"Light"`, `"Regular"`, `"Bold"`, `"Black"`
- Style: `"Italic"`, `"Oblique"`
- Combine: `"Arial Bold Italic"`, `"Times New Roman Bold"`

**Note:** Available fonts depend on your system.

---

## Text and Background Colors

Control foreground and background colors independently.

**Example - Colored Text on Background:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create(
    "ALERT\nSYSTEM",
    font="Arial Bold",
    size=50,
    color=(1.0, 0.2, 0.2, 1.0),
    bg_color=(0.0, 0.0, 0.3, 1.0),
    align='centre'
)

save(text_img, 'text_colors.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_colors.png" alt="Colored text" style="width: 100%;">
  </div>
</div>

**Use for:** Badges, labels, alerts, UI elements with solid backgrounds

---

## Text Alignment

Align multi-line text left, center, or right.

**Example - Center Aligned:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create(
    "Left Line\nCenter Line\nRight Line",
    size=30,
    align='centre',
    bg_color=(0.9, 0.9, 0.9, 1.0)
)

save(text_img, 'text_align_centre.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_align_centre.png" alt="Center aligned" style="width: 100%;">
  </div>
</div>

**Example - Right Aligned:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create(
    "Line One\nLine Two\nLine Three",
    size=30,
    align='right',
    bg_color=(0.9, 0.9, 0.9, 1.0)
)

save(text_img, 'text_align_right.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_align_right.png" alt="Right aligned" style="width: 100%;">
  </div>
</div>

**Available alignments:** `'left'`, `'centre'`, `'right'`

---

## Letter Spacing (Tracking)

Adjust horizontal space between characters.

**Example - Wide Tracking:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create(
    "CINEMATIC",
    font="Arial Bold",
    size=40,
    letter_spacing=10.0,
    color=(1.0, 1.0, 1.0, 1.0),
    bg_color=(0.0, 0.0, 0.0, 1.0)
)

save(text_img, 'text_tracking.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_tracking.png" alt="Letter spacing" style="width: 100%;">
  </div>
</div>

**Use for:** Title cards, logos, stylized headings, luxury branding

**Note:** Negative values tighten spacing; positive values expand it.

---

## Line Spacing

Control vertical distance between lines of text.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create(
    "Line 1\nLine 2\nLine 3",
    size=30,
    spacing=50,
    bg_color=(0.8, 0.8, 0.8, 1.0)
)

save(text_img, 'text_spacing.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_spacing.png" alt="Line spacing" style="width: 100%;">
  </div>
</div>

**Use for:** Improving readability, posters, artistic layouts

---

## Rich Text Markup

Use Pango markup for rich formatting within a single text block.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, save

text_img = Text.create(
    'Normal <b>Bold</b> <i>Italic</i>\n'
    '<span foreground="orange">Orange</span> '
    'and <sub>subscript</sub>',
    size=40,
    align='centre'
)

save(text_img, 'text_markup.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_markup.png" alt="Rich text markup" style="width: 100%;">
  </div>
</div>

**Supported markup tags:**

- `<b>...</b>` - Bold
- `<i>...</i>` - Italic
- `<u>...</u>` - Underline
- `<s>...</s>` - Strikethrough
- `<sup>...</sup>` - Superscript
- `<sub>...</sub>` - Subscript
- `<span foreground="color">...</span>` - Text color (use color names or hex)
- `<span size="larger">...</span>` - Size variations
- `<tt>...</tt>` - Monospace

**Note:** When using markup with `<` or `>` characters, Pango automatically enables RGBA rendering for proper compositing.

---

## UI Button Example

Combine text rendering with effects for polished UI elements.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Text, Image, Fill, Blend, Effect, save

# Create text
text_img = Text.create(
    "START GAME",
    font="Arial Bold",
    size=30,
    color=(1.0, 1.0, 1.0, 1.0)
)

# Create button background
pad_w, pad_h = 60, 30
button = Image(text_img.width + pad_w, 
               text_img.height + pad_h)

# Style button
Fill.color(button, (0.2, 0.6, 1.0, 1.0))
Effect.rounded_corners(button, 15)

# Composite text onto button
Blend.normal(button, text_img, anchor='center')

# Add drop shadow
final = Effect.drop_shadow(
    button,
    blur=10,
    offset_y=5,
    color=(0.0, 0.0, 0.0, 0.5)
)

save(final, 'text_ui_button.png')

# Cleanup
text_img.free()
button.free()
final.free()
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/text_ui_button.png" alt="UI button" style="width: 100%;">
  </div>
</div>

---

## Buffer Reuse

Reuse buffers for batch text rendering to avoid repeated allocations.

**Example:**
```python
from pyimagecuda import Image, ImageU8, Text, save

# Pre-allocate buffers (ensure sufficient capacity)
dst = Image(1024, 512)
u8_temp = ImageU8(1024, 512)

labels = ["Label 1", "Label 2", "Label 3"]

for i, label in enumerate(labels):
    Text.create(
        label,
        size=48,
        dst_buffer=dst,
        u8_buffer=u8_temp
    )
    
    save(dst, f"label_{i}.png")

dst.free()
u8_temp.free()
```

**Note:** Buffers must have capacity for the largest text output. The `u8_buffer` is used internally for intermediate conversions.

---

## Technical Notes

**Rendering pipeline:**

1. Pango renders text to CPU memory (with anti-aliasing)
2. Creates alpha mask from text
3. Applies colors (foreground and background)
4. Uploads to GPU as float32 RGBA

**Font rendering:**

- Uses system-installed fonts
- Full sub-pixel anti-aliasing
- Supports Unicode (emoji, international characters)
- Pango markup for rich text

**Transparency:**

- Default background is fully transparent
- Alpha channel preserved for compositing
- Set `bg_color` alpha to 0.0 for transparent backgrounds

---