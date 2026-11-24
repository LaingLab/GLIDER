"""Cut the white mouse-and-wing art out of the black-on-white logo master."""

from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).parent
src = Image.open(HERE / "../../packaging/icons/glider_source_original.png").convert("L")
src = src.crop((40, 40, 2008, 2008))  # drop the master's thin border line
alpha = ImageOps.invert(src).point(lambda v: 0 if v < 20 else v)
art = Image.new("RGBA", src.size, (255, 255, 255, 0))
art.putalpha(alpha)
art.crop(art.getbbox()).save(HERE / "public/glider.png")
