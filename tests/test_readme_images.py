"""
The README must not link to images that are not in the repository.

GitHub renders a missing image as a broken-image icon and says nothing louder,
so a screenshot deleted from `assets/` while its `![](...)` line stays behind
produces a README that looks fine in a diff and broken on the page. That is
exactly how the inherited Jesse screenshots ended up here in the first place:
they were referenced but not ours, and nothing checked that the two agreed.

These tests also pin the screenshots to this project, because "the file exists"
and "the file shows this product" are different claims and only the first is
mechanically checkable.
"""

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / 'README.md'
ASSETS = REPO / 'assets'

# Markdown ![alt](path) and HTML <img src="path"> / href="path".
# The trailing character class must exclude ")" as well as quotes, whitespace
# and ">": markdown paths end at the closing paren, and swallowing it makes
# every single reference look like a missing file.
IMAGE_REF = re.compile(
    r'(?:!\[[^\]]*\]\(|<img[^>]+src="|<a[^>]+href=")((?:assets|docs)/[^"\s>)]+)'
)


def readme() -> str:
    return README.read_text(encoding='utf-8')


def test_every_image_the_readme_names_exists():
    missing = sorted(
        ref for ref in set(IMAGE_REF.findall(readme())) if not (REPO / ref).exists()
    )
    assert not missing, f'README references images that are not in the repo: {missing}'


def test_the_header_has_a_logo():
    """The logo is the first thing anyone sees. Referencing the SVG keeps it
    crisp at any size and lets it carry the brand gradient."""
    head = readme().split('\n', 40)
    assert any('assets/algorithex-logo' in line for line in head), (
        'the README header must reference the Algorithex logo'
    )


def test_the_screenshots_are_ours_not_the_upstream_projects():
    """Guards the rebranding. Jesse's screenshots were shipped under the
    Algorithex name, which shows a reader someone else's product."""
    screenshots = ASSETS / 'screenshots'
    if not screenshots.is_dir():
        pytest.skip('no screenshots directory')

    legacy = [
        p.name for p in sorted(screenshots.iterdir())
        if p.suffix.lower() in {'.jpg', '.jpeg'}
    ]
    assert not legacy, (
        'the inherited Jesse screenshots (.jpg) are back in assets/screenshots; '
        f'{legacy} show the upstream product, not this one'
    )


def test_no_screenshot_is_left_unreferenced():
    """Dead weight in a repository, and a trap: the next person edits the
    README, not the orphan, and ships the old image by accident."""
    screenshots = ASSETS / 'screenshots'
    if not screenshots.is_dir():
        pytest.skip('no screenshots directory')

    on_disk = {p.name for p in screenshots.iterdir() if p.is_file()}
    referenced = {ref.split('/')[-1] for ref in IMAGE_REF.findall(readme())}
    orphans = sorted(on_disk - referenced)
    assert not orphans, (
        f'files in assets/screenshots that the README never shows: {orphans}'
    )


def test_the_logo_files_are_shipped():
    for name in ('algorithex-logo.svg', 'algorithex-mark.svg', 'algorithex-logo.png'):
        path = ASSETS / name
        assert path.is_file(), f'{name} is missing from assets/'
        assert path.stat().st_size > 500, f'{name} looks empty or truncated'