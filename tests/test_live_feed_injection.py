"""
Tests for the dashboard live-feed injection.

The dashboard frontend is a compiled Nuxt bundle with no source in the repo and
it routes in hash mode, so the Live page cannot be given a market view by
editing a component or by serving a different route. The only seam that survives
both facts is one script tag in index.html, which is what these tests pin.

They assert the wiring rather than the rendering: that the tag exists and points
at a file that is actually shipped, that the file is served revalidating rather
than cached for a year, and that the script carries the guardrails the rest of
this project takes for granted.
"""

from pathlib import Path

import pytest

SCRIPT_NAME = 'algorithex-live-feed.js'
STATIC = Path(__file__).resolve().parents[1] / 'algorithex' / 'static'


@pytest.fixture(scope='module')
def index_html():
    return (STATIC / 'index.html').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def feed_js():
    return (STATIC / SCRIPT_NAME).read_text(encoding='utf-8')


# --- the injection seam --------------------------------------------------


def test_index_html_loads_the_live_feed(index_html):
    assert SCRIPT_NAME in index_html, (
        'the dashboard will not show live market data without this tag'
    )


def test_the_tag_is_a_plain_deferred_script_not_a_module(index_html):
    """A module script would be subject to the bundle's import map and CSP."""
    tag = next(t for t in index_html.split('<script') if SCRIPT_NAME in t)
    assert 'src="/' + SCRIPT_NAME + '"' in tag
    assert 'type="module"' not in tag
    assert 'defer' in tag


def test_the_tag_sits_in_the_head(index_html):
    """In the head with defer, so it parses before the bundle mounts and the
    observer catches the mount that follows."""
    assert index_html.index(SCRIPT_NAME) < index_html.index('</head>')


def test_the_script_file_is_actually_shipped(feed_js):
    assert len(feed_js) > 2000, 'the injected file looks empty or truncated'


def test_it_is_injected_exactly_once(index_html):
    assert index_html.count(SCRIPT_NAME) == 1, (
        'a duplicate tag would mount two panels per page'
    )


# --- served correctly ----------------------------------------------------


def test_the_script_is_served_from_the_static_mount():
    """It must sit beside index.html, or the mount at / will not find it."""
    assert (STATIC / SCRIPT_NAME).is_file()
    assert (STATIC / 'index.html').is_file()


def test_the_static_mount_serves_it_revalidating():
    """A year-long immutable cache on an unhashed file would freeze the panel
    at whatever version the browser first fetched."""
    from algorithex import CachedStaticFiles

    # Underscore-prefixed paths are the immutable ones; this file is not.
    assert CachedStaticFiles.IMMUTABLE_PREFIX == '_nuxt/'
    assert not SCRIPT_NAME.startswith('_nuxt/')


# --- guardrails the rest of the project assumes ----------------------------


def test_the_feed_is_declared_read_only(feed_js):
    assert 'READ ONLY' in feed_js


def test_the_feed_never_mentions_placing_an_order(feed_js):
    """No mutating verb may appear as a call. This is the boundary the whole
    module series is built on and the injection must not quietly break it."""
    for banned in ("place_order", "cancel_order", "createOrder", "/order'"):
        assert banned not in feed_js, f'the feed references {banned}'


def test_the_feed_only_calls_the_read_only_endpoint(feed_js):
    assert "'/live/market'" in feed_js
    assert 'live/strategy-ch' not in feed_js


def test_it_reads_the_session_the_dashboard_already_writes(feed_js):
    assert "localStorage.getItem('main')" in feed_js
    assert 'authToken' in feed_js


def test_it_uses_a_shadow_root(feed_js):
    """Isolation both ways: the compiled stylesheet only contains the utility
    classes the original build used, and our markup must not inherit its cascade."""
    assert "attachShadow({ mode: 'open' })" in feed_js


def test_errors_cannot_escape_into_the_dashboard(feed_js):
    """A throw in here must not reach the app's own error boundary."""
    assert 'try { mount(); } catch' in feed_js
    assert 'catch (e) {\n      // A rendering fault must not become an unhandled rejection.' in feed_js \
        or 'A rendering fault must not become an unhandled rejection' in feed_js


def test_the_loading_flag_is_per_host_not_module_state(feed_js):
    """A module-level flag outlives the panel it guards. Nuxt replaces the Live
    page subtree on navigation, so a stuck global silently starves every panel
    that follows -- which is exactly the failure this was rewritten to avoid."""
    assert 'host.__loading' in feed_js
    assert 'inFlight' not in feed_js, (
        'a module-level in-flight flag will wedge when Nuxt swaps the panel out'
    )


def test_responses_are_written_to_the_currently_mounted_panel(feed_js):
    """Rendering into the panel the request started from loses the data when
    Nuxt has replaced it in the meantime."""
    assert "document.getElementById(HOST_ID)" in feed_js
    assert 'write(' in feed_js


def test_it_mounts_only_on_the_live_route(feed_js):
    assert "ROUTE_PREFIX = '#/live'" in feed_js
    assert 'function unmount()' in feed_js


def test_the_script_passes_a_syntax_check():
    """A parse error would leave the dashboard working and the panel silently
    absent -- the same symptom as a missing script tag."""
    import shutil
    import subprocess

    node = shutil.which('node')
    if not node:
        pytest.skip('node is not available to syntax-check the injected script')
    result = subprocess.run(
        [node, '--check', str(STATIC / SCRIPT_NAME)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
