"""Best-effort, HTTPS-only downloader for the CSIC 2010 text files.

The reliable way to obtain the dataset is to place the three ``.txt`` files in
``data/raw/`` manually (see ``data/README.md``); upstream hosting for CSIC 2010
changes over time.

Security
--------
* Only ``https://`` URLs are accepted - any other scheme is rejected before a
  connection is opened, and every redirect is re-validated so an
  ``https -> http`` downgrade is refused too.
* Downloads stream to a temporary file in the destination directory and are
  atomically moved into place, so a failed or partial download never leaves a
  corrupt file behind.
* Network failures are surfaced as a clear, actionable error rather than a raw
  traceback, and the message never contains payload data.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from ..utils import logger

# Public HTTPS mirrors of the original CSIC 2010 files. The GSI research mirror
# is the primary source; the GitHub mirror is tried as a fallback. Manual
# placement (see data/README.md) remains an option if both are unreachable.
_GITLAB_BASE = (
    "https://gitlab.fing.edu.uy/gsi/web-application-attacks-datasets"
    "/-/raw/master/csic_2010"
)
_GITHUB_BASE = (
    "https://raw.githubusercontent.com/Monkey-D-Groot/Machine-Learning-on-CSIC-2010/master"
)
_CSIC_FILES = (
    "normalTrafficTraining.txt",
    "normalTrafficTest.txt",
    "anomalousTrafficTest.txt",
)
_DEFAULT_URLS: dict[str, str] = {name: f"{_GITLAB_BASE}/{name}" for name in _CSIC_FILES}
_FALLBACK_URLS: dict[str, str] = {name: f"{_GITHUB_BASE}/{name}" for name in _CSIC_FILES}


def _validate_https(url: str) -> None:
    """Reject any URL whose scheme is not ``https``."""
    scheme = urlsplit(url).scheme.lower()
    if scheme != "https":
        raise ValueError(
            f"Refusing to download over a non-HTTPS URL (scheme={scheme!r}); "
            "only https:// URLs are allowed."
        )


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate the scheme on every redirect so https cannot be downgraded."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urlsplit(newurl).scheme.lower()
        if scheme != "https":
            raise urllib.error.URLError(
                f"Refusing non-HTTPS redirect to scheme {scheme!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# urllib follows redirects, so validating only the initial URL is not enough;
# this opener rejects any non-https redirect target as well.
_HTTPS_ONLY_OPENER = urllib.request.build_opener(_HttpsOnlyRedirectHandler())


def _download_one(url: str, target: pathlib.Path, *, timeout: int) -> None:
    """Stream a single validated HTTPS URL to ``target`` atomically."""
    tmp_path: pathlib.Path | None = None
    try:
        # The initial scheme is validated as https by the caller and every
        # redirect is re-validated by _HttpsOnlyRedirectHandler, so the transfer
        # cannot be downgraded to http:// (or redirected into file://) midway.
        with _HTTPS_ONLY_OPENER.open(url, timeout=timeout) as response:
            handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".part")
            tmp_path = pathlib.Path(tmp_name)
            with open(handle, "wb") as tmp_file:
                shutil.copyfileobj(response, tmp_file)
        tmp_path.replace(target)
        tmp_path = None
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Failed to download {target.name} over the network. Please download "
            "the CSIC2010 files manually as described in data/README.md."
        ) from error
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _download_with_fallback(
    urls: list[str], target: pathlib.Path, *, timeout: int
) -> None:
    """Try each HTTPS candidate URL in order until one succeeds."""
    last_error: RuntimeError | None = None
    for url in urls:
        _validate_https(url)
        try:
            _download_one(url, target, timeout=timeout)
            return
        except RuntimeError as error:
            last_error = error
            logger.warning("Mirror failed for %s; trying the next one.", target.name)
    raise last_error or RuntimeError(f"No download URL available for {target.name}.")


def download_csic2010(
    dest: pathlib.Path,
    *,
    urls: dict[str, str] | None = None,
    timeout: int = 60,
) -> list[pathlib.Path]:
    """Download the three CSIC 2010 files into ``dest``.

    Args:
        dest: Destination directory; created if it does not exist.
        urls: Optional mapping of ``{filename: https_url}``. When omitted, each
            file is fetched from the GSI mirror with a GitHub mirror fallback.
            Manual placement remains an option (see ``data/README.md``).
        timeout: Per-request network timeout in seconds.

    Returns:
        The paths of the downloaded files, in the order they were requested.

    Raises:
        ValueError: If any URL is not ``https://``.
        RuntimeError: On network failure or if a downloaded file is empty.
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if urls is None:
        candidates = {
            name: [_DEFAULT_URLS[name], _FALLBACK_URLS[name]] for name in _DEFAULT_URLS
        }
    else:
        candidates = {name: [url] for name, url in urls.items()}

    saved: list[pathlib.Path] = []
    for filename, url_list in candidates.items():
        safe_name = pathlib.Path(filename).name
        if safe_name != filename:
            raise ValueError(
                f"Refusing filename with path separators: {filename!r}."
            )
        target = dest / safe_name
        _download_with_fallback(url_list, target, timeout=timeout)
        if target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file {filename} is empty. Please download the "
                "CSIC2010 files manually as described in data/README.md."
            )
        logger.info("Downloaded %s", filename)
        saved.append(target)
    return saved
