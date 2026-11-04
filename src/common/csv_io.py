"""Logical CSV EOF checks with portable file labels."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

CSV_TAIL_COLUMNS = ["file", "size_bytes", "status", "expected_fields", "actual_fields", "reason", "skipped_rows", "header_text", "last_line_text", "last_line_truncated"]
CSV_TAIL_BYTES = 64 * 1024

def _tail_result(check: dict, status: str, actual: int | str, reason: str) -> None:
    changed = (check["status"], check["actual_fields"], check["reason"]) != (
        status,
        actual,
        reason,
    )
    check.update(status=status, actual_fields=actual, reason=reason)
    if changed:
        prefix = "WARNING" if status != "complete" else "CSV tail"
        print(
            f"{prefix}: {check['file']}: {status} ({reason or 'field count matches'})"
        )


def _check_csv_tail(path: Path, *, display: str | None = None) -> dict[str, Any]:
    """Inspect bounded input bytes; large logical boundaries need reader confirmation."""
    check = dict(
        file=display or path.name,
        size_bytes="",
        status="pending",
        expected_fields="",
        actual_fields="",
        reason="",
        skipped_rows=0,
        header_text="",
        last_line_text="",
        last_line_truncated=False,
    )
    if not path.is_file():
        _tail_result(check, "missing", "", "missing_file")
        return check
    check["size_bytes"] = path.stat().st_size
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(iter(stream.readline, ""), strict=True), [])
            header_end = stream.tell()
        with path.open("rb") as stream:
            check["header_text"] = (
                stream.read(header_end).decode("utf-8-sig").rstrip("\r\n")
            )
            # Include the preceding byte to recognize a line exactly at the bound.
            tail_start = max(0, check["size_bytes"] - CSV_TAIL_BYTES - 1)
            stream.seek(tail_start)
            tail = stream.read().rstrip(b"\r\n")
        separator = tail.rfind(b"\n")
        last_line = tail[separator + 1 :]
        check["last_line_truncated"] = tail_start > 0 and separator < 0
        if check["last_line_truncated"]:
            last_line = last_line[-CSV_TAIL_BYTES:]
        check["last_line_text"] = last_line.decode("utf-8", errors="replace")
        check["expected_fields"] = len(header)
        if not header:
            _tail_result(check, "unreadable", "", "empty_or_missing_header")
        elif check["size_bytes"] <= CSV_TAIL_BYTES:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = _CsvDictReader(stream, check)
                list(reader)
            # Inspection identifies a bad tail but does not consume application rows.
            check["skipped_rows"] = 0
        else:
            try:
                check["actual_fields"] = len(
                    next(csv.reader([last_line.decode("utf-8")], strict=True))
                )
            except (csv.Error, UnicodeError, StopIteration):
                pass
            # A physical line can belong to a quoted field opened outside this suffix.
            _tail_result(
                check,
                "unverified",
                check["actual_fields"],
                "logical_record_boundary_requires_reader_confirmation",
            )
    except (OSError, UnicodeError, csv.Error) as error:
        _tail_result(check, "unreadable", "", type(error).__name__)
    return check


class _CsvDictReader(csv.DictReader):
    """Skip only a structurally incomplete record consumed at the actual EOF."""

    def __init__(self, stream: Any, check: dict) -> None:
        # readline preserves tell(), unlike TextIOWrapper's iteration protocol.
        super().__init__(iter(stream.readline, ""), strict=True)
        self.stream, self.check = stream, check
        self.last_fields: int | str = ""

    def __next__(self) -> dict:
        try:
            row = super().__next__()
        except csv.Error as error:
            if (
                str(error) != "unexpected end of data"
                or not self.check["expected_fields"]
                or not self._at_eof()
            ):
                raise
            self._skip("unterminated_quoted_record", "")
            raise StopIteration from error
        except StopIteration:
            if self.check["status"] not in {"incomplete", "invalid"}:
                _tail_result(
                    self.check,
                    "complete",
                    self.last_fields,
                    "header_only" if self.last_fields == "" else "",
                )
            raise
        expected = len(self.fieldnames or [])
        actual = sum(row.get(name) is not None for name in self.fieldnames or [])
        actual += len(row.get(None, []))
        self.last_fields = actual
        if actual < expected and self._at_eof():
            self._skip("missing_final_fields", actual)
            raise StopIteration
        if actual < expected:
            raise csv.Error(
                f"Missing CSV fields before EOF: expected {expected}, got {actual}"
            )
        if self._at_eof():
            _tail_result(
                self.check,
                "complete" if actual == expected else "invalid",
                actual,
                "" if actual == expected else "extra_final_fields",
            )
        return row

    def _at_eof(self) -> bool:
        return self.stream.tell() == self.check["size_bytes"]

    def _skip(self, reason: str, actual: int | str) -> None:
        self.check["skipped_rows"] = 1
        _tail_result(self.check, "incomplete", actual, reason)


def _csv_reader(stream: Any, check: dict | None = None) -> _CsvDictReader:
    return _CsvDictReader(
        stream, check if check is not None else _check_csv_tail(Path(stream.name))
    )


def _validate_csv_rejects(path: Path, check: dict, rejects: list[tuple]) -> None:
    """Validate DuckDB rejects against the raw EOF suffix before publishing rows."""
    display = check['file']
    records = {(position, raw) for position, raw in rejects}
    if not records:
        if check['status'] != 'incomplete':
            _tail_result(check, 'complete', check['expected_fields'], '')
        return
    if len(records) != 1:
        raise ValueError(f'Multiple malformed CSV records in {display}; refusing to skip')
    position, reported = records.pop()
    with path.open('rb') as stream:
        stream.seek(max(0, position - 1))
        suffix = stream.read(max(CSV_TAIL_BYTES, len(reported.encode('utf-8')) * 2 + 8))
        if stream.tell() != check['size_bytes']:
            raise ValueError(f'Malformed CSV record is not at EOF in {display}')
    raw = suffix.decode('utf-8').strip('\r\n')
    reported = reported.strip('\r\n')
    alternatives = {reported, reported.replace('\\r', '\r').replace('\\n', '\n')}
    if raw not in alternatives:
        raise ValueError(f'Malformed CSV record is not the final record in {display}')
    try:
        rows = list(csv.reader(io.StringIO(raw, newline=''), strict=True))
        if len(rows) != 1 or len(rows[0]) >= check['expected_fields']:
            raise ValueError(f'Final CSV error is not an incomplete record in {display}')
        actual, reason = (len(rows[0]), 'missing_final_fields')
    except csv.Error as error:
        if str(error) != 'unexpected end of data':
            raise ValueError(f'Unsupported final CSV error in {display}') from error
        actual, reason = ('', 'unterminated_quoted_record')
    check['skipped_rows'] = 1
    _tail_result(check, 'incomplete', actual, reason)
