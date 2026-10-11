"""Small table renderer and arrow-key picker using the standard library."""

import curses
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import math
import sys
import textwrap


_ACTIVE = None
ESCAPE_DELAY_MS = 50


def _with_escape_delay(screen, callback, *args):
    """Keep bare Escape responsive while retaining arrow-key decoding."""
    previous = curses.get_escdelay()
    curses.set_escdelay(ESCAPE_DELAY_MS)
    try:
        return callback(screen, *args)
    finally:
        curses.set_escdelay(previous)


def run_ui(callback, *args):
    """Keep one curses screen alive for an entire interactive application."""
    if _ACTIVE is not None:
        return callback(*args)
    try:
        return curses.wrapper(_with_escape_delay, _run_ui, callback, args)
    except curses.error as error:
        raise ValueError("Cannot open terminal UI. Use --help for command-line options") from error


def _run_ui(screen, callback, args):
    global _ACTIVE
    previous = _ACTIVE
    session = TerminalSession(screen)
    _ACTIVE = session
    try:
        with redirect_stdout(session), redirect_stderr(session):
            return callback(*args)
    finally:
        _ACTIVE = previous


def read_input(prompt):
    """Text entry stays in the active screen; commands retain ordinary stdin."""
    return _ACTIVE.read_input(prompt) if _ACTIVE is not None else input(prompt)


def comparison_ui():
    """Arrow controls are available inside the continuous terminal UI."""
    return _ACTIVE is not None


def tone_slider_lines(frequencies, levels, changes, adjustable):
    """Two schematic rails on one scale, relative to this pair's starting peaks.

    The five visible stops are 0.5 dB apart initially. Larger changes expand
    the shared scale; exact changes remain printed below each rail.
    """
    span = max(1.0, math.ceil(max(abs(change) for change in changes)))
    stops = [span - index * span / 2 for index in range(5)]
    markers = [min(range(5), key=lambda index: abs(stops[index] - change)) for change in changes]
    def columns(left, right):
        return left.center(29) + right.center(29)
    lines = [columns(*(f"{'>' if index == adjustable else ' '} {hz:g} Hz"
                       for index, hz in enumerate(frequencies))),
             columns(*(f"{'ADJUST' if index == adjustable else 'Reference'} | {level:.1f} dBFS"
                       for index, level in enumerate(levels)))]
    for index, stop in enumerate(stops):
        # Pad tick labels separately, then center the three-character rail
        # symbol under the frequency. Decimal and multi-digit ticks must
        # never move the rail sideways.
        rails = [f"{stop:+5.1f} {'[O]' if index == marker else ('-+-' if stop == 0 else ' | ')}"
                 .rjust(16).ljust(29) for marker in markers]
        lines.append(columns(*rails))
    lines.append(columns(*(f"{change:+.1f} dB since start" for change in changes)))
    lines.append("Changes since this pair began; rails share an automatic scale.")
    return lines


@contextmanager
def music_preview_controls():
    """Poll stop keys without leaving curses or blocking the audio player."""
    session = _ACTIVE
    if session is not None:
        session.preview_active = True
        try:
            session.render()
            yield session.poll_preview_stop
        finally:
            session.preview_active = False
            session.render()
    elif sys.stdin.isatty():
        import os
        import select
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        previous = termios.tcgetattr(descriptor)

        def poll_stop():
            if not select.select([descriptor], [], [], 0)[0]:
                return False
            key = os.read(descriptor, 1)
            if key == b"\x03":
                raise KeyboardInterrupt
            return key in (b"\r", b"\n", b"\x1b", b"s", b"S", b"q", b"Q")

        try:
            tty.setcbreak(descriptor)
            yield poll_stop
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
    else:
        yield None  # Redirected/scripted stdin retains its ordinary behavior.


def show_screen(title, sections, *, status=None, comparison=False):
    """Replace the current task screen; keep redirected output readable."""
    if _ACTIVE is not None:
        _ACTIVE.show_screen(title, sections, status, comparison=comparison)
        return
    if title is not None:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    if status:
        print("\n".join(status))
    for heading, lines in sections:
        print(f"\n{heading.upper()}")
        print("\n".join(lines))


@contextmanager
def preserve_screen():
    """Restore a task after inline help, including on cancellation or errors."""
    session = _ACTIVE
    if session is None:
        yield
        return
    fields = ("title", "status", "lines", "pending", "section_headings", "scroll", "start_at_top", "prompt", "value", "cursor", "comparison_active")
    previous = {field: getattr(session, field) for field in fields}
    try:
        yield
    finally:
        for field, value in previous.items():
            setattr(session, field, value)
        session.render()


def _place_cursor(screen, row, column, visible=False):
    """Set focus after drawing, even when cursor visibility is unsupported."""
    try:
        curses.curs_set(1 if visible else 0)
    except curses.error:
        pass
    try:
        screen.move(row, column)
    except curses.error:
        pass  # A resize may race cursor placement.


class TerminalSession:
    """In-screen output history and editable input, using only curses."""

    def __init__(self, screen):
        self.screen = screen
        self.screen.keypad(True)
        self.title = "phonix"
        self.lines = []
        self.pending = ""
        self.scroll = 0
        self.prompt = ""
        self.value = ""
        self.cursor = 0
        self.status = ()
        self.section_headings = set()
        self.start_at_top = False
        self.preview_active = False
        self.comparison_active = False

    def show_screen(self, title, sections, status, *, comparison=False):
        self.comparison_active = comparison
        if title is not None:
            self.title = title
        if status is not None:
            self.status = tuple(status)
        self.lines, self.pending, self.scroll = [], "", 0
        self.start_at_top = True
        self.section_headings = {heading.upper() for heading, _ in sections}
        for heading, lines in sections:
            if self.lines:
                self.lines.append("")
            self.lines.extend([heading.upper(), *lines])
        self.render()

    def isatty(self):
        return True

    def write(self, value):
        parts = (self.pending + value).split("\n")
        self.lines.extend(parts[:-1])
        self.lines = self.lines[-1000:]
        self.pending = parts[-1]
        self.scroll = 0
        self.render()
        return len(value)

    def flush(self):
        self.render()

    def render(self):
        screen = self.screen
        screen.erase()
        height, width = screen.getmaxyx()

        def draw(row, value, style=0):
            if row < height and width > 1:
                try:
                    screen.addnstr(row, 0, _display_text(value), width - 1, style)
                except curses.error:
                    pass  # A resize may race drawing a frame.

        if height < 7 or width < 20:
            draw(0, "Enlarge terminal, or Escape to return.")
            _place_cursor(screen, 0, 0)
            screen.refresh()
            return
        draw(0, "phonix | " + self.title, curses.A_BOLD)
        content = self.lines + ([self.pending] if self.pending else [])
        wrapped = [part for line in content for part in (textwrap.wrap(
            _display_text(line), width - 1, replace_whitespace=False, drop_whitespace=False) or [""])]
        prompt_lines = textwrap.wrap(_display_text(self.prompt), width - 1) if self.prompt else []
        prompt_lines = prompt_lines[:max(1, height // 3)]
        pinned = self.status[:max(0, height - 6 - len(prompt_lines))]
        for row, line in enumerate(pinned, 1):
            draw(row, line, curses.A_BOLD)
        start_row = 1 + len(pinned)
        available = max(1, height - 4 - len(prompt_lines) - len(pinned))
        maximum_scroll = max(0, len(wrapped) - available)
        self.scroll = maximum_scroll if self.start_at_top else min(self.scroll, maximum_scroll)
        self.start_at_top = False
        end = len(wrapped) - self.scroll
        start = max(0, end - available)
        for row, line in enumerate(wrapped[start:end], start_row):
            draw(row, line, curses.A_BOLD if line in self.section_headings else 0)
        for row, line in enumerate(prompt_lines, height - 3 - len(prompt_lines)):
            draw(row, line, curses.A_BOLD)
        if self.prompt:
            offset = max(0, self.cursor - (width - 4))
            draw(height - 3, "> " + self.value[offset:])
        if self.preview_active:
            draw(height - 2, "Music playing | Enter, Esc or s: stop preview")
            draw(height - 1, "Stopping returns to volume setup | Ctrl-C: leave calibration")
        elif self.comparison_active:
            draw(height - 2, "Up/Down: +/-0.5 dB | Enter: play pair | Esc: save and return")
            draw(height - 1, "Type command + Enter | m/g: manual/guided | PgUp/PgDn: scroll")
        else:
            draw(height - 2, "Up/Down, PgUp/PgDn: scroll output | Enter: submit | Esc: back")
            draw(height - 1, "Left/Right: move cursor | Backspace: edit | Ctrl-C: stop")
        if self.prompt:
            _place_cursor(screen, height - 3, 2 + self.cursor - offset, visible=True)
        else:
            _place_cursor(screen, 0, 0)
        screen.refresh()

    def poll_preview_stop(self):
        self.screen.timeout(0)
        try:
            try:
                key = self.screen.get_wch()
            except curses.error:
                return False
        finally:
            self.screen.timeout(-1)
        if key == "\x03":
            raise KeyboardInterrupt
        if key == curses.KEY_RESIZE:
            self.render()
        return key in ("\n", "\r", curses.KEY_ENTER, "\x1b", "s", "S", "q", "Q")

    def read_input(self, prompt):
        self.prompt, self.value, self.cursor, self.scroll = prompt, "", 0, 0
        self.start_at_top = bool(self.section_headings)
        try:
            while True:
                self.render()
                key = self.screen.get_wch()
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == "\x1b":
                    return None
                if key == curses.KEY_RESIZE:
                    continue
                height, width = self.screen.getmaxyx()
                if height < 7 or width < 20:
                    continue
                if self.comparison_active and key in (curses.KEY_UP, curses.KEY_DOWN):
                    return "up" if key == curses.KEY_UP else "down"
                if key in ("\n", "\r", curses.KEY_ENTER):
                    self.lines.append(self.prompt + self.value)
                    return self.value
                if key in (curses.KEY_UP, curses.KEY_PPAGE):
                    self.scroll += 1 if key == curses.KEY_UP else max(1, height - 6)
                elif key in (curses.KEY_DOWN, curses.KEY_NPAGE):
                    self.scroll = max(0, self.scroll - (1 if key == curses.KEY_DOWN else max(1, height - 6)))
                elif key == curses.KEY_LEFT:
                    self.cursor = max(0, self.cursor - 1)
                elif key == curses.KEY_RIGHT:
                    self.cursor = min(len(self.value), self.cursor + 1)
                elif key == curses.KEY_HOME:
                    self.cursor = 0
                elif key == curses.KEY_END:
                    self.cursor = len(self.value)
                elif key in ("\b", "\x7f", curses.KEY_BACKSPACE) and self.cursor:
                    self.value = self.value[:self.cursor - 1] + self.value[self.cursor:]
                    self.cursor -= 1
                elif key == curses.KEY_DC:
                    self.value = self.value[:self.cursor] + self.value[self.cursor + 1:]
                elif isinstance(key, str) and key.isprintable() and len(self.value) < 4096:
                    self.value = self.value[:self.cursor] + key + self.value[self.cursor:]
                    self.cursor += len(key)
        finally:
            self.prompt = ""
            try:
                curses.curs_set(0)
            except curses.error:
                pass


def _display_text(value):
    return "".join(character if character.isprintable() else "?" for character in str(value))


def table_lines(headers, rows):
    """Return aligned text suitable for a terminal or redirected output."""
    cells = [[_display_text(value) for value in row] for row in [headers, *rows]]
    widths = [max(len(row[column]) for row in cells) for column in range(len(headers))]
    lines = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in cells]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    return lines


def select_table(title, headers, rows, initial=0, *, description_column=None):
    """Pick a row, optionally showing its description below the table."""
    if not rows:
        return None
    if _ACTIVE is not None:
        _ACTIVE.comparison_active = False
        _ACTIVE.title = title
        _ACTIVE.status = ()
        _ACTIVE.section_headings = set()
        _ACTIVE.start_at_top = False
        return _select_table(_ACTIVE.screen, title, headers, rows, initial, description_column)
    try:
        return curses.wrapper(_with_escape_delay, _select_table, title, headers, rows, initial, description_column)
    except curses.error as error:
        raise ValueError("Cannot open terminal menu. Use list --plain for a text table") from error


def _select_table(screen, title, headers, rows, initial, description_column=None):
    screen.keypad(True)
    lines = table_lines(headers, rows)
    selected = max(0, min(initial, len(rows) - 1))
    top, horizontal = 0, 0
    while True:
        screen.erase()
        height, width = screen.getmaxyx()

        def draw(y, text, attribute=0):
            if y < height and width > 1:
                screen.addnstr(y, 0, _display_text(text), width - 1, attribute)

        if height < 7 or width < 20:
            draw(0, "Enlarge terminal, or q to return.")
            _place_cursor(screen, 0, 0)
        else:
            visible = height - 6
            top = min(top, selected)
            top = max(top, selected - visible + 1)
            horizontal = min(horizontal, max(0, len(lines[0]) - (width - 3)))
            draw(0, title, curses.A_BOLD)
            draw(1, "  " + lines[0][horizontal:], curses.A_BOLD)
            draw(2, "  " + lines[1][horizontal:])
            for position, index in enumerate(range(top, min(len(rows), top + visible))):
                active = index == selected
                draw(position + 3, ("> " if active else "  ") + lines[index + 2][horizontal:],
                     curses.A_REVERSE if active else 0)
            detail = (f"Selected: {rows[selected][0]}" if description_column is None
                      else str(rows[selected][description_column]))
            draw(height - 3, detail)
            draw(height - 2, f"Up/Down: select | Enter: open | q/Esc: back | {selected + 1}/{len(rows)}")
            draw(height - 1, "Left/Right: scroll columns")
            _place_cursor(screen, selected - top + 3, 0)
        screen.refresh()
        key = screen.get_wch()
        if key in ("q", "Q", "\x1b"):
            return None
        if key == "\x03":
            raise KeyboardInterrupt
        if height < 7 or width < 20:
            continue
        if key in (curses.KEY_UP, "k"):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, "j"):
            selected = min(len(rows) - 1, selected + 1)
        elif key == curses.KEY_HOME:
            selected = 0
        elif key == curses.KEY_END:
            selected = len(rows) - 1
        elif key == curses.KEY_PPAGE:
            selected = max(0, selected - visible)
        elif key == curses.KEY_NPAGE:
            selected = min(len(rows) - 1, selected + visible)
        elif key == curses.KEY_LEFT:
            horizontal = max(0, horizontal - 10)
        elif key == curses.KEY_RIGHT:
            horizontal += 10
        elif key in ("\n", "\r", curses.KEY_ENTER):
            return selected
