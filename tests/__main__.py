"""Run the baseline suite with color-coded screen statuses."""

import sys
import unittest
import warnings


GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


class ColoredTextTestResult(unittest.TextTestResult):
    """Color unittest's per-test status while preserving its normal output."""

    def _color(self, text, color):
        return f"{color}{text}{RESET}"

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        if self.showAll:
            self._write_status(test, self._color("ok", GREEN))

    def addSkip(self, test, reason):
        unittest.TestResult.addSkip(self, test, reason)
        if self.showAll:
            status = self._color(f"skipped {reason!r}", YELLOW)
            self._write_status(test, status)

    def addExpectedFailure(self, test, err):
        unittest.TestResult.addExpectedFailure(self, test, err)
        if self.showAll:
            self._write_status(test, self._color("expected failure", YELLOW))

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        if self.showAll:
            self._write_status(test, self._color("FAIL", RED))

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        if self.showAll:
            self._write_status(test, self._color("ERROR", RED))

    def addUnexpectedSuccess(self, test):
        unittest.TestResult.addUnexpectedSuccess(self, test)
        if self.showAll:
            self._write_status(test, self._color("unexpected success", RED))


def _colored_showwarning():
    def showwarning(message, category, filename, lineno, file=None, line=None):
        stream = file if file is not None else sys.stderr
        rendered = warnings.formatwarning(message, category, filename, lineno, line)
        stream.write(f"{YELLOW}{rendered}{RESET}")

    return showwarning


def main():
    suite = unittest.defaultTestLoader.discover("tests")
    original_showwarning = warnings.showwarning
    warnings.showwarning = _colored_showwarning()
    try:
        result = unittest.TextTestRunner(
            verbosity=2,
            resultclass=ColoredTextTestResult,
        ).run(suite)
    finally:
        warnings.showwarning = original_showwarning
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
