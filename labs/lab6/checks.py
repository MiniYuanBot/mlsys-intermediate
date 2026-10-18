"""Provided test harness: one PASS/FAIL line per check, never stops at the first failure."""

import traceback

import torch


class Report:
    def __init__(self, part):
        self.part = part
        self.passed = 0
        self.failed = []

    def _record(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"PASS {name}", flush=True)
        else:
            self.failed.append(name)
            print(f"FAIL {name}: {detail}", flush=True)
        return ok

    def check(self, name, ok, expected="", actual=""):
        return self._record(name, bool(ok), f"expected {expected} vs actual {actual}")

    def equal(self, name, actual, expected):
        return self._record(name, actual == expected, f"expected {expected!r} vs actual {actual!r}")

    def close(self, name, actual, expected, rtol=1e-5, atol=1e-6):
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except (AssertionError, TypeError, RuntimeError) as error:
            text = " ".join(str(error).split())[:200]
            return self._record(name, False, f"expected close (rtol={rtol}, atol={atol}): {text}")
        return self._record(name, True)

    def raises(self, name, exception, fn):
        try:
            fn()
        except exception:
            return self._record(name, True)
        except NotImplementedError:
            raise
        except Exception as error:  # noqa: BLE001 - report the wrong exception type
            return self._record(
                name, False, f"expected {exception.__name__} vs {type(error).__name__}: {error}"
            )
        return self._record(name, False, f"expected {exception.__name__} vs no exception")

    def group(self, name, fn):
        """Run one test function; an exception becomes a single FAIL line for it."""
        try:
            fn(self)
        except NotImplementedError as error:
            self._record(name, False, f"TODO {error} not implemented")
        except Exception as error:  # noqa: BLE001
            frame = traceback.extract_tb(error.__traceback__)[-1]
            where = f"{frame.filename.split('/')[-1]}:{frame.lineno}"
            self._record(name, False, f"raised {type(error).__name__} at {where}: {error}")

    def summary(self):
        total = self.passed + len(self.failed)
        tail = f"; failed: {', '.join(self.failed)}" if self.failed else ""
        print(f"SUMMARY {self.part}: {self.passed}/{total} checks passed{tail}", flush=True)
        return 1 if self.failed else 0
