"""De-insight TUI stability test — 20 rounds of automated testing."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from textual.css.query import NoMatches
from tui import ChatInput

PASS = 0
FAIL = 0
ERRORS = []


def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


async def run_tests():
    from tui import DeInsightApp

    for round_num in range(1, 21):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/20")
        print(f"{'='*50}")

        # ── Test 1: App boots and composes ──
        try:
            app = DeInsightApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                # T1: App mounted
                report("App mounted", True)

                # T2: MenuBar exists
                try:
                    menu = app.query_one("#menu-bar")
                    report("MenuBar exists", menu is not None)
                except NoMatches:
                    report("MenuBar exists", False, "NoMatches")

                # T3: Input exists and focusable
                try:
                    inp = app.query_one("#chat-input", ChatInput)
                    report("Input exists", inp is not None)
                except NoMatches:
                    report("Input exists", False, "NoMatches")

                # T4: Welcome block
                try:
                    wb = app.query("WelcomeBlock")
                    report("WelcomeBlock present", len(wb) > 0)
                except Exception as e:
                    report("WelcomeBlock present", False, str(e))

                # T5: Right panel
                try:
                    rp = app.query_one("#right-panel")
                    report("Right panel exists", rp is not None)
                except NoMatches:
                    report("Right panel exists", False, "NoMatches")

                # T6: Research panel
                try:
                    rp = app.query_one("#research-panel")
                    report("Research panel exists", rp is not None)
                except NoMatches:
                    report("Research panel exists", False, "NoMatches")

                # T7: Memory panel
                try:
                    mp = app.query_one("#memory-panel")
                    report("Memory panel exists", mp is not None)
                except NoMatches:
                    report("Memory panel exists", False, "NoMatches")

                # T8: StatusBar
                try:
                    sb = app.query_one("#status-bar")
                    report("StatusBar exists", sb is not None)
                except NoMatches:
                    report("StatusBar exists", False, "NoMatches")

                # T9: Open ImportModal via ctrl+f, check back button, then dismiss
                try:
                    await pilot.press("ctrl+f")
                    await pilot.pause()
                    from panels import ImportModal
                    screens = [s for s in app.screen_stack if isinstance(s, ImportModal)]
                    has_modal = len(screens) > 0
                    report("ImportModal opens (ctrl+f)", has_modal)
                    if has_modal:
                        # Check back button exists
                        back_btns = screens[0].query(".back-btn")
                        report("ImportModal has back-btn", len(back_btns) > 0)
                        # Dismiss
                        await pilot.press("escape")
                        await pilot.pause()
                        screens2 = [s for s in app.screen_stack if isinstance(s, ImportModal)]
                        report("ImportModal dismissed (esc)", len(screens2) == 0)
                    else:
                        report("ImportModal has back-btn", False, "modal not open")
                        report("ImportModal dismissed (esc)", False, "modal not open")
                except Exception as e:
                    report("ImportModal opens (ctrl+f)", False, str(e)[:80])
                    report("ImportModal has back-btn", False, "skipped")
                    report("ImportModal dismissed (esc)", False, "skipped")

                # T10: Open SearchModal via ctrl+k, check back button, then dismiss
                try:
                    await pilot.press("ctrl+k")
                    await pilot.pause()
                    from panels import SearchModal
                    screens = [s for s in app.screen_stack if isinstance(s, SearchModal)]
                    has_modal = len(screens) > 0
                    report("SearchModal opens (ctrl+k)", has_modal)
                    if has_modal:
                        back_btns = screens[0].query(".back-btn")
                        report("SearchModal has back-btn", len(back_btns) > 0)
                        await pilot.press("escape")
                        await pilot.pause()
                        screens2 = [s for s in app.screen_stack if isinstance(s, SearchModal)]
                        report("SearchModal dismissed (esc)", len(screens2) == 0)
                    else:
                        report("SearchModal has back-btn", False, "modal not open")
                        report("SearchModal dismissed (esc)", False, "modal not open")
                except Exception as e:
                    report("SearchModal opens (ctrl+k)", False, str(e)[:80])
                    report("SearchModal has back-btn", False, "skipped")
                    report("SearchModal dismissed (esc)", False, "skipped")

                # T11: Open MemoryManageModal via ctrl+m, check back button, then dismiss
                try:
                    await pilot.press("ctrl+m")
                    await pilot.pause()
                    await pilot.pause()  # extra pause for async loading
                    from panels import MemoryManageModal
                    screens = [s for s in app.screen_stack if isinstance(s, MemoryManageModal)]
                    has_modal = len(screens) > 0
                    report("MemoryManageModal opens (ctrl+m)", has_modal)
                    if has_modal:
                        back_btns = screens[0].query(".back-btn")
                        report("MemoryManageModal has back-btn", len(back_btns) > 0)
                        # Check stats bar
                        try:
                            stats = screens[0].query_one("#mem-stats")
                            report("MemoryManageModal stats bar", stats is not None)
                        except NoMatches:
                            report("MemoryManageModal stats bar", False, "NoMatches")
                        # Check filter buttons
                        filter_btns = screens[0].query(".filter-btn")
                        report("MemoryManageModal filter btns", len(filter_btns) == 4)
                        await pilot.press("escape")
                        await pilot.pause()
                        screens2 = [s for s in app.screen_stack if isinstance(s, MemoryManageModal)]
                        report("MemoryManageModal dismissed (esc)", len(screens2) == 0)
                    else:
                        report("MemoryManageModal has back-btn", False, "modal not open")
                        report("MemoryManageModal stats bar", False, "modal not open")
                        report("MemoryManageModal filter btns", False, "modal not open")
                        report("MemoryManageModal dismissed (esc)", False, "modal not open")
                except Exception as e:
                    report("MemoryManageModal opens (ctrl+m)", False, str(e)[:80])
                    report("MemoryManageModal has back-btn", False, "skipped")
                    report("MemoryManageModal stats bar", False, "skipped")
                    report("MemoryManageModal filter btns", False, "skipped")
                    report("MemoryManageModal dismissed (esc)", False, "skipped")

                # T12: Open settings via ctrl+s, then dismiss
                try:
                    await pilot.press("ctrl+s")
                    await pilot.pause()
                    from settings import SettingsScreen
                    screens = [s for s in app.screen_stack if isinstance(s, SettingsScreen)]
                    has_modal = len(screens) > 0
                    report("SettingsScreen opens (ctrl+s)", has_modal)
                    if has_modal:
                        await pilot.press("escape")
                        await pilot.pause()
                        screens2 = [s for s in app.screen_stack if isinstance(s, SettingsScreen)]
                        report("SettingsScreen dismissed (esc)", len(screens2) == 0)
                    else:
                        report("SettingsScreen dismissed (esc)", False, "modal not open")
                except Exception as e:
                    report("SettingsScreen opens (ctrl+s)", False, str(e)[:80])
                    report("SettingsScreen dismissed (esc)", False, "skipped")

                # T13: Toggle mode via ctrl+e
                try:
                    old_mode = app.mode
                    await pilot.press("ctrl+e")
                    await pilot.pause()
                    new_mode = app.mode
                    toggled = (old_mode != new_mode)
                    report("Mode toggle (ctrl+e)", toggled, f"{old_mode} -> {new_mode}")
                    # Toggle back
                    await pilot.press("ctrl+e")
                    await pilot.pause()
                    report("Mode toggle back", app.mode == old_mode)
                except Exception as e:
                    report("Mode toggle (ctrl+e)", False, str(e)[:80])
                    report("Mode toggle back", False, "skipped")

                # T14: New chat (ctrl+n) clears messages
                try:
                    await pilot.press("ctrl+n")
                    await pilot.pause()
                    report("New chat (ctrl+n)", len(app.messages) == 0)
                except Exception as e:
                    report("New chat (ctrl+n)", False, str(e)[:80])

                # T15: Slash command /help
                try:
                    inp = app.query_one("#chat-input", ChatInput)
                    inp.text = "/help"
                    await pilot.press("enter")
                    await pilot.pause()
                    # Should not add to messages (slash commands are handled separately)
                    report("Slash /help handled", True)
                except Exception as e:
                    report("Slash /help handled", False, str(e)[:80])

                # T16: Rapid modal open/close cycle (stress test)
                try:
                    for _ in range(5):
                        await pilot.press("ctrl+f")
                        await pilot.pause()
                        await pilot.press("escape")
                        await pilot.pause()
                    report("Rapid ImportModal open/close x5", True)
                except Exception as e:
                    report("Rapid ImportModal open/close x5", False, str(e)[:80])

                # T17: Rapid MemoryManage open/close cycle
                try:
                    for _ in range(3):
                        await pilot.press("ctrl+m")
                        await pilot.pause()
                        await pilot.press("escape")
                        await pilot.pause()
                    report("Rapid MemoryManage open/close x3", True)
                except Exception as e:
                    report("Rapid MemoryManage open/close x3", False, str(e)[:80])

                # T18: Open ImportModal and use back button (simulate via Button.press)
                try:
                    await pilot.press("ctrl+f")
                    await pilot.pause()
                    from panels import ImportModal
                    screens = [s for s in app.screen_stack if isinstance(s, ImportModal)]
                    if screens:
                        back_btns = list(screens[0].query(".back-btn"))
                        if back_btns:
                            back_btns[0].press()
                            await pilot.pause()
                            screens2 = [s for s in app.screen_stack if isinstance(s, ImportModal)]
                            report("Back button dismisses ImportModal", len(screens2) == 0)
                        else:
                            report("Back button dismisses ImportModal", False, "no back-btn found")
                    else:
                        report("Back button dismisses ImportModal", False, "modal not open")
                except Exception as e:
                    report("Back button dismisses ImportModal", False, str(e)[:80])
                    try:
                        await pilot.press("escape")
                        await pilot.pause()
                    except Exception:
                        pass

                # T19: Multiple modals don't stack incorrectly
                try:
                    await pilot.press("ctrl+f")
                    await pilot.pause()
                    await pilot.press("escape")
                    await pilot.pause()
                    await pilot.press("ctrl+k")
                    await pilot.pause()
                    await pilot.press("escape")
                    await pilot.pause()
                    await pilot.press("ctrl+m")
                    await pilot.pause()
                    await pilot.press("escape")
                    await pilot.pause()
                    # Should be back to main screen
                    stack_size = len(app.screen_stack)
                    report("Modal stack clean after open/close cycle", stack_size == 1)
                except Exception as e:
                    report("Modal stack clean after open/close cycle", False, str(e)[:80])

                # T20: Input field remains functional after modals
                try:
                    inp = app.query_one("#chat-input", ChatInput)
                    inp.text = "test message"
                    has_value = inp.text == "test message"
                    inp.text = ""
                    report("Input functional after modals", has_value)
                except Exception as e:
                    report("Input functional after modals", False, str(e)[:80])

        except Exception as e:
            report(f"Round {round_num} CRASH", False, str(e)[:120])

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print(f"{'='*50}")
    if ERRORS:
        print("\nFailures:")
        for e in ERRORS:
            print(e)
    else:
        print("\nAll tests passed!")
    return FAIL == 0


if __name__ == "__main__":
    ok = asyncio.run(run_tests())
    sys.exit(0 if ok else 1)
