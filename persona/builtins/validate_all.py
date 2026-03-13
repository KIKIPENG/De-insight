#!/usr/bin/env python3
"""Validate all movement JSON files against Schema v2 requirements."""

import json
import re
import sys
import os
from pathlib import Path

MOVEMENTS_DIR = Path(__file__).parent
REQUIRED_TOP_FIELDS = [
    "movement_id", "name", "domain", "period", "geography",
    "historical_context", "founders_and_masters", "core_texts",
    "core_spirit", "opposition", "problems_solved", "problems_created",
    "originality_analysis", "influence_chain", "judge_persona_seed", "writing_style"
]
REQUIRED_HIST_FIELDS = ["why_it_emerged", "social_energy", "purpose", "key_timeline"]
REQUIRED_QUOTE_FIELDS = ["text", "source", "meaning"]
BANNED_PATTERN = re.compile(r'不是.*?而是')


def validate_file(filepath):
    errors = []
    warnings = []
    fname = filepath.name

    # 1. JSON validity
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"INVALID JSON: {e}")
        return errors, warnings

    # 2. Top-level fields
    for field in REQUIRED_TOP_FIELDS:
        if field not in data:
            errors.append(f"Missing top-level field: {field}")

    # 3. historical_context sub-fields
    if "historical_context" in data:
        hc = data["historical_context"]
        for field in REQUIRED_HIST_FIELDS:
            if field not in hc:
                errors.append(f"Missing historical_context.{field}")
        if "key_timeline" in hc:
            tl = hc["key_timeline"]
            if len(tl) < 8:
                warnings.append(f"key_timeline has only {len(tl)} events (recommend 10-15)")

    # 4. founders_and_masters
    if "founders_and_masters" in data:
        masters = data["founders_and_masters"]
        if len(masters) < 3:
            warnings.append(f"Only {len(masters)} masters (recommend 3-5)")
        for m in masters:
            name = m.get("name", {}).get("en", "unknown")
            if "key_quotes" not in m:
                if "key_quote" in m:
                    errors.append(f"{name}: still has old 'key_quote' string, needs upgrade to 'key_quotes' array")
                else:
                    errors.append(f"{name}: missing key_quotes")
            else:
                quotes = m["key_quotes"]
                if not isinstance(quotes, list):
                    errors.append(f"{name}: key_quotes is not an array")
                elif len(quotes) < 3:
                    warnings.append(f"{name}: only {len(quotes)} quotes (recommend 4-6)")
                else:
                    for i, q in enumerate(quotes):
                        for qf in REQUIRED_QUOTE_FIELDS:
                            if qf not in q:
                                errors.append(f"{name} quote[{i}]: missing '{qf}'")
                        if q.get("source", "") in ["attributed", "attributed, lectures", ""]:
                            warnings.append(f"{name} quote[{i}]: weak source '{q.get('source', '')}'")

    # 5. core_spirit WHY check
    if "core_spirit" in data:
        cs = data["core_spirit"]
        for key in ["what_they_do", "what_they_refuse"]:
            items = cs.get(key, [])
            no_why = [item[:60] for item in items if "WHY" not in item and "why" not in item.lower()]
            if no_why:
                errors.append(f"core_spirit.{key}: {len(no_why)}/{len(items)} items missing WHY")

    # 6. Banned pattern check
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    matches = BANNED_PATTERN.findall(text)
    if matches:
        errors.append(f"BANNED PATTERN '不是...而是' found {len(matches)} times")

    # 7. originality_analysis completeness
    if "originality_analysis" in data:
        oa = data["originality_analysis"]
        for field in ["breakthrough_idea", "what_was_truly_new", "what_was_borrowed",
                       "creative_leap", "inspiration_potential", "originality_type"]:
            if field not in oa:
                warnings.append(f"originality_analysis missing: {field}")

    return errors, warnings


def main():
    json_files = sorted(MOVEMENTS_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found!")
        sys.exit(1)

    total_errors = 0
    total_warnings = 0

    print(f"Validating {len(json_files)} movement files...\n")
    print("=" * 70)

    for fp in json_files:
        errors, warnings = validate_file(fp)
        total_errors += len(errors)
        total_warnings += len(warnings)

        status = "PASS" if not errors else "FAIL"
        warn_str = f" ({len(warnings)} warnings)" if warnings else ""
        size_kb = fp.stat().st_size / 1024

        print(f"{'PASS' if not errors else 'FAIL'} {fp.name} ({size_kb:.0f}KB){warn_str}")

        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")

        if errors or warnings:
            print()

    print("=" * 70)
    print(f"\nTotal: {len(json_files)} files, {total_errors} errors, {total_warnings} warnings")

    if total_errors > 0:
        sys.exit(1)
    else:
        print("All files passed validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
