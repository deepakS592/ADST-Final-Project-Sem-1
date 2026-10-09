"""Compatibility helper retained for older project tests; UI is provided by Flask."""


def _driver_label(driver_code, rows, include_team=True):
    selected = rows[rows["Driver"].astype(str).str.upper().eq(str(driver_code).upper())].copy()
    names = selected["DriverName"].dropna().astype(str)
    names = names[names.str.strip().ne("")]
    numbers = selected["DriverNumber"].dropna().astype(str)
    numbers = numbers[numbers.str.strip().ne("")]
    label = names.iloc[-1] if not names.empty else str(driver_code).upper()
    if not numbers.empty:
        label += f" (#{numbers.iloc[-1]})"
    if include_team:
        teams = selected["Team"].dropna().astype(str)
        teams = teams[teams.str.strip().ne("")]
        if not teams.empty:
            label += f" — {teams.iloc[-1]}"
    return label


def main():
    from app import app
    app.run(host="127.0.0.1", port=5000, debug=False)
