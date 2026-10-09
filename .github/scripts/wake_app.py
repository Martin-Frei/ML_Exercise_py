"""Open the Streamlit app in a headless browser and wake it up if it sleeps.

Runs every 6 hours via GitHub Actions (.github/workflows/keep_awake.yml).
Streamlit Community Cloud puts apps to sleep after 12 hours without visits.
"""

import os

from playwright.sync_api import sync_playwright

APP_URL = os.environ["APP_URL"]  # set in the workflow file
WAKE_BUTTON_TEXT = "Yes, get this app back up!"  # text on Streamlit's sleep page

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(APP_URL, wait_until="networkidle", timeout=120_000)

    button = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
    if button.count() > 0:
        button.click()
        page.wait_for_timeout(60_000)  # give the app time to boot
        print("App was asleep: wake button clicked")
    else:
        print("App was awake")

    browser.close()
    