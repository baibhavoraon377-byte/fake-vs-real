// popup.js - moved from inline to external file to comply with CSP
const BASE_URL = "https://fake-vs-real-hduyc4x8emtd4ckofztnbd.streamlit.app/";

document.addEventListener('DOMContentLoaded', () => {
  const openBtn = document.getElementById('openBtn');
  const textInput = document.getElementById('text');

  openBtn.addEventListener('click', async () => {
    const text = textInput.value || '';
    const url = text ? (BASE_URL + '?text=' + encodeURIComponent(text)) : BASE_URL;
    chrome.tabs.create({ url: url });
    window.close();
  });

  // Try to prefill selection from the active tab
  tryPrefillSelection();
});

async function tryPrefillSelection() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) return;

    // Use scripting API to execute a function in the page and return the selection
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        try {
          return window.getSelection ? window.getSelection().toString() : '';
        } catch (e) {
          return '';
        }
      }
    });

    if (results && results.length > 0 && results[0].result) {
      const sel = results[0].result.trim();
      if (sel.length > 0) {
        document.getElementById('text').value = sel;
      }
    }
  } catch (e) {
    // ignore errors (e.g., new tab without URL, chrome:// pages, etc.)
    // console.log('prefill error', e);
  }
}
