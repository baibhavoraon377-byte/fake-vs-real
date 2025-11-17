// background.js - service worker for context menu
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check_selection",
    title: "Check selected text with Fake-vs-Real",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "check_selection") {
    const text = info.selectionText || "";
    const url = "https://fake-vs-real-hduyc4x8emtd4ckofztnbd.streamlit.app/" + (text ? ("?text=" + encodeURIComponent(text)) : "");
    chrome.tabs.create({ url: url });
  }
});
