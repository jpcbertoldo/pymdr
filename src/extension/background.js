chrome.runtime.onInstalled.addListener(function() {

    // state0 = {
    // };

    chrome.declarativeContent.onPageChanged.removeRules(undefined, function() {
        chrome.declarativeContent.onPageChanged.addRules([
            {
                conditions: [new chrome.declarativeContent.PageStateMatcher()],
                actions: [new chrome.declarativeContent.ShowPageAction()]
            }
        ]);
    });
});