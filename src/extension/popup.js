function sendAnalyzeRequest(url) {
	const apiUrl = "http://127.0.0.1:2020/api/";
	const data = JSON.stringify({"url": url});

	const xhr = new XMLHttpRequest();
	xhr.open("POST", apiUrl, true);
	xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
	xhr.setRequestHeader('Access-Control-Allow-Origin', 'pymdr-extension');
	xhr.send(data);

	xhr.onreadystatechange = function() {
	    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
	    	const response = JSON.parse(this.response); 
	    	const output_filepath = response["output-filepath"];

			const outputFilepathButton = document.getElementById("outputFilepathButton");
			outputFilepathButton.onclick = async function readClipboard () {
				if (!navigator.clipboard) {return}
				try {
					await navigator.clipboard.writeText(output_filepath);
				} catch (err) {
					console.error('Failed to copy!', err)
				}
				showCopiedMessage();
			};

			hideLoader();  
			showOutputButton();
			enableOutputButton();
	    }
	};

	xhr.ontimeout = function() {
		showOutputButton();
  		hideLoader();  
	}
}

let analyzeButton = document.getElementById("analyzeButton");
analyzeButton.onclick = function analyzeOnClick() {

	hideOutputButton();
	showLoader();

	const getUrlCallback = function (tabs) {
		this.url = tabs[0].url;
		sendAnalyzeRequest(this.url);
	}.bind(this);

	chrome.tabs.query(
		{'active': true, 'lastFocusedWindow': true}, getUrlCallback
	);
};

function showOutputButton() {
  document.getElementById("outputButtonWrapper").style.display = "block";
}

function hideOutputButton() {
  document.getElementById("outputButtonWrapper").style.display = "none";
}

function enableOutputButton() {
  document.getElementById("outputFilepathButton").disabled = false;
}

function showLoader() {
  document.getElementById("loader").style.display = "block";
}

function hideLoader() {
  document.getElementById("loader").style.display = "none";
}

function showCopiedMessage() {
  document.getElementById("copiedMessage").style.display = "block";
}


// save  save  save  save  save  save  save  save  save  save  save  save  save  save  save  save  save


function sendSavePageRequest(url, n_data_records) {
	const apiUrl = "http://127.0.0.1:2020/api/save_page";
	const data = JSON.stringify({"url": url, "n_data_records": n_data_records});

	const xhr = new XMLHttpRequest();
	xhr.open("POST", apiUrl, true);
	xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
	xhr.setRequestHeader('Access-Control-Allow-Origin', 'pymdr-extension');
	xhr.send(data);

	xhr.onreadystatechange = function() {
	    if (this.readyState === XMLHttpRequest.DONE ) {
	    	if (this.status === 200) {
				disableSavePageButton();
				setSaveMessage("saved")
			}
	    	else {
				// setSaveMessage("something went wrong (see console)");
				setSaveMessage(this.responseText);
			}
	    }
	};
}

let savePageButton = document.getElementById("savePageButton");
savePageButton.onclick = function savePageOnClick() {
	console.log('in savePageOnClick');
	const getUrlCallback = function (tabs) {
		this.url = tabs[0].url;
		const nDataRecords = parseInt(document.getElementById("nDataRecords").value);
		sendSavePageRequest(this.url, nDataRecords);
	}.bind(this);

	chrome.tabs.query(
		{'active': true, 'lastFocusedWindow': true}, getUrlCallback
	);
};

function disableSavePageButton() {
  document.getElementById("savePageButton").disabled = true;
}

function setSaveMessage(msg) {
	console.log('in setSaveMessage');
	document.getElementById("saveMessage").style.display = "block";
	document.getElementById("saveMessage").innerText = msg;
	document.getElementById("saveMessage").textContent = msg;
}