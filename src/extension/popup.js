// let changeColor = document.getElementById('changeColor');

// chrome.storage.sync.get('color', function(data) {
//   changeColor.style.backgroundColor = data.color;
//   changeColor.setAttribute('value', data.color);
// });

// changeColor.onclick = function(element) {
// let color = element.target.value;
// chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
//   chrome.tabs.executeScript(
//       tabs[0].id,
//       {code: 'document.body.style.backgroundColor = "' + color + '";'});
// });
// };

// let callApi = document.getElementById('callApi');


// let goToOuput = document.getElementById('goToOuput');

function sendRequest(url) {
	const apiUrl = "http://127.0.0.1:2020/api/"
	const data = JSON.stringify({"url": url});

	var xhr = new XMLHttpRequest();
	xhr.open("POST", apiUrl, true);
	xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
	xhr.setRequestHeader('Access-Control-Allow-Origin', 'pymdr-extension');
	xhr.send(data);

	xhr.onreadystatechange = function() {
	    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
	    	const response = JSON.parse(this.response); 
	    	const output_filepath = response["output-filepath"];

			var goToOuput = document.getElementById("ouputFilepathButton");
			ouputFilepathButton.onclick = async function readClipboard () {
				if (!navigator.clipboard) {return}
				try {
					await navigator.clipboard.writeText(output_filepath);
				} catch (err) {
					console.error('Failed to copy!', err)
				}
				showCopiedMessage();
			};

			hideLoader();  
			showOuputButton();
			enableOutputButton();
	    }
	}

	xhr.ontimeout = function() {
		showOuputButton()
  		hideLoader();  
	}
}

let analyzeButton = document.getElementById("analyzeButton")
analyzeButton.onclick = function analyzeOnClick() {

	hideOuputButton();
	showLoader();

	var getUrlCallback = function (tabs) {
		this.url = tabs[0].url;
		sendRequest(this.url);
	}.bind(this);

	chrome.tabs.query(
		{'active': true, 'lastFocusedWindow': true}, getUrlCallback
	);
}

function showOuputButton() {
  document.getElementById("outputButtonWrapper").style.display = "block";
}

function hideOuputButton() {
  document.getElementById("outputButtonWrapper").style.display = "none";
}

function enableOutputButton() {
  document.getElementById("ouputFilepathButton").disabled = false;
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

