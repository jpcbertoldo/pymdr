This is an auxiliar extension for Google Chrome that communicates with the API.

It has two functionalities:
    - call MDR on the open page
    - save the page with a ground truth number of data records

To use this, add a new extension to your chrome in developer mode.
See instructions on how to do this at the beginning (2nd step) of this tutorial:
    https://developer.chrome.com/extensions/getstarted

The extension only gets the current page's url and calls the API.
So, before using it, launch the script 'launch.sh' in src/api.